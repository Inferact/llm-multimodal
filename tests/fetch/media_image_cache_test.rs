//! The image-byte cache hook of the media connector: an image fetched over
//! http(s) is served from the cache when it hits and stored into it when it
//! misses, keyed by scope and URL; a request without a scope skips the cache
//! where a scope is required; video and audio never touch it.
#![allow(clippy::expect_used, clippy::print_stderr)]

use std::{
    collections::HashMap,
    fmt::Write as _,
    future::Future,
    io::Write as _,
    net::TcpListener,
    pin::Pin,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use bytes::Bytes;
use llm_multimodal::{
    AsyncMultiModalTracker, CacheKeyQueryDrop, CacheOutcome, FetchReport, MediaBytesCache,
    MediaConnector, MediaConnectorConfig, MediaContentPart, MediaSource, Modality, TrackedMedia,
    VideoFetchConfig,
};
use sha2::{Digest, Sha256};

use crate::common;

const TINY_PNG_B64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn tiny_png() -> Vec<u8> {
    BASE64_STANDARD.decode(TINY_PNG_B64).expect("png")
}

/// The key the connector must compute for `scope` and `url`, by the rules
/// the crate documents: hex SHA-256 of the scope, `:`, then
/// `scheme://host[:port]/path[?kept query]`.
fn expected_key(scope: &str, url: &str) -> String {
    let mut key = String::new();
    for byte in Sha256::digest(scope.as_bytes()) {
        let _ = write!(key, "{byte:02x}");
    }
    key.push(':');
    key.push_str(url);
    key
}

/// A cache over a map, recording every `get` and `put` it is asked for.
#[derive(Default)]
struct RecordingCache {
    store: Mutex<HashMap<String, Bytes>>,
    gets: Mutex<Vec<String>>,
    puts: Mutex<Vec<(String, Bytes)>>,
}

impl RecordingCache {
    fn gets(&self) -> Vec<String> {
        self.gets.lock().expect("gets").clone()
    }

    fn puts(&self) -> Vec<(String, Bytes)> {
        self.puts.lock().expect("puts").clone()
    }

    fn put_now(&self, key: String, bytes: Bytes) {
        self.store.lock().expect("store").insert(key, bytes);
    }
}

impl MediaBytesCache for RecordingCache {
    fn get<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Option<Bytes>> + Send + 'a>> {
        Box::pin(async move {
            self.gets.lock().expect("gets").push(key.to_string());
            self.store.lock().expect("store").get(key).cloned()
        })
    }

    fn put<'a>(
        &'a self,
        key: String,
        bytes: Bytes,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            self.puts
                .lock()
                .expect("puts")
                .push((key.clone(), bytes.clone()));
            self.store.lock().expect("store").insert(key, bytes);
        })
    }
}

/// A cache whose `get` always misses: the fetch it fronts must fall back to
/// the origin.
struct AlwaysMissCache;

impl MediaBytesCache for AlwaysMissCache {
    fn get<'a>(
        &'a self,
        _key: &'a str,
    ) -> Pin<Box<dyn Future<Output = Option<Bytes>> + Send + 'a>> {
        Box::pin(async { None })
    }

    fn put<'a>(
        &'a self,
        _key: String,
        _bytes: Bytes,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async {})
    }
}

/// An origin on loopback answering every request with the tiny PNG; the
/// counter is the requests it saw.
fn serve_png() -> (String, Arc<std::sync::atomic::AtomicUsize>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("local addr").port();
    let hits = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&hits);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let seen = Arc::clone(&seen);
            thread::spawn(move || {
                common::read_head(&mut stream);
                seen.fetch_add(1, Ordering::SeqCst);
                let body = tiny_png();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\n\
                     Connection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.write_all(&body);
                let _ = stream.flush();
            });
        }
    });
    (format!("http://127.0.0.1:{port}/picture.png"), hits)
}

/// A connector over `cache` that may fetch from these tests' loopback
/// origins (the address policy refuses them unless private media hosts are
/// allowed).
fn connector_with_cache(cache: Arc<RecordingCache>) -> MediaConnector {
    MediaConnector::new(
        common::http_client_builder().no_proxy(),
        MediaConnectorConfig {
            allow_private_hosts: true,
            image_bytes_cache: Some(cache),
            ..MediaConnectorConfig::default()
        },
    )
    .expect("media connector")
}

async fn fetch_one_image(
    tracker: AsyncMultiModalTracker,
    url: String,
) -> Arc<llm_multimodal::ImageFrame> {
    let mut tracker = tracker;
    tracker
        .push_part(MediaContentPart::ImageUrl {
            url,
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        })
        .expect("a part");
    let mut images = tracker
        .finalize()
        .await
        .expect("the image resolves")
        .data
        .remove(&Modality::Image)
        .expect("the request carries an image");
    let Some(TrackedMedia::Image(frame)) = images.pop() else {
        panic!("the slot must hold an image");
    };
    frame
}

/// A cache hit decodes the cached bytes without any HTTP request, and the
/// frame's hash is the hash of a fetch of the same bytes.
#[tokio::test]
async fn a_hit_decodes_the_cached_bytes_without_a_fetch() {
    let (url, hits) = serve_png();
    let cache = Arc::new(RecordingCache::default());
    cache.put_now(expected_key("session-1", &url), Bytes::from(tiny_png()));
    let connector = connector_with_cache(Arc::clone(&cache));
    static REPORTS: Mutex<Vec<FetchReport>> = Mutex::new(Vec::new());
    fn observe(_label: &str, report: &FetchReport) {
        REPORTS.lock().expect("reports").push(*report);
    }
    let tracker = AsyncMultiModalTracker::new(Arc::new(connector))
        .with_cache_scope(Some("session-1".to_string()))
        .with_fetch_observer(Arc::from("m"), observe);
    let frame = fetch_one_image(tracker, url.clone()).await;
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "no HTTP fetch"
    );
    assert_eq!(cache.gets(), vec![expected_key("session-1", &url)]);
    assert!(cache.puts().is_empty(), "a hit stores nothing");
    assert_eq!(
        frame.hash,
        llm_multimodal::hasher::hash_image(&tiny_png()),
        "the frame is the fetch of the same bytes"
    );
    let reports = REPORTS.lock().expect("reports");
    let report = reports.last().expect("the fetch report");
    assert_eq!(report.cache, CacheOutcome::Hit);
    assert_eq!(report.attempts, 0, "a hit fetches nothing");
}

/// A cache miss fetches from the origin and stores the fetched bytes under
/// the expected key, in the background.
#[tokio::test]
async fn a_miss_fetches_and_stores_the_bytes() {
    let (url, hits) = serve_png();
    let cache = Arc::new(RecordingCache::default());
    let connector = connector_with_cache(Arc::clone(&cache));
    let tracker = AsyncMultiModalTracker::new(Arc::new(connector))
        .with_cache_scope(Some("session-1".to_string()));
    let frame = fetch_one_image(tracker, url.clone()).await;
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the origin was fetched"
    );
    assert_eq!(cache.gets(), vec![expected_key("session-1", &url)]);
    // The write-back is in the background; wait it out.
    let key = expected_key("session-1", &url);
    for _ in 0..200 {
        if !cache.puts().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        cache.puts(),
        vec![(key.clone(), Bytes::from(tiny_png()))],
        "the fetched bytes are stored under the key"
    );
    // A second fetch of the same URL and scope is the hit the store made.
    let tracker = AsyncMultiModalTracker::new(Arc::new(connector_with_cache(Arc::clone(&cache))))
        .with_cache_scope(Some("session-1".to_string()));
    let cached = fetch_one_image(tracker, url).await;
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the second fetch is a hit"
    );
    assert_eq!(cached.hash, frame.hash, "a hit hashes like the fetch");
}

/// With a scope required (the default) and none given, the cache is not
/// consulted and nothing is stored: the fetch goes to the origin.
#[tokio::test]
async fn without_a_scope_a_required_scope_skips_the_cache() {
    let (url, hits) = serve_png();
    let cache = Arc::new(RecordingCache::default());
    let connector = connector_with_cache(Arc::clone(&cache));
    let tracker = AsyncMultiModalTracker::new(Arc::new(connector));
    fetch_one_image(tracker, url).await;
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the origin was fetched"
    );
    assert!(cache.gets().is_empty(), "no get without a scope");
    assert!(cache.puts().is_empty(), "no put without a scope");
}

/// A cache that always misses falls back to fetching from the origin.
#[tokio::test]
async fn a_cache_that_misses_falls_back_to_the_origin() {
    let (url, hits) = serve_png();
    let connector = MediaConnector::new(
        common::http_client_builder().no_proxy(),
        MediaConnectorConfig {
            allow_private_hosts: true,
            image_bytes_cache: Some(Arc::new(AlwaysMissCache)),
            ..MediaConnectorConfig::default()
        },
    )
    .expect("media connector");
    let tracker = AsyncMultiModalTracker::new(Arc::new(connector))
        .with_cache_scope(Some("session-1".to_string()));
    let frame = fetch_one_image(tracker, url).await;
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(frame.raw_bytes(), tiny_png().as_slice());
}

/// Video is never cached: no get, no put, whatever the clip.
#[tokio::test]
async fn a_video_never_touches_the_cache() {
    let (url, _hits) = serve_png();
    let cache = Arc::new(RecordingCache::default());
    let connector = connector_with_cache(Arc::clone(&cache));
    // The PNG is no video: the fetch runs, the decode fails, and either way
    // the cache must not have figured in it.
    let _ = connector
        .fetch_video(MediaSource::Url(url), VideoFetchConfig::default())
        .await;
    assert!(cache.gets().is_empty(), "no get for a video");
    assert!(cache.puts().is_empty(), "no put for a video");
}

/// The query-drop policy reaches the key: with `All`, a URL whose query
/// differs hits the entry another query stored.
#[tokio::test]
async fn the_query_drop_policy_reaches_the_key() {
    let (base_url, hits) = serve_png();
    let cache = Arc::new(RecordingCache::default());
    cache.put_now(
        expected_key("session-1", &base_url),
        Bytes::from(tiny_png()),
    );
    let store: Arc<dyn MediaBytesCache> = cache.clone();
    let connector = MediaConnector::new(
        common::http_client_builder().no_proxy(),
        MediaConnectorConfig {
            allow_private_hosts: true,
            image_bytes_cache: Some(store),
            image_cache_query_drop: CacheKeyQueryDrop::All,
            ..MediaConnectorConfig::default()
        },
    )
    .expect("media connector");
    let tracker = AsyncMultiModalTracker::new(Arc::new(connector))
        .with_cache_scope(Some("session-1".to_string()));
    let frame = fetch_one_image(tracker, format!("{base_url}?Expires=1&Signature=x")).await;
    assert_eq!(
        hits.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "the signed URL hits the bare key"
    );
    assert_eq!(frame.raw_bytes(), tiny_png().as_slice());
}
