//! The fetch retries and the parallel-fetch limit of the media connector: a
//! fetch that fails with a timeout or a connection error is retried up to
//! `fetch_retries` times, each attempt under the full fetch budget; a status
//! the origin answered is never retried; and at most
//! `max_concurrent_fetches` fetches by URL run at once over every request a
//! connector serves.
#![allow(clippy::expect_used, clippy::print_stderr)]

use std::{
    io::Write,
    net::TcpListener,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use llm_multimodal::{
    AsyncMultiModalTracker, FetchOutcome, FetchReport, ImageFetchConfig, MediaConnector,
    MediaConnectorConfig, MediaContentPart, MediaSource, Modality,
};

use crate::common;

const TINY_PNG_B64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn tiny_png() -> Vec<u8> {
    BASE64_STANDARD.decode(TINY_PNG_B64).expect("png")
}

fn ok_response(body: &[u8]) -> Vec<u8> {
    let mut response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    response.extend_from_slice(body);
    response
}

/// A connector over `config` that may fetch from these tests' loopback
/// origins (the address policy refuses them unless private media hosts are
/// allowed).
fn connector(config: MediaConnectorConfig) -> MediaConnector {
    MediaConnector::new(
        common::http_client_builder().no_proxy(),
        MediaConnectorConfig {
            allow_private_hosts: true,
            ..config
        },
    )
    .expect("media connector")
}

/// An origin that stalls its first connection past any fetch budget and
/// answers the tiny PNG to every connection after it; the counter is the
/// requests it saw.
fn serve_stall_then_png() -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("local addr").port();
    let hits = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&hits);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let seen = Arc::clone(&seen);
            // A thread per connection: the stall of the first must not hold
            // the retry's connection waiting to be accepted.
            thread::spawn(move || {
                common::read_head(&mut stream);
                let n = seen.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    thread::sleep(Duration::from_secs(30));
                    return;
                }
                let _ = stream.write_all(&ok_response(&tiny_png()));
                let _ = stream.flush();
            });
        }
    });
    (format!("http://127.0.0.1:{port}/stall.png"), hits)
}

/// A fetch that times out once is retried under `fetch_retries = 1` and
/// succeeds on the retry, and the report names both attempts.
#[tokio::test]
async fn a_timed_out_fetch_is_retried_once() {
    let (url, hits) = serve_stall_then_png();
    let connector = connector(MediaConnectorConfig {
        fetch_timeout: Duration::from_secs(1),
        fetch_retries: 1,
        fetch_retry_backoff: Duration::from_millis(1),
        ..MediaConnectorConfig::default()
    });
    static REPORTS: Mutex<Vec<FetchReport>> = Mutex::new(Vec::new());
    fn observe(_label: &str, report: &FetchReport) {
        REPORTS.lock().expect("reports").push(*report);
    }
    let mut tracker = AsyncMultiModalTracker::new(Arc::new(connector))
        .with_fetch_observer(Arc::from("m"), observe);
    tracker
        .push_part(MediaContentPart::ImageUrl {
            url,
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        })
        .expect("a part");
    tracker
        .finalize()
        .await
        .expect("the retry fetches the image");
    assert_eq!(hits.load(Ordering::SeqCst), 2, "a stall, then the retry");
    let reports = REPORTS.lock().expect("reports");
    let report = reports
        .iter()
        .find(|report| report.outcome == FetchOutcome::Ok);
    assert_eq!(report.expect("the fetch").attempts, 2);
}

/// A status the origin answered is never retried: the fetch of a 404 fails
/// at once, whatever `fetch_retries` allows.
#[tokio::test]
async fn a_status_error_is_never_retried() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("local addr").port();
    let hits = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&hits);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            common::read_head(&mut stream);
            seen.fetch_add(1, Ordering::SeqCst);
            let _ = stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: 7\r\n\
                  Connection: close\r\n\r\nmissing",
            );
            let _ = stream.flush();
        }
    });
    let connector = connector(MediaConnectorConfig {
        fetch_timeout: Duration::from_secs(5),
        fetch_retries: 3,
        fetch_retry_backoff: Duration::from_millis(1),
        ..MediaConnectorConfig::default()
    });
    let error = connector
        .fetch_image(
            MediaSource::Url(format!("http://127.0.0.1:{port}/missing.png")),
            ImageFetchConfig::default(),
        )
        .await
        .expect_err("a 404 fails");
    assert!(error.to_string().contains("404"), "{error}");
    assert_eq!(hits.load(Ordering::SeqCst), 1, "no retry of a status");
}

/// Without retries the first timeout fails the fetch, as before.
#[tokio::test]
async fn without_retries_the_first_timeout_fails() {
    let (url, hits) = serve_stall_then_png();
    let connector = connector(MediaConnectorConfig {
        fetch_timeout: Duration::from_secs(1),
        ..MediaConnectorConfig::default()
    });
    let error = connector
        .fetch_image(MediaSource::Url(url.clone()), ImageFetchConfig::default())
        .await
        .expect_err("the stall runs out the budget");
    let message = error.to_string();
    assert!(
        message.contains(&url) && message.contains("timed out after 1s"),
        "{message}"
    );
    assert_eq!(hits.load(Ordering::SeqCst), 1, "no retry is configured");
}

/// `max_concurrent_fetches = 2` holds five image fetches to two requests in
/// flight at the origin, and every fetch succeeds.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn at_most_max_concurrent_fetches_run_at_once() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("local addr").port();
    let in_flight = Arc::new(AtomicUsize::new(0));
    let max_in_flight = Arc::new(AtomicUsize::new(0));
    let served = Arc::new(AtomicUsize::new(0));
    {
        let in_flight = Arc::clone(&in_flight);
        let max_in_flight = Arc::clone(&max_in_flight);
        let served = Arc::clone(&served);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let in_flight = Arc::clone(&in_flight);
                let max_in_flight = Arc::clone(&max_in_flight);
                let served = Arc::clone(&served);
                thread::spawn(move || {
                    common::read_head(&mut stream);
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    max_in_flight.fetch_max(now, Ordering::SeqCst);
                    // Long enough that an unlimited connector would run all
                    // five fetches at once.
                    thread::sleep(Duration::from_millis(300));
                    let _ = stream.write_all(&ok_response(&tiny_png()));
                    let _ = stream.flush();
                    served.fetch_add(1, Ordering::SeqCst);
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });
    }
    let connector = connector(MediaConnectorConfig {
        fetch_timeout: Duration::from_secs(5),
        max_concurrent_fetches: Some(2),
        ..MediaConnectorConfig::default()
    });
    let mut tracker = AsyncMultiModalTracker::new(Arc::new(connector));
    for index in 0..5 {
        tracker
            .push_part(MediaContentPart::ImageUrl {
                url: format!("http://127.0.0.1:{port}/{index}.png"),
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            })
            .expect("a part");
    }
    let images = tracker
        .finalize()
        .await
        .expect("every fetch succeeds")
        .data
        .remove(&Modality::Image)
        .expect("the request carries images");
    assert_eq!(images.len(), 5);
    assert_eq!(served.load(Ordering::SeqCst), 5, "every image was fetched");
    assert!(
        max_in_flight.load(Ordering::SeqCst) <= 2,
        "in flight at the origin: {}",
        max_in_flight.load(Ordering::SeqCst)
    );
}
