//! The shared image-byte cache a [`MediaConnector`](crate::MediaConnector)
//! can fetch through, as vLLM's Rust frontend runs one over Valkey: the
//! crate defines the hook and the keys; the caller's implementation owns the
//! store, its errors and its timeouts.
//!
//! Only images fetched over http(s) are cached; video and audio never touch
//! the cache. A hit decodes the cached bytes under the same rules as fetched
//! bytes, so a frame and its hash are the same whether the bytes came from
//! the cache or from the origin.

use std::{fmt::Write as _, future::Future, pin::Pin, sync::Arc};

use bytes::Bytes;
use sha2::{Digest, Sha256};
use url::Url;

/// A shared byte cache for fetched images. An implementation swallows its
/// own errors and enforces its own timeouts: any failure of the store is a
/// miss, never a failure of the request.
///
/// The methods return boxed futures rather than being `async fn` so the
/// trait stays object-safe ([`Arc<dyn MediaBytesCache>`]) without a
/// dependency on an async-trait shim.
pub trait MediaBytesCache: Send + Sync {
    /// The bytes stored under `key`, or `None` on a miss or any failure.
    fn get<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Option<Bytes>> + Send + 'a>>;

    /// Store `bytes` under `key`. The caller starts this in the background
    /// and never awaits it: a slow or failing store must not stall the
    /// request, and a failed write is a lost entry, nothing more.
    fn put<'a>(
        &'a self,
        key: String,
        bytes: Bytes,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

/// Which query parameters of an image URL the cache key drops: signed
/// origins (OSS signs its URLs with `OSSAccessKeyId`, `Expires` and
/// `Signature`, say) sign the same bytes under ever-changing parameters, so
/// the signature's parameters must not figure in the key.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum CacheKeyQueryDrop {
    /// Keep every parameter.
    #[default]
    None,
    /// Drop the whole query.
    All,
    /// Drop the named parameters (compared exactly, case included) and keep
    /// the rest, in their original order.
    Named(Vec<String>),
}

/// How the image-byte cache figured in a fetch, for a
/// [`FetchReport`](crate::FetchReport).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheOutcome {
    /// The bytes came from the cache; nothing was fetched.
    Hit,
    /// The cache did not have the bytes; they were fetched and are being
    /// stored.
    Miss,
    /// The cache was not used: none is configured, the media is not an
    /// image fetched over http(s), or a scope is required and the request
    /// gave none.
    NotUsed,
}

impl CacheOutcome {
    /// The label value: `hit`, `miss` or `not_used`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Miss => "miss",
            Self::NotUsed => "not_used",
        }
    }
}

/// The image-byte cache a connector fetches through, with its key policy.
#[derive(Clone)]
pub(crate) struct ImageByteCache {
    pub(crate) store: Arc<dyn MediaBytesCache>,
    query_drop: CacheKeyQueryDrop,
    scope_required: bool,
}

impl ImageByteCache {
    pub(crate) fn new(
        store: Arc<dyn MediaBytesCache>,
        query_drop: CacheKeyQueryDrop,
        scope_required: bool,
    ) -> Self {
        Self {
            store,
            query_drop,
            scope_required,
        }
    }

    /// The key an image URL is cached under with `scope`, or `None` when
    /// this fetch does not touch the cache: the URL is not http(s), or a
    /// scope is required and the request gave none. Where a scope is not
    /// required and none is given, the key scopes to the empty scope.
    pub(crate) fn key(&self, url: &Url, scope: Option<&str>) -> Option<String> {
        if !matches!(url.scheme(), "http" | "https") {
            return None;
        }
        let scope = match (scope, self.scope_required) {
            (Some(scope), _) => scope,
            (None, true) => return None,
            (None, false) => "",
        };
        Some(cache_key(scope, url, &self.query_drop))
    }
}

/// The key an image URL is cached under with `scope`: the hex SHA-256 of
/// the scope, `:`, then `scheme://host[:port]/path` with the query
/// parameters `query_drop` keeps, in their original order. The host is
/// lowercased and the path's case kept; the fragment is dropped. No
/// namespace prefix: the cache implementation adds its own.
fn cache_key(scope: &str, url: &Url, query_drop: &CacheKeyQueryDrop) -> String {
    let mut key = hex_sha256(scope.as_bytes());
    key.push(':');
    key.push_str(url.scheme());
    key.push_str("://");
    if let Some(host) = url.host_str() {
        key.push_str(&host.to_ascii_lowercase());
    }
    if let Some(port) = url.port() {
        let _ = write!(key, ":{port}");
    }
    key.push_str(url.path());
    if let Some(query) = kept_query(url, query_drop) {
        key.push('?');
        key.push_str(&query);
    }
    key
}

/// The query `url` keys under, with `query_drop` applied and the kept
/// parameters in their original order and spelling; `None` when nothing is
/// kept.
fn kept_query(url: &Url, query_drop: &CacheKeyQueryDrop) -> Option<String> {
    let query = url.query()?;
    match query_drop {
        CacheKeyQueryDrop::None => Some(query.to_string()),
        CacheKeyQueryDrop::All => None,
        CacheKeyQueryDrop::Named(names) => {
            let kept: Vec<&str> = query
                .split('&')
                .filter(|pair| {
                    let name = pair.split_once('=').map_or(*pair, |(name, _)| name);
                    !names.iter().any(|dropped| dropped == name)
                })
                .collect();
            (!kept.is_empty()).then(|| kept.join("&"))
        }
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_of(scope: &str, url: &str, query_drop: &CacheKeyQueryDrop) -> String {
        cache_key(scope, &Url::parse(url).expect("a url"), query_drop)
    }

    /// The key is the hex SHA-256 of the scope, then the URL's authority
    /// and path.
    #[test]
    fn the_scope_is_hashed_and_the_url_follows() {
        let key = key_of(
            "session-1",
            "http://media.example:8080/a/b.png",
            &CacheKeyQueryDrop::None,
        );
        let (scope, rest) = key.split_once(':').expect("scope, then url");
        assert_eq!(scope.len(), 64, "hex sha-256");
        assert!(scope.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(
            scope,
            hex_sha256(b"session-1"),
            "the scope is hashed, never in the clear"
        );
        assert_eq!(rest, "http://media.example:8080/a/b.png");
        // A distinct scope is a distinct key for the same URL.
        assert_ne!(
            key_of(
                "session-2",
                "http://media.example:8080/a/b.png",
                &CacheKeyQueryDrop::None
            ),
            key
        );
    }

    /// The host is lowercased and keeps a non-default port, the path keeps
    /// its case, and the fragment never figures in the key.
    #[test]
    fn host_is_lowercased_path_case_kept_fragment_dropped() {
        assert_eq!(
            key_of(
                "s",
                "http://MEDIA.Example.COM/AbC/Def.PNG#frag",
                &CacheKeyQueryDrop::None
            ),
            format!("{}:http://media.example.com/AbC/Def.PNG", hex_sha256(b"s"))
        );
        let with_port = key_of(
            "s",
            "http://media.example:8443/a.png",
            &CacheKeyQueryDrop::None,
        );
        assert!(
            with_port.ends_with(":http://media.example:8443/a.png"),
            "{with_port}"
        );
    }

    /// `All` drops the whole query; `Named` drops only the listed
    /// parameters and keeps the rest in their original order and spelling.
    #[test]
    fn the_query_policy_drops_what_it_names() {
        let url = "http://media.example/a.png?keep=1&OSSAccessKeyId=x&Expires=2&Signature=y&Keep=2";
        assert_eq!(
            key_of("s", url, &CacheKeyQueryDrop::All),
            format!("{}:http://media.example/a.png", hex_sha256(b"s"))
        );
        let dropped = CacheKeyQueryDrop::Named(vec![
            "OSSAccessKeyId".to_string(),
            "Expires".to_string(),
            "Signature".to_string(),
        ]);
        assert_eq!(
            key_of("s", url, &dropped),
            format!(
                "{}:http://media.example/a.png?keep=1&Keep=2",
                hex_sha256(b"s")
            ),
            "the kept parameters stay in order, compared exactly (Keep != keep)"
        );
        assert_eq!(
            key_of("s", url, &CacheKeyQueryDrop::None),
            format!(
                "{}:http://media.example/a.png?keep=1&OSSAccessKeyId=x&Expires=2&Signature=y&Keep=2",
                hex_sha256(b"s")
            )
        );
        // A list that drops everything leaves no query at all.
        let all_named = CacheKeyQueryDrop::Named(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(
            key_of("s", "http://media.example/a.png?a=1&b=2", &all_named),
            format!("{}:http://media.example/a.png", hex_sha256(b"s"))
        );
    }

    /// A cache that requires a scope keys nothing without one; one that
    /// does not scopes the key to the empty scope. Only http(s) URLs key.
    #[test]
    fn the_key_policy_decides_what_is_cached() {
        let url = Url::parse("http://media.example/a.png").expect("a url");
        let cache = ImageByteCache::new(Arc::new(NoCache), CacheKeyQueryDrop::None, true);
        assert_eq!(cache.key(&url, None), None);
        assert!(cache.key(&url, Some("s")).is_some());
        let cache = ImageByteCache::new(Arc::new(NoCache), CacheKeyQueryDrop::None, false);
        assert_eq!(
            cache.key(&url, None),
            Some(format!("{}:http://media.example/a.png", hex_sha256(b"")))
        );
        let data = Url::parse("data:image/png;base64,AAAA").expect("a data url");
        assert_eq!(cache.key(&data, Some("s")), None);
    }

    struct NoCache;

    impl MediaBytesCache for NoCache {
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
}
