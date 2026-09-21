//! The interned identifier expression nodes name things with.
//!
//! `Ident` caches the hash of its text once, so a name used a thousand times
//! in a program is hashed once and cloned as an `Arc` bump. `Hash` delegates to
//! the `str` rather than the cached value, because `Borrow<str>` requires the
//! two to agree: a map keyed by `Ident` must still answer a `&str` lookup.

use rustc_hash::FxHasher;
use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

/// Interned identifier used by expression nodes.
///
/// `Ident` is cheap to clone and keeps expression trees from repeatedly
/// allocating owned `String` values for the same variable or buffer names.
#[derive(Clone, Eq, PartialEq)]
pub struct Ident {
    text: Arc<str>,
    hash: u64,
}

impl Ident {
    #[inline]
    fn prehash(text: &str) -> u64 {
        let mut hasher = FxHasher::default();
        text.hash(&mut hasher);
        hasher.finish()
    }

    #[must_use]
    #[inline]
    /// Construct an identifier from shared text while caching its hash once.
    pub fn new(text: Arc<str>) -> Self {
        let hash = Self::prehash(&text);
        Self { text, hash }
    }

    /// Clone the underlying interned string handle without copying UTF-8 bytes.
    #[must_use]
    #[inline]
    pub fn shared_text(&self) -> Arc<str> {
        Arc::clone(&self.text)
    }

    /// Return another identifier handle to the same interned text without
    /// reallocating text or recomputing the cached hash.
    #[must_use]
    #[inline]
    pub fn duplicate_handle(&self) -> Self {
        Self {
            text: Arc::clone(&self.text),
            hash: self.hash,
        }
    }

    /// Return the identifier text.
    #[must_use]
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Return the cached hash used by hash-map/set lookups.
    #[must_use]
    #[inline]
    pub fn cached_hash(&self) -> u64 {
        self.hash
    }
}

impl From<&str> for Ident {
    #[inline]
    fn from(value: &str) -> Self {
        Self::new(Arc::from(value))
    }
}

impl From<String> for Ident {
    #[inline]
    fn from(value: String) -> Self {
        Self::new(Arc::from(value))
    }
}

impl From<Arc<str>> for Ident {
    #[inline]
    fn from(value: Arc<str>) -> Self {
        Self::new(value)
    }
}

impl From<&String> for Ident {
    #[inline]
    fn from(value: &String) -> Self {
        Self::from(value.as_str())
    }
}

impl From<&Ident> for Ident {
    #[inline]
    fn from(value: &Ident) -> Self {
        value.clone()
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Ident").field(&self.as_str()).finish()
    }
}

impl Hash for Ident {
    /// Audit P-IDENT-BORROW (2026-04-29): hash via the underlying str so the
    /// `Hash` impl matches the `Borrow<str>` impl, preserving the
    /// `HashMap::get<Q: Borrow<K> + Hash + Eq>` invariant. The
    /// pre-fix `state.write_u64(self.hash)` produced a different u64 than
    /// `<str as Hash>::hash` for the same hasher (which writes bytes + a
    /// length terminator), so any `FxHashMap<Ident, V>::get(&str)` lookup
    /// silently missed the inserted entry. Callers that want the cached
    /// `FxHash` for a fast equality-check key call [`Ident::cached_hash`]
    /// directly.
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.text.hash(state);
    }
}

impl Deref for Ident {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl AsRef<str> for Ident {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for Ident {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for Ident {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<str> for Ident {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for Ident {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialOrd for Ident {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ident {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}
