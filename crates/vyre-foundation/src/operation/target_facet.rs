//! Validated target identity and facet registrations.

use std::borrow::Cow;

/// Validated target identity carried by target-owned facet registrations.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TargetId(Cow<'static, str>);

impl TargetId {
    /// Construct a borrowed target identity from an owner-defined stable spelling.
    ///
    /// # Errors
    ///
    /// Empty or whitespace-padded identities are rejected.
    pub const fn new(id: &'static str) -> Result<Self, &'static str> {
        if id.is_empty() || has_surrounding_ascii_whitespace(id.as_bytes()) {
            return Err("target identity must be non-empty and contain no surrounding whitespace");
        }
        Ok(Self(Cow::Borrowed(id)))
    }

    /// Construct an owned target identity from persisted or caller-supplied data.
    ///
    /// # Errors
    ///
    /// Empty or whitespace-padded identities are rejected.
    pub fn from_owned(id: String) -> Result<Self, &'static str> {
        if id.is_empty() || has_surrounding_ascii_whitespace(id.as_bytes()) {
            return Err("target identity must be non-empty and contain no surrounding whitespace");
        }
        Ok(Self(Cow::Owned(id)))
    }

    /// Return the stable owner-defined spelling.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }

    /// Construct a validated borrowed target identity for a compile-time constant.
    ///
    /// # Panics
    ///
    /// Panics when the identity is empty or has surrounding whitespace.
    #[must_use]
    pub const fn expect_valid(id: &'static str) -> Self {
        if id.is_empty() || has_surrounding_ascii_whitespace(id.as_bytes()) {
            panic!("target identity must be non-empty and contain no surrounding whitespace");
        }
        Self(Cow::Borrowed(id))
    }
}

impl serde::Serialize for TargetId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for TargetId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let id = <String as serde::Deserialize>::deserialize(deserializer)?;
        Self::from_owned(id).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for TargetId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl PartialEq<&str> for TargetId {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

const fn has_surrounding_ascii_whitespace(bytes: &[u8]) -> bool {
    matches!(bytes.first(), Some(byte) if byte.is_ascii_whitespace())
        || matches!(bytes.last(), Some(byte) if byte.is_ascii_whitespace())
}

/// Derived target-specific capability keyed by canonical semantic operation id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetOperationFacet {
    /// Canonical semantic operation id.
    pub operation_id: &'static str,
    /// Validated target identity from the concrete driver's registration.
    pub target_id: TargetId,
    /// Target facet schema version.
    pub version: u32,
}
