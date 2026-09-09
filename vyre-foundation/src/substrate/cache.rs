//! Versioned cache keys and stale cache rejection for compiler substrate records.
//!
//! Every cached, serialized, or on-disk compiler shape carries an explicit schema
//! version, target fingerprint, and content digest. Stale or mismatched entries
//! fail closed rather than serving outdated data.

use core::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::views::CompilerLevelStage;

/// Default schema version for substrate cache keys.
pub const SUBSTRATE_CACHE_SCHEMA_VERSION: u16 = 1;

/// Strongly-typed versioned cache key for compiler queries and artifact slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VersionedCacheKey {
    /// Architectural compiler level owning this cached artifact or fact.
    pub level: CompilerLevelStage,
    /// Schema format version number.
    pub schema_version: u16,
    /// Cryptographic 256-bit content digest of input IR or facts.
    pub content_digest: [u8; 32],
    /// Target hardware / device fact fingerprint.
    pub target_fingerprint: u64,
}

impl VersionedCacheKey {
    /// Construct a new versioned cache key.
    #[must_use]
    pub fn new(
        level: CompilerLevelStage,
        schema_version: u16,
        content_digest: [u8; 32],
        target_fingerprint: u64,
    ) -> Self {
        Self {
            level,
            schema_version,
            content_digest,
            target_fingerprint,
        }
    }

    /// Derive a cache key from arbitrary input bytes using BLAKE3.
    pub fn from_content(
        level: CompilerLevelStage,
        schema_version: u16,
        input_bytes: &[u8],
        target_fingerprint: u64,
    ) -> Self {
        let hash = blake3::hash(input_bytes);
        Self {
            level,
            schema_version,
            content_digest: *hash.as_bytes(),
            target_fingerprint,
        }
    }
}

impl fmt::Display for VersionedCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cache[lvl{}::v{}::tgt{:016x}::{:02x}{:02x}{:02x}{:02x}]",
            self.level as u8,
            self.schema_version,
            self.target_fingerprint,
            self.content_digest[0],
            self.content_digest[1],
            self.content_digest[2],
            self.content_digest[3]
        )
    }
}

/// Errors occurring during cache entry lookup, validation, or version verification.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StaleCacheError {
    /// Cache entry has a mismatched schema version.
    #[error("Stale cache entry: expected schema version {expected}, found version {found}")]
    VersionMismatch {
        /// Expected current schema version.
        expected: u16,
        /// Version recorded in the cached record.
        found: u16,
    },
    /// Cache entry belongs to a different compiler level.
    #[error("Cache level mismatch: expected level {expected:?}, found level {found:?}")]
    LevelMismatch {
        /// Expected compiler level.
        expected: CompilerLevelStage,
        /// Recorded compiler level.
        found: CompilerLevelStage,
    },
    /// Target hardware fingerprint mismatch.
    #[error("Target fingerprint mismatch: expected target 0x{expected:016x}, found 0x{found:016x}")]
    TargetMismatch {
        /// Current target fingerprint.
        expected: u64,
        /// Fingerprint recorded in the cached record.
        found: u64,
    },
    /// Corrupt cache payload or failed integrity check.
    #[error("Corrupt cache payload: {0}")]
    CorruptPayload(String),
}

/// Version-checked cache record wrapping an arbitrary payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionedCacheEntry<T> {
    /// Versioned key used to index this entry.
    pub key: VersionedCacheKey,
    /// Cached payload data.
    pub payload: T,
    /// Monotonic engine revision at the time of creation.
    pub created_revision: u64,
}

impl<T> VersionedCacheEntry<T> {
    /// Construct a new versioned cache entry.
    pub fn new(key: VersionedCacheKey, payload: T, created_revision: u64) -> Self {
        Self {
            key,
            payload,
            created_revision,
        }
    }

    /// Validate that the entry matches the expected schema version and return the payload.
    ///
    /// Rejects stale entries fail-closed.
    pub fn validate(
        &self,
        expected_version: u16,
        expected_target: u64,
    ) -> Result<&T, StaleCacheError> {
        if self.key.schema_version != expected_version {
            return Err(StaleCacheError::VersionMismatch {
                expected: expected_version,
                found: self.key.schema_version,
            });
        }
        if self.key.target_fingerprint != expected_target {
            return Err(StaleCacheError::TargetMismatch {
                expected: expected_target,
                found: self.key.target_fingerprint,
            });
        }
        Ok(&self.payload)
    }
}
