//! Memory zeroization, cryptographic discard, and tenant cache namespace isolation (Row 119).

use super::label::{ConfidentialityLevel, TenantId};
use core::ops::{Deref, DerefMut};

/// Buffer that guarantees its allocated bytes are zeroized upon drop or reset.
#[derive(Debug, PartialEq, Eq)]
pub struct SanitizedBuffer {
    data: Vec<u8>,
}

impl SanitizedBuffer {
    /// Create a zero-initialized buffer with a given byte capacity.
    pub fn zeroed(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
        }
    }

    /// Wrap an existing vector into a sanitized buffer.
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Return underlying byte slice.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Return mutable underlying byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Explicitly overwrite all contents with zeroes.
    pub fn zeroize(&mut self) {
        self.data.fill(0);
    }

    /// Cryptographically overwrite all contents before clearing.
    pub fn crypto_discard(&mut self) {
        let seed = blake3::hash(&self.data).as_bytes().to_owned();
        for (i, byte) in self.data.iter_mut().enumerate() {
            let prng_byte = seed[i % seed.len()] ^ (i as u8);
            *byte = prng_byte;
        }
        self.zeroize();
    }
}

impl Deref for SanitizedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for SanitizedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl Drop for SanitizedBuffer {
    fn drop(&mut self) {
        self.zeroize();
    }
}

/// Tenant cache namespace isolation helper.
pub struct TenantCacheNamespace;

impl TenantCacheNamespace {
    /// Derive a tenant-isolated cache key combining TenantId, ConfidentialityLevel, and raw key digest.
    pub fn derive_key(
        tenant_id: TenantId,
        confidentiality: ConfidentialityLevel,
        raw_key: &[u8],
    ) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre_tenant_cache_namespace_v1");
        hasher.update(&tenant_id.as_u128().to_le_bytes());
        hasher.update(&[confidentiality as u8]);
        hasher.update(raw_key);
        *hasher.finalize().as_bytes()
    }
}
