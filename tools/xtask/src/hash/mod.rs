//! Dependency-free SHA-256 for cache keys.

mod sha256;
mod sha256_hex;
#[cfg(test)]
#[path = "../../tests/internal/hash/mod.rs"]
mod tests;

pub use sha256::sha256;
pub use sha256_hex::sha256_hex;
