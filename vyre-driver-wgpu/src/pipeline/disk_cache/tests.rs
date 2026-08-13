//! Contract tests for the pipeline disk cache.
//!
//! Split into one module per concern so the production file stays focused on
//! production code.
#![allow(missing_docs)]

use super::*;

mod cache_key_contracts;
mod cache_miss_tracing;

/// Serializes the tests that swap the process-wide disk-cache root.
static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn fixed_digest_hex_hash_is_lowercase_and_stack_encoded() {
    let mut digest = [0_u8; 32];
    digest[0] = 0xab;
    digest[31] = 0x7f;

    let hex = hex_hash(&digest);

    assert_eq!(hex.len(), 64);
    assert!(hex.starts_with("ab00"));
    assert!(hex.ends_with("007f"));
    assert!(hex.bytes().all(|byte| byte.is_ascii_hexdigit()));
    assert_eq!(hex, hex.to_ascii_lowercase());
}
