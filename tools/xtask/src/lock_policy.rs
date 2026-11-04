//! The xtask layer's owner of lock poison policy.
//!
//! `vyre-foundation::failure_domain` states the workspace policy, and every
//! crate that may depend on a vyre crate calls it directly. This crate may not:
//! `Cargo.toml` declares that no vyre crate appears in its dependencies, so a
//! gate still resolves a checkout root while the workspace does not compile.
//! One layer-local owner is therefore the only representable form of the same
//! decision here, and `lock-poison-policy` lists this file as an owner rather
//! than exempting each call site.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

/// Take a memo, discarding it after a panic.
///
/// Every entry is recomputed from the key it is filed under, so a half-written
/// memo is dropped and refilled rather than read. Clearing the poison flag
/// keeps the cost of one panic at one recomputation instead of one per lookup
/// for the life of the process, which is the failure a silent `if let Ok` skip
/// hides: the memo stops memoizing and nothing reports it.
pub fn govern_memo<'a, K: Ord, V>(
    memo: &'a Mutex<BTreeMap<K, V>>,
    owner: &str,
    state: &str,
) -> MutexGuard<'a, BTreeMap<K, V>> {
    match memo.lock() {
        Ok(guard) => guard,
        Err(poison) => {
            eprintln!(
                "xtask: {owner} recovered a poisoned lock over {state} by discarding it. A thread \
                 panicked while that lock was held, so the memo is half written and is refilled \
                 on demand. Fix: report the earlier panic."
            );
            memo.clear_poison();
            let mut guard = poison.into_inner();
            guard.clear();
            guard
        }
    }
}
