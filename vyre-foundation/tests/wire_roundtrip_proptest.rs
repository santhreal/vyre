//! Wire-format roundtrip proptest. The support chunk holds the shared
//! generators; the invariant cases are declared under it so they keep seeing
//! those generators.
#![allow(dead_code)]

#[path = "contract_cases/wire_roundtrip_proptest_support__extension_kind.rs"]
mod wire_roundtrip_proptest_support_extension_kind;
