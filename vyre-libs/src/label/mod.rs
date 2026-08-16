//! Parked composition (belongs in vyre-libs): label → NodeSet resolver.
//!
//! Given a node-tags buffer (each word = tag bitmask over a
//! registered TagFamily) and a family-mask constant, emit a NodeSet
//! bitset marking every node whose tag mask intersects the family.
//!
//! Downstream analyzer's `@family` lookup lowers to one dispatch of this
//! primitive. Labels themselves live in TOML and are merged into a
//! single per-node tag bitmap during host-side scan; once that tag
//! buffer is on device, every `@shell_family`, `@network_sink`, …
//! reference reuses the same resolver with a different mask constant.

pub mod resolve_family;

/// Shared value-to-NodeSet filter kernel.
///
/// `label::resolve_family`, `predicate::node_kind_eq` and
/// `predicate::literal_of` are the same lane-per-value bitset write under three
/// predicates. It lives under `label` rather than `predicate` because
/// `predicate` already enables `label` while the reverse edge does not exist.
#[cfg(any(feature = "label", feature = "predicate"))]
pub(crate) mod nodeset_filter;
