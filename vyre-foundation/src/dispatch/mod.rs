//! Operation schemas and opaque IR extension hooks.

/// Callable operation signature types and identifier interning.
pub mod dialect_lookup;
/// Inventory-registered extension hooks (`OpaqueExprResolver`,
/// `OpaqueNodeResolver`, etc).
pub mod extension;
