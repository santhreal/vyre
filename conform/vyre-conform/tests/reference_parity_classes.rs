//! Regression contracts for the divergence classes `prove` refused to emit on.
//!
//! Each module owns one class of `(backend, op)` divergence against
//! `vyre-reference` and asserts the property at the boundary every member of
//! that class passes through, not at the op that happened to report it. The
//! rosters are read from the live registries at run time: a new op, or a new
//! backend, joins these assertions without anyone editing a list.
//!
//! What none of these catch: they judge shape, geometry and encoding, which is
//! what the classes were. Whether a kernel computes the right numbers is
//! `prove` itself, and no assertion here substitutes for running it.

#[path = "reference_parity_classes/literal_round_trip.rs"]
mod literal_round_trip;
#[path = "reference_parity_classes/output_binding_order.rs"]
mod output_binding_order;
#[path = "reference_parity_classes/transcendental_budget.rs"]
mod transcendental_budget;
#[path = "reference_parity_classes/workgroup_geometry.rs"]
mod workgroup_geometry;
