//! Every declared CSR sweep shape group is swept somewhere in this crate.
//!
//! The shape stream is owned by `tests/support/csr_sweep/mod.rs` and shared with
//! `vyre-libs`. Sharing it removes the copies but not the hole they had: a
//! hostile shape can be declared and drawn by one crate only, and nothing fails
//! in the crate that ignores it. That is how a padded-tail frontier, the input a
//! word-granular kernel gets wrong, existed in one of the five previous copies
//! and in none of the others.
//!
//! This contract reads the declared groups from the table and the drawn groups
//! out of this crate's own test sources, both at run time, so neither side is a
//! list kept here.

#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]

#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;

#[test]
fn primitive_sweeps_draw_every_declared_csr_shape_group() {
    csr_sweep::assert_every_group_is_swept("vyre-primitives");
}
