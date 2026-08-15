//! Every declared CSR sweep shape group is swept somewhere in this crate.
//!
//! The substrate arm of the same contract the primitive crate carries: the shape
//! stream in `tests/support/csr_sweep/mod.rs` declares the groups, this crate's
//! matrices draw them, and a group nothing here draws fails by name. Both sides
//! are read at run time, the declared set from the table and the drawn set from
//! this crate's test sources.

#![forbid(unsafe_code)]
#![cfg(feature = "cpu-parity")]

#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;

#[test]
fn substrate_sweeps_draw_every_declared_csr_shape_group() {
    csr_sweep::assert_every_group_is_swept("vyre-libs");
}
