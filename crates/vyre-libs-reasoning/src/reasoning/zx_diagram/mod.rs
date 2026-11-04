//! ZX-diagram rewriting test adapters.
//!
//! ZX is a graphical language for linear maps. A diagram is an undirected
//! multigraph whose vertices, called spiders, each carry a color (Z or X) and a
//! phase. The rewrite rules the optimizer applies for pattern simplification
//! live here:
//!
//! * spider fusion (S1): adjacent same-color spiders merge, summing phases.
//! * identity removal (S2): a phase-zero spider whose two edges run to two
//!   other same-color spiders is dropped, splicing those edges into one.
//! * color change (H): conjugation by a Hadamard turns a Z-spider into an
//!   X-spider and back.
//!
//! Sequential ZX-diagram rewriting witnesses are centralized in
//! `vyre_reference::composition_witness`. This module provides test-scoped
//! adapters for parity verification.

#[cfg(test)]
pub(crate) mod rewrite;
