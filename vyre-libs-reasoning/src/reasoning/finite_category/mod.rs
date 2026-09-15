//! Finite categories: Yoneda embedding, adjoint pairs, Kan extensions test adapters.
//!
//! A category here is finite: a fixed object set plus a Hom-set cardinality
//! table. Morphisms are `u32` ids, identities are implicit, and composition is
//! not modelled, because every question this module answers is a cardinality
//! question. Functors carry an object map.
//!
//! Sequential categorical mathematical witnesses are centralized in
//! `vyre_reference::composition_witness`. This module provides test-scoped
//! adapters for parity verification.

#[cfg(test)]
pub(crate) mod adjoint;
#[cfg(test)]
pub(crate) mod kan_extension;
#[cfg(test)]
pub(crate) mod yoneda;
