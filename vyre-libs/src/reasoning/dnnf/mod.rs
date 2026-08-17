//! d-DNNF knowledge compilation test-adapter and device evaluator bindings.
//!
//! d-DNNF is the canonical knowledge-compilation target. A Boolean formula is
//! rewritten into a directed acyclic graph of AND/OR gates over literals such
//! that:
//!
//! * **Decomposability** (D): every AND gate's children share no variables.
//! * **Determinism** (d): every OR gate's children are pairwise inconsistent.
//!
//! Those two invariants are what make model counting and weighted model
//! counting linear in the gate count rather than exponential in the variables.
//!
//! Sequential d-DNNF compilation and exact model counting are centralized in
//! `vyre_reference::composition_witness`.
//!
//! The device program that evaluates a compiled DAG bottom-up over the graph
//! wave scheduler is [`crate::graph::knowledge_compile::ddnnf_evaluate`],
//! which owns the gate encoding constants and the registered op id along with it.

#[cfg(test)]
pub(crate) mod compile;
