//! d-DNNF knowledge compilation: host compiler, device evaluator, model count.
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
//! `analysis::knowledge_compile_pass_precondition` compiles each pass
//! precondition once at startup and evaluates the DAG per Program at dispatch
//! time, which is what turns pass-precondition evaluation from exponential per
//! pass per Program into linear per gate.
//!
//! [`compile_dnnf`] and [`model_count`] run on the host. The device program
//! that evaluates a compiled DAG bottom-up over the graph wave scheduler is
//! [`crate::graph::knowledge_compile::ddnnf_evaluate`], which owns the gate
//! encoding constants and the registered op id along with it.

pub(crate) mod compile;

pub use compile::{compile_dnnf, is_satisfiable, is_tautology, model_count, DnnfDag, DnnfGate};
