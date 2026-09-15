//! Reusable conform lenses: ways of comparing backend output to a truth
//! oracle, one primitive per semantic.
//!
//! Every parity test runs a fixture witness, compares reference and target
//! execution, or drives a stateful operation to its registered convergence
//! bound. One module per lens, each exposing `run`, over three shared
//! concerns: how a program is executed, how a fixpoint pair is identified, and
//! how an iterative state vector is carried and projected.

pub mod backend_parity;
pub mod buffer_state;
pub mod convergence;
pub mod execution;
pub mod fixpoint;
pub mod iterative;
pub mod outcome;
pub mod witness;
