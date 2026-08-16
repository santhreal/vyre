//! Primitive contract pins for structural graph, causal, and logic kernels.
//!
//! `vyre-primitives` owns these graph algorithms, every builder that emits
//! them, and every CPU oracle they are compared against. This module pins the
//! contracts the graph dispatch layer above it relies on: each builder stamps
//! its own generator on the program it returns, each checked builder rejects a
//! degenerate shape with a diagnostic instead of panicking, and each CPU oracle
//! answers the worked cases the dispatch layer's parity runs assume.
//!
//! It used to also carry a `dispatch` module and a `references` module of
//! one-line wrappers, one per primitive builder and per CPU oracle, each
//! restating the primitive's parameter list and calling it. A wrapper that
//! forwards is not an owner: it gave the algorithm a second name, no second
//! behaviour, and a second signature to keep in step. Callers name
//! `crate::graph` directly.

#[cfg(test)]
mod tests;
