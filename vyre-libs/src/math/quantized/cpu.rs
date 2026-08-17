//! CPU reference oracles and packing helpers for packed INT4 primitives.

// Used only by the CPU-parity oracle fns below, which are all gated behind
// `cfg(any(test, feature = "cpu-parity"))`; gate the import to match so a
// production build (cpu-parity off) carries neither the helpers nor a dead import.
