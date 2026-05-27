//! Public-facade smoke example.
//!
//! Demonstrates the consumer-facing API path:
//!   - Only imports from `vyre::*`.
//!   - No `vyre_core::*`, `vyre_foundation::*`, `vyre_driver::*`,
//!     `vyre_primitives::*` reaches.
//!
//! `scripts/check_examples_public_facade.sh` enforces this rule for
//! every example under `examples/`. New examples land green or fail
//! the gate.

use vyre::ir::Program;

fn main() {
    // Build an empty program through the public surface only. The
    // example deliberately does not exercise a backend  -  that's what
    // the three-substrate parity manifest covers. The point here is
    // that `vyre::ir::Program` is reachable without internal imports.
    let program = Program::default();
    let buffer_count = program.buffers.len();
    let node_count = program.entry.len();
    println!(
        "public-facade smoke: built an empty Program (buffers={buffer_count}, entry_nodes={node_count}) via the vyre crate alone."
    );
}
