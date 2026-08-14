//! Test-only program composition helpers shared by parity suites.
//!
//! Compiled in two contexts, matching `scan::test_fixtures`:
//!
//! 1. `#[cfg(test)]` - always available to in-tree tests.
//! 2. `feature = "test-fixtures"` - exported to crates whose own parity
//!    suites dispatch multi-stage programs built from this crate.

use vyre_foundation::ir::Program;

/// Concatenate `programs` into one program with a shared workgroup size.
///
/// Buffers and entry nodes are appended in argument order, which is the
/// order a multi-stage parity suite dispatches them in.
#[must_use]
pub fn wrap_program_sequence(programs: &[&Program], workgroup_size: [u32; 3]) -> Program {
    let buffer_count = programs.iter().map(|program| program.buffers().len()).sum();
    let entry_count = programs.iter().map(|program| program.entry().len()).sum();
    let mut buffers = Vec::with_capacity(buffer_count);
    let mut entry = Vec::with_capacity(entry_count);

    for program in programs {
        buffers.extend_from_slice(program.buffers());
        entry.extend_from_slice(program.entry());
    }

    Program::wrapped(buffers, workgroup_size, entry)
}
