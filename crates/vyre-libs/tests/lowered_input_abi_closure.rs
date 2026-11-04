//! WHY: a device route runs the program the registered optimizer and the
//! schedule legalization produce, and it feeds that program the witness list
//! the catalog declares. Both steps are semantics-preserving, so both must be
//! ABI-preserving: the buffers that take host bytes, and the ones the backend
//! allocates, are the caller's contract. A pass that reclassifies a fused
//! intermediate from backend-allocated to host-staged leaves the reference
//! interpreter answering from the semantic program while a device refuses the
//! same witness list by count, which is how `flows_to_with_sanitizer` reported
//! "expected 9 initial input buffer(s) but received 8" from a fixture the
//! reference evaluated.
//!
//! Closes: the host input and backend-output ABI across `optimize` and
//! `lower_logical_schedule`, over every catalog entry that builds a program.
//! The entry set is read at run time, so a newly registered operation is
//! covered with nothing else edited.
//!
//! Does not catch: whether the lowered program computes the same values. That
//! is the parity sweep's job. It also says nothing about buffer sizes, only
//! about which declarations carry host bytes and in what order.

#![allow(deprecated)]

use vyre::ir::BufferDecl;
use vyre::Program;
use vyre_libs::operation_catalog::library_entries;

/// The dispatch ABI of a program: the host-staged inputs and the backend
/// outputs, by name, in binding order.
#[derive(PartialEq, Eq, Debug)]
struct DispatchAbi {
    inputs: Vec<String>,
    outputs: Vec<String>,
}

fn dispatch_abi(program: &Program) -> DispatchAbi {
    DispatchAbi {
        inputs: names(program, BufferDecl::consumes_host_input),
        outputs: names(program, BufferDecl::is_backend_allocated_output),
    }
}

fn names(program: &Program, admits: fn(&BufferDecl) -> bool) -> Vec<String> {
    program
        .buffers()
        .iter()
        .filter(|buffer| admits(buffer))
        .map(|buffer| buffer.name().to_string())
        .collect()
}

#[test]
fn optimizing_a_registered_operation_preserves_its_dispatch_abi() {
    let mut checked = 0usize;
    let mut drifted = Vec::new();

    for entry in library_entries() {
        let Some(build) = entry.build else {
            continue;
        };
        let program = build();
        let semantic = dispatch_abi(&program);
        let optimized = vyre_foundation::optimizer::optimize(program)
            .expect("Fix: the registered optimizer must converge on a registered operation.");
        let optimized_abi = dispatch_abi(&optimized);
        checked += 1;
        if optimized_abi != semantic {
            drifted.push(format!(
                "{}: optimizing moved the dispatch ABI from {semantic:?} to {optimized_abi:?}",
                entry.id
            ));
        }
    }

    assert!(
        drifted.is_empty(),
        "Fix: the optimizer must not move a buffer across the host boundary; a witness list built \
         from the semantic program is what a device dispatch receives. Drifted:\n{}",
        drifted.join("\n")
    );
    assert!(
        checked > 0,
        "Fix: the catalog must expose buildable operations for this closure to cover."
    );
}

#[test]
fn lowering_a_schedule_preserves_the_dispatch_abi() {
    let mut checked = 0usize;
    let mut drifted = Vec::new();

    for entry in library_entries() {
        let Some(build) = entry.build else {
            continue;
        };
        let program = build();
        let optimized = vyre_foundation::optimizer::optimize(program)
            .expect("Fix: the registered optimizer must converge on a registered operation.");
        let before = dispatch_abi(&optimized);
        let (lowered, _) =
            vyre_foundation::transform::schedule_lowering::lower_logical_schedule(optimized);
        let after = dispatch_abi(&lowered);
        checked += 1;
        if after != before {
            drifted.push(format!(
                "{}: schedule lowering moved the dispatch ABI from {before:?} to {after:?}",
                entry.id
            ));
        }
    }

    assert!(
        drifted.is_empty(),
        "Fix: schedule legalization selects a schedule and must leave the host boundary alone. \
         Drifted:\n{}",
        drifted.join("\n")
    );
    assert!(
        checked > 0,
        "Fix: the catalog must expose buildable operations for this closure to cover."
    );
}
