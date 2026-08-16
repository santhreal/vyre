//! Binding-layout contracts for every registered graph primitive.
//!
//! WHY: a graph primitive appends its own buffers onto the read-only
//! ProgramGraph bundle, and for a long time each one restated that append by
//! hand. Restated layouts drift in ways nothing observes: two programs sized the
//! same frontier bitset differently (one floored the word count at one, one did
//! not, so a zero-node graph got a zero-word binding from one and a one-word
//! binding from the other), and a variant that inserted an extra input reused the
//! binding index another program documents as the output frontier. A backend sees
//! either of those only as a mis-sized or mis-aimed allocation, which is why the
//! per-op oracle suites stayed green through both.
//!
//! These gates read the registry at run time, so a primitive added tomorrow is
//! covered without anyone listing it here. They assert the two properties a
//! restated layout breaks: bindings are unique and contiguous from zero, and a
//! `changed` convergence flag sits read-write directly above the read-write
//! frontier it reports on.
//!
//! What they do NOT catch: a wrong count on a buffer whose size is not derivable
//! from the program alone, and a primitive whose program is never registered. A
//! count of zero is NOT a signal here, it is the documented sentinel for a
//! runtime-sized storage buffer, which `path_reconstruct` uses for its parent
//! array. The IR-validity gate in `graph_builders_emit_valid_ir.rs` covers the
//! unregistered builders.

#![cfg(feature = "graph")]

use std::collections::BTreeMap;

use vyre_foundation::ir::{BufferAccess, Program};
use vyre_foundation::operation::OperationRegistry;

/// Every registered operation whose canonical id sits in the graph domain.
fn graph_primitive_programs() -> Vec<(&'static str, Program)> {
    let programs: Vec<_> = OperationRegistry::global()
        .iter()
        .filter(|op| op.id.starts_with("vyre-primitives::graph::"))
        .filter_map(|op| op.program().map(|program| (op.id, program)))
        .collect();
    assert!(
        programs.len() >= 20,
        "Fix: only {} graph primitive programs are reachable from this test binary, so these \
         contracts would pass by reading almost nothing. Enable the features that gate the graph \
         registrations.",
        programs.len()
    );
    programs
}

/// Storage and uniform bindings are the ABI a backend allocates against. A
/// duplicate index silently aliases two buffers, and a gap means a host that
/// binds by position writes into the wrong slot.
#[test]
fn every_graph_primitive_declares_unique_contiguous_bindings() {
    let mut broken = BTreeMap::new();
    for (id, program) in graph_primitive_programs() {
        let mut bindings: Vec<u32> = program
            .buffers
            .iter()
            .filter(|buffer| buffer.access != BufferAccess::Workgroup)
            .map(|buffer| buffer.binding)
            .collect();
        bindings.sort_unstable();
        let expected: Vec<u32> = (0..bindings.len() as u32).collect();
        if bindings != expected {
            broken.insert(id, bindings);
        }
    }
    assert_eq!(
        broken,
        BTreeMap::new(),
        "Fix: these graph primitives declare bindings that are not the contiguous range \
         0..buffer_count. A repeated index aliases two buffers onto one allocation; a gap makes \
         positional host binding write to the wrong slot. Declare each buffer through the owners \
         in `graph::program_graph` rather than restating the layout."
    );
}

/// The frontier accumulator and the changed flag are one pair with one order:
/// the flag reports on the frontier directly below it. A primitive that appends
/// the pair by hand can invert them, or aim the flag at a read-only input, and
/// nothing downstream distinguishes that from a legitimate ABI.
#[test]
fn every_changed_flag_sits_above_the_frontier_it_reports_on() {
    let mut wrong = BTreeMap::new();
    for (id, program) in graph_primitive_programs() {
        let Some(changed) = program
            .buffers
            .iter()
            .find(|buffer| &*buffer.name == "changed")
        else {
            continue;
        };
        let below = program
            .buffers
            .iter()
            .find(|buffer| buffer.binding + 1 == changed.binding)
            .map(|buffer| buffer.access.clone());
        let observed = (changed.access.clone(), below, changed.count == 0);
        let expected = (
            BufferAccess::ReadWrite,
            Some(BufferAccess::ReadWrite),
            false,
        );
        if observed != expected {
            wrong.insert(id, format!("{observed:?}"));
        }
    }
    assert_eq!(
        wrong,
        BTreeMap::new(),
        "Fix: these primitives own a `changed` flag that is not read-write, is sized zero, or \
         does not sit directly above a read-write frontier. Append the pair through \
         `graph::program_graph::push_frontier_changed_buffers` instead of restating it."
    );
}

/// The frontier owner floors the word count at one, so a zero-node graph gets a
/// bindable single word rather than a zero-length allocation. Two primitives
/// sizing their own frontiers disagreed on exactly this.
#[test]
fn the_frontier_owner_floors_a_zero_node_graph_at_one_word() {
    use vyre_libs::graph::program_graph::{
        frontier_buffer, push_frontier_changed_buffers, BINDING_PRIMITIVE_START,
    };

    let empty = frontier_buffer(
        "frontier",
        BINDING_PRIMITIVE_START,
        BufferAccess::ReadWrite,
        0,
    );
    assert_eq!(empty.count, 1, "a zero-node frontier still needs one word");
    let sized = frontier_buffer(
        "frontier",
        BINDING_PRIMITIVE_START,
        BufferAccess::ReadWrite,
        33,
    );
    assert_eq!(sized.count, 2, "33 nodes need two 32-bit words");

    let mut pair = Vec::new();
    push_frontier_changed_buffers(&mut pair, "frontier_out", "changed", 0);
    assert_eq!(pair.len(), 2);
    assert_eq!(pair[0].binding, BINDING_PRIMITIVE_START);
    assert_eq!(pair[0].access, BufferAccess::ReadWrite);
    assert_eq!(pair[0].count, 1);
    assert_eq!(pair[1].binding, BINDING_PRIMITIVE_START + 1);
    assert_eq!(pair[1].access, BufferAccess::ReadWrite);
    assert_eq!(pair[1].count, 1);
}
