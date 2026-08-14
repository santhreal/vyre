//! The GPU DCE pass must hand the backend exactly the input slots its analysis
//! program declares.
//!
//! `gpu_dce` builds its persistent-BFS analysis program and then fills a fixed
//! number of input byte buffers. Those two numbers live in different files, and
//! nothing but a live GPU dispatch used to compare them. When 0.7.0 added the
//! `converged` output to the persistent-BFS layout the program grew a ninth
//! input slot, the filler kept writing eight, and every CUDA DCE test failed at
//! dispatch time with "expected 9 input buffer(s) from Program declarations but
//! received 8". These tests move that check off the GPU: they run `gpu_dce`
//! against a recording dispatcher, so a slot-count drift fails on any host, in
//! milliseconds, and names the mismatch directly.
//!
//! A ReadWrite buffer is why the count is not just "the read-only buffers":
//! backends bind ReadWrite as InputOutput, so it consumes an input slot as well
//! as an output slot. The oracle below encodes that rule independently of the
//! production code rather than importing it, so the test still fails if the
//! production rule is what drifts.
#![forbid(unsafe_code)]

use std::cell::RefCell;

use vyre_foundation::ir::{BufferAccess, Expr, MemoryKind, Node, Program};
use vyre_pass_engine::optimizer::dce_via_encoded::gpu_dce;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Number of dispatch input slots a program declares.
///
/// Independent oracle for the binding rule in `vyre-driver`: every buffer that
/// is not workgroup-shared, not persistent, and not write-only occupies an
/// input slot, and a ReadWrite buffer occupies one in addition to its output
/// slot.
fn declared_input_slots(program: &Program) -> usize {
    program
        .buffers()
        .iter()
        .filter(|buffer| {
            if buffer.kind() == MemoryKind::Shared
                || buffer.kind() == MemoryKind::Persistent
                || buffer.access() == BufferAccess::Workgroup
            {
                return false;
            }
            if buffer.is_output || buffer.pipeline_live_out {
                return false;
            }
            matches!(
                buffer.access(),
                BufferAccess::ReadOnly | BufferAccess::ReadWrite | BufferAccess::Uniform
            )
        })
        .count()
}

/// Dispatcher that records what `gpu_dce` actually handed it and answers with a
/// converged, root-only liveness closure so the pass runs to completion.
#[derive(Default)]
struct RecordingDispatcher {
    calls: RefCell<Vec<(usize, usize)>>,
    grids: RefCell<Vec<Option<[u32; 3]>>>,
}

impl ProgramDispatcher for RecordingDispatcher {
    fn dispatch(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        self.calls
            .borrow_mut()
            .push((inputs.len(), declared_input_slots(program)));
        self.grids.borrow_mut().push(grid_override);

        // Echo the caller's own frontier_out slot back as the liveness set, then
        // report changed = 0 and converged = 1 so the pass does not refuse.
        let frontier_out = inputs
            .get(6)
            .cloned()
            .ok_or_else(|| DispatchError::BackendError("missing frontier_out slot".to_string()))?;
        Ok(vec![
            frontier_out,
            0u32.to_le_bytes().to_vec(),
            1u32.to_le_bytes().to_vec(),
        ])
    }
}

fn wrapped(entry: Vec<Node>) -> Program {
    Program::wrapped(Vec::new(), [1, 1, 1], entry)
}

#[test]
fn gpu_dce_fills_every_input_slot_its_analysis_program_declares() {
    let dispatcher = RecordingDispatcher::default();
    gpu_dce(
        wrapped(vec![
            Node::let_bind("a", Expr::u32(7)),
            Node::let_bind("b", Expr::u32(9)),
        ]),
        &dispatcher,
    )
    .expect("Fix: gpu_dce must complete against a converged recording dispatcher");

    let calls = dispatcher.calls.borrow();
    assert!(
        !calls.is_empty(),
        "Fix: gpu_dce must dispatch its liveness analysis at least once."
    );
    for (index, (supplied, declared)) in calls.iter().enumerate() {
        assert_eq!(
            supplied, declared,
            "Fix: gpu_dce dispatch {index} supplied {supplied} input slot(s) for an analysis \
             program declaring {declared}. Update the slot filler in \
             vyre-pass-engine/src/optimizer/dce_via_encoded.rs to match the program."
        );
    }
}

#[test]
fn the_dce_analysis_program_declares_nine_input_slots() {
    let dispatcher = RecordingDispatcher::default();
    gpu_dce(
        wrapped(vec![Node::let_bind("a", Expr::u32(7))]),
        &dispatcher,
    )
    .expect("Fix: gpu_dce must complete against a converged recording dispatcher");

    let calls = dispatcher.calls.borrow();
    let (supplied, declared) = calls
        .first()
        .copied()
        .expect("Fix: gpu_dce must dispatch its liveness analysis at least once.");
    assert_eq!(
        declared, 9,
        "Fix: the persistent-BFS DCE layout is six read-only graph buffers plus the ReadWrite \
         frontier_out, changed and converged slots. A different count means the layout changed; \
         update this pin and the slot filler together."
    );
    assert_eq!(supplied, 9);
}

/// `gpu_dce` MUST pin its liveness analysis to exactly one workgroup.
///
/// This is a correctness contract, not a tuning preference, and it is invisible on
/// any single-workgroup-sized input, which is why it needs a test that inspects the
/// launch geometry directly rather than an output.
///
/// The analysis kernel detects growth by testing whether THIS lane's `atomic_or`
/// flipped the target bit. Exactly one lane in the grid wins any given flip, so
/// across several workgroups a duplicate group can win a discovery, leaving the one
/// group that covers the whole node range with `changed == 0`. That group records a
/// fixpoint it never reached and stops relaxing while the newly discovered node's
/// own edges are unexpanded, and `gpu_dce` then deletes live code against the
/// truncated closure. `pipeline_resident` pins the same program for the same reason.
///
/// A regression here reads as `None`, which is the whole grid, or as any `[x, _, _]`
/// with `x > 1`. Both are rejected.
#[test]
fn gpu_dce_pins_its_analysis_to_a_single_workgroup() {
    let dispatcher = RecordingDispatcher::default();
    gpu_dce(
        wrapped(vec![
            Node::let_bind("a", Expr::u32(7)),
            Node::let_bind("b", Expr::u32(9)),
        ]),
        &dispatcher,
    )
    .expect("Fix: gpu_dce must complete against a converged recording dispatcher");

    let grids = dispatcher.grids.borrow();
    assert!(
        !grids.is_empty(),
        "Fix: gpu_dce must dispatch its liveness analysis at least once."
    );
    for (index, grid) in grids.iter().enumerate() {
        assert_eq!(
            *grid,
            Some([1, 1, 1]),
            "Fix: gpu_dce dispatch {index} launched with grid {grid:?} instead of a single \
             workgroup. The analysis kernel's early exit attributes each discovery to the one \
             lane whose atomic_or flipped the bit, so splitting it across workgroups lets a \
             duplicate group steal a discovery and strand the essential group at changed = 0. \
             Pin it with Some([1, 1, 1]) in dce_via_encoded.rs."
        );
    }
}
