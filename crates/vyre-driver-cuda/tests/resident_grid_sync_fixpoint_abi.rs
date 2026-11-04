//! Launch-ABI contract for the device-resident grid-sync fixpoint route.
//!
//! `dispatch_resident_grid_sync_fixpoint_into` binds one resident resource per
//! program binding, positionally, against a launch signature the backend
//! derives from the same binding table. A roster that disagrees with that
//! signature by one slot does not fail: the kernel reads the next buffer in
//! line as the one before it, resolves an element count of zero, and stores
//! nothing. The result is a populated output buffer full of zeros, which every
//! shape and length check accepts.
//!
//! The program here is the real fused grid-stride tree reduction. It is the
//! smallest shipped program that carries all three binding classes at once: a
//! read-only host input, two workgroup-local scratch buffers that occupy a
//! binding slot but no launch operand, and two read-write outputs. A roster
//! built by skipping a class, or by counting one that the launch does not
//! consume, is off by one here and correct on any program without workgroup
//! memory, which is what every fake-backend test of this route uses.
//!
//! The assertion is the exact reduction total. A non-empty check passes on the
//! zero-filled buffer the defect produces.

#![cfg(feature = "device-tests")]

use vyre_driver::grid_sync::dispatch_resident_grid_sync_fixpoint_into;
use vyre_driver::{BindingPlan, BindingRole, DispatchConfig};
use vyre_driver_cuda::cuda_factory;
use vyre_libs::reduce::grid_stride_tree;

/// Elements reduced. One Mi u32 elements is the size the adaptive-routing
/// release case reduces, and it is past the single-tile form of the builder, so
/// the program fuses two passes around a whole-grid fence and therefore splits.
const COUNT: u32 = 1 << 20;

/// Workgroup width. The release case resolves the tile to the largest power of
/// two the part admits per workgroup, which is 1024 on every CUDA part, so 1024
/// is the width the shipped path lowers. It is also the width that fills the
/// subgroup-reduction scratch tile with as many subgroups as a workgroup can
/// hold, which is what exposes ordering defects between the two levels of that
/// reduction.
const TILE: u32 = 1024;

/// Reduction operand for element `index`, matching the release case generator.
fn value_at(index: u32) -> u32 {
    index.wrapping_mul(17).wrapping_add(3) & 0xff
}

fn u32_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Assemble the caller input slice list from the program's binding table.
///
/// The order and the count are read from [`BindingPlan`] rather than written
/// out here, so a binding class that starts or stops consuming a host slot
/// changes this list without any edit. A binding whose name is not in `named`
/// is a failure and not a zero-length placeholder: a placeholder is exactly the
/// silent-zeros outcome this suite exists to catch.
fn inputs_from_binding_table(
    program: &vyre::ir::Program,
    named: &[(&str, Vec<u8>)],
) -> Vec<Vec<u8>> {
    let plan = BindingPlan::build(program)
        .expect("Fix: the fused grid-stride tree program must produce a valid binding plan.");
    let mut slots: Vec<Option<Vec<u8>>> = vec![None; plan.input_indices.len()];
    for binding in &plan.bindings {
        let Some(input_index) = binding.input_index else {
            continue;
        };
        let bytes = named
            .iter()
            .find(|(name, _)| *name == binding.name.as_ref())
            .map(|(_, bytes)| bytes.clone())
            .unwrap_or_else(|| {
                panic!(
                    "Fix: program binding `{}` consumes host input slot {input_index} but this \
                     test supplied no bytes for it. Add the operand rather than binding an empty \
                     slice: an unfilled slot binds a buffer at the wrong size and the launch \
                     returns zeros instead of failing.",
                    binding.name
                )
            });
        slots[input_index] = Some(bytes);
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(input_index, bytes)| {
            bytes.unwrap_or_else(|| {
                panic!(
                    "Fix: host input slot {input_index} was left unfilled by the binding table \
                     walk. Every slot the plan declares must be claimed by exactly one binding."
                )
            })
        })
        .collect()
}

/// The resident roster the launch consumes: one resource per binding the launch
/// signature takes an operand for, in binding-plan order.
///
/// This is derived from the plan at run time and not from a written-out list of
/// buffer names, so a binding class added to the frozen `BufferAccess` contract
/// changes the expected roster here in the same commit it changes the plan.
fn expected_roster_names(program: &vyre::ir::Program) -> Vec<String> {
    let plan = BindingPlan::build(program)
        .expect("Fix: the fused grid-stride tree program must produce a valid binding plan.");
    plan.bindings
        .iter()
        .filter(|binding| binding.role != BindingRole::Shared)
        .map(|binding| binding.name.to_string())
        .collect()
}

#[test]
fn resident_grid_sync_fixpoint_reduces_the_real_large_tree_program() {
    let blocks = grid_stride_tree::grid_stride_tree_sum_u32_blocks(COUNT, TILE);
    let program = grid_stride_tree::grid_stride_tree_sum_u32("values", "out", COUNT, TILE);

    // The program must actually carry the three binding classes this contract is
    // about, otherwise the gate passes on a shape that cannot expose the defect.
    let plan = BindingPlan::build(&program)
        .expect("Fix: the fused grid-stride tree program must produce a valid binding plan.");
    assert!(
        !plan.shared_indices.is_empty(),
        "Fix: the fused grid-stride tree program declared no workgroup-local buffer, so this gate \
         no longer covers a binding class that occupies a plan slot and no launch operand. \
         Reduction tile width {TILE} must lower through workgroup scratch."
    );
    let roster = expected_roster_names(&program);
    assert!(
        roster.len() < plan.bindings.len(),
        "Fix: every binding in the fused grid-stride tree program is a launch operand, so the \
         roster and the plan can no longer disagree and this gate is vacuous."
    );

    let values: Vec<u32> = (0..COUNT).map(value_at).collect();
    let expected = u32_bytes(&[values.iter().copied().fold(0u32, u32::wrapping_add)]);
    let named = vec![
        ("values", u32_bytes(&values)),
        ("out", u32_bytes(&[0])),
        ("__out_gst_partials", u32_bytes(&vec![0; blocks as usize])),
    ];
    let inputs = inputs_from_binding_table(&program, &named);
    let input_refs: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();

    let backend =
        cuda_factory().expect("Fix: the CUDA backend factory must succeed on a GPU test host.");
    assert!(
        backend.supports_resident_dispatch(),
        "Fix: the registered CUDA backend reports no resident dispatch, so the resident grid-sync \
         fixpoint route cannot be exercised and this gate would pass vacuously."
    );

    let mut config = DispatchConfig::default();
    // Pass 1 strides the input over exactly this grid and sizes its partial
    // buffer to it, so the grid is a contract of the program and not a tuning
    // knob. Leaving it to inference spans the widest declared buffer.
    config.grid_override = Some([blocks, 1, 1]);

    let mut outputs = Vec::new();
    dispatch_resident_grid_sync_fixpoint_into(
        backend.as_ref(),
        &program,
        &input_refs,
        &config,
        &mut outputs,
    )
    .expect("Fix: the resident grid-sync fixpoint must dispatch the fused tree reduction.");

    let total = outputs
        .last()
        .map(Vec::as_slice)
        .expect("Fix: the resident grid-sync fixpoint must return one buffer per program output.");
    assert_eq!(
        total,
        expected.as_slice(),
        "Fix: the resident grid-sync fixpoint reduced {COUNT} elements to {total:02x?} instead of \
         {expected:02x?}. All zeros means the resident roster and the launch signature disagree by \
         a slot, so the kernel read the next buffer in line as the one before it and stored \
         nothing. A total that is close but high, and varies between runs, means the two levels of \
         the workgroup reduction are unordered against each other and one level summed a partial \
         in place of a lane value. The roster the launch consumes is {roster:?} against a binding \
         plan of {} binding(s).",
        plan.bindings.len()
    );
}
