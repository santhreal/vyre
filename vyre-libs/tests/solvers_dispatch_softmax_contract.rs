//! Delegation and normalization contract for `solvers::dispatch_softmax`.
//!
//! WHY: `dispatch_softmax` is a Category A wrapper over the Category C owner
//! `vyre_primitives::math::differentiable::softmax_step`. Nothing named it:
//! no `inventory::submit!` block, no test. A wrapper with no coverage is the
//! exact place a delegation quietly becomes a copy, and a copy of a
//! fixed-point normalizer is a second owner of a rounding rule.
//!
//! Two properties are worth pinning and neither is visible from the wrapper's
//! one-line body once someone edits it. The first is that the emitted IR is
//! still the primitive's, identity included, so the crate boundary holds. The
//! second is what that IR computes: `out[j] = (pre_exp[j] << 16) / max(sum, 1)`
//! in unsigned 16.16 fixed point, including the branch that keeps an all-zero
//! input from dividing by zero, and the degenerate `n == 0` shape.
//!
//! What this does not catch: a change made identically to the wrapper and the
//! primitive. That is the primitive's own parity suite to defend.

#![forbid(unsafe_code)]

use vyre_foundation::ir::{BufferAccess, Node, Program};
use vyre_libs::solvers::dataflow_compaction_pipeline::dispatch_softmax;
use vyre_primitives::math::differentiable::{softmax_step, OP_ID};
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

const PRE_EXP: &str = "pre_exp";
const OUT: &str = "out";

/// One 16.16 fixed-point unit.
const ONE: u32 = 1 << 16;

fn run(pre_exp: &[u32]) -> Vec<u32> {
    let n = u32::try_from(pre_exp.len()).expect("fixture length fits a u32");
    let program = dispatch_softmax(PRE_EXP, OUT, n);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32_slice(pre_exp)),
            Value::from(pack_u32_slice(&vec![0u32; pre_exp.len()])),
        ],
    )
    .expect("Fix: dispatch_softmax must execute on the reference interpreter");
    assert_eq!(
        outputs.len(),
        1,
        "the normalized vector is the only published buffer"
    );
    decode_u32_le_bytes_all(&outputs[0].to_bytes())
}

fn sole_region(program: &Program) -> (&str, &[Node]) {
    match program.entry() {
        [Node::Region {
            generator, body, ..
        }] => (generator.as_str(), body.as_ref().as_slice()),
        entry => panic!("Fix: expected one wrapping region, got {entry:?}"),
    }
}

/// The wrapper emits the primitive's program, identity included, for every
/// shape it is asked for. Goes red the moment `dispatch_softmax` grows a body
/// of its own, retags the region with a `vyre-libs::` id, or starts rewriting
/// what the primitive returned.
#[test]
fn dispatch_softmax_emits_the_primitive_owner_unchanged() {
    for n in [1u32, 2, 4, 7, 256] {
        let wrapper = dispatch_softmax(PRE_EXP, OUT, n);
        let owner = softmax_step(PRE_EXP, OUT, n);
        assert_eq!(
            wrapper.fingerprint(),
            owner.fingerprint(),
            "dispatch_softmax(n = {n}) no longer emits the primitive's program"
        );
        assert_eq!(
            sole_region(&wrapper).0,
            OP_ID,
            "the region identity must stay with the Category C owner"
        );
    }
}

/// The dispatch frame: which buffers exist, in which order, with which access
/// and which element count. Goes red if a binding is transposed, if the input
/// stops being read-only, or if either count stops tracking `n`.
#[test]
fn dispatch_softmax_declares_a_read_only_input_and_a_read_write_output() {
    let n = 7;
    let program = dispatch_softmax(PRE_EXP, OUT, n);
    let buffers = program.buffers();
    assert_eq!(buffers.len(), 2, "softmax reads one vector and writes one");

    assert_eq!(buffers[0].name(), PRE_EXP);
    assert_eq!(buffers[0].binding, 0);
    assert_eq!(buffers[0].access(), BufferAccess::ReadOnly);
    assert_eq!(buffers[0].count, n);

    assert_eq!(buffers[1].name(), OUT);
    assert_eq!(buffers[1].binding, 1);
    assert_eq!(buffers[1].access(), BufferAccess::ReadWrite);
    assert_eq!(buffers[1].count, n);

    assert_eq!(
        program.workgroup_size(),
        [256, 1, 1],
        "the single-lane normalizer keeps the standard launch geometry"
    );
}

/// A uniform input normalizes to an exactly uniform distribution: four equal
/// lanes each become a quarter of one in 16.16. Goes red on an off-by-one in
/// the numerator shift, which would land on 8192 or 32768 instead.
#[test]
fn a_uniform_input_normalizes_to_equal_shares() {
    assert_eq!(run(&[1_000; 4]), vec![ONE / 4; 4]);
    assert_eq!(run(&[1_000; 2]), vec![ONE / 2; 2]);
    assert_eq!(run(&[1_000]), vec![ONE]);
}

/// Every lane is `(pre_exp[j] << 16) / sum`, truncating. Goes red if the
/// divide is rounded, if the shift moves to the denominator, or if a lane is
/// normalized against a partial sum rather than the whole vector.
#[test]
fn each_lane_is_its_share_of_the_whole_vector() {
    let input = [1_000, 1_000, 1_000, 3_000];
    let sum: u64 = input.iter().map(|value| u64::from(*value)).sum();
    let expected: Vec<u32> = input
        .iter()
        .map(|value| u32::try_from((u64::from(*value) << 16) / sum).expect("share fits a u32"))
        .collect();

    assert_eq!(expected, vec![10_922, 10_922, 10_922, 32_768]);
    assert_eq!(run(&input), expected);
}

/// The numerator is shifted before the divide, so a lane is representable
/// only while `value << 16` fits a u32. The primitive's own input format is
/// 16.16, whose unit is `1 << 16`, so the usable domain stops one below that:
/// this is the largest lane the shift survives.
///
/// Goes red if the numerator gains a wider intermediate and the boundary
/// moves, which would be a semantic change the callers of a fixed-point
/// primitive have to know about, not a silent improvement.
#[test]
fn the_largest_representable_lane_still_normalizes_exactly() {
    let max_lane = ONE - 1;
    assert_eq!(run(&[max_lane]), vec![ONE]);
    assert_eq!(run(&[max_lane, max_lane]), vec![ONE / 2; 2]);

    // One past the domain: `value << 16` wraps to zero in u32, so the whole
    // vector reads as empty and the clamp publishes zeros rather than a
    // plausible-looking wrong distribution.
    assert_eq!(run(&[ONE, ONE]), vec![0, 0]);
}

/// An all-zero input has no distribution to report, and the partition function
/// is clamped to one rather than dividing by zero. Goes red if the clamp is
/// dropped, which turns a legitimate input into a trap or a poison value.
#[test]
fn an_all_zero_input_yields_zeros_instead_of_dividing_by_zero() {
    assert_eq!(run(&[0; 4]), vec![0; 4]);
    assert_eq!(run(&[0]), vec![0]);
}

/// Zero-length is refused as IR, not as a host panic: the builder is
/// infallible so registry fixtures and generated code can call it, and the
/// refusal has to survive into the program. Goes red if the guard is removed,
/// or if it starts returning a normal program over an empty vector.
#[test]
fn a_zero_length_request_builds_a_trap_rather_than_a_dispatch() {
    let program = dispatch_softmax(PRE_EXP, OUT, 0);
    let buffers = program.buffers();
    assert_eq!(buffers.len(), 1, "the trap publishes only its output slot");
    assert_eq!(buffers[0].name(), OUT);
    assert!(
        buffers[0].is_output(),
        "the refusal is still reported through the output buffer"
    );
    assert_eq!(buffers[0].count, 1);

    let (generator, body) = sole_region(&program);
    assert_eq!(generator, OP_ID);
    assert!(
        matches!(body, [Node::Trap { .. }]),
        "a zero-length request must trap, got {body:?}"
    );
}
