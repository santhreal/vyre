//! Contract tests for reference interpreter execution of Tile operations.
//!
//! The value cases are `vyre_test_support::tile_programs::tile_cases`, the same
//! corpus the lowering contract lowers and simulates, so the oracle and the
//! lowering are compared on one set of programs with one set of expected
//! values. A case that only the oracle can answer, a rejection, stays here.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Ident, Layout, Node, Program, Residency,
    SubgroupReduceOp, Tile,
};
use vyre_reference::value::Value;
use vyre_test_support::tile_programs::{self, tile_cases};

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

#[test]
fn reference_eval_computes_every_tile_case() {
    for case in tile_cases() {
        let name = case.name;
        let inputs: Vec<Value> = case
            .inputs
            .iter()
            .map(|buffer| Value::from(encode_f32(buffer)))
            .collect();

        let outputs = vyre_reference::ReferenceRequest::standard(&case.program, &inputs)
            .outputs()
            .unwrap_or_else(|e| panic!("case {name} must evaluate on the oracle: {e}"));

        let actual = decode_f32(&outputs[0].to_bytes());
        assert_eq!(
            actual, case.expected,
            "case {name} oracle output did not match expected"
        );
    }
}

#[test]
fn reference_eval_rejects_elementwise_operand_that_does_not_divide_the_output() {
    // A 3-element input against a 4-element output has no broadcast, so it is
    // rejected rather than folded to whichever length happens to fit.
    let a_data = vec![10.0f32, 20.0, 30.0, 40.0];
    let b_data = vec![1.0f32, 2.0, 3.0];
    let tile_a = Tile::new(
        DataType::F32,
        vec![2, 2],
        Layout::RowMajor,
        Residency::Register,
    );
    let tile_b = Tile::new(
        DataType::F32,
        vec![3],
        Layout::RowMajor,
        Residency::Register,
    );

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
            BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::F32).with_count(3),
            BufferDecl::output("out", 2, DataType::F32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::tile_load(
                "t_a",
                tile_a,
                "a",
                vec![Expr::u32(0), Expr::u32(0)],
                Layout::RowMajor,
            ),
            Node::tile_load("t_b", tile_b, "b", vec![Expr::u32(0)], Layout::RowMajor),
            Node::tile_elementwise(
                "sum",
                vec![Ident::from("t_a"), Ident::from("t_b")],
                vec![Node::let_bind(
                    "sum",
                    Expr::add(Expr::var("t_a"), Expr::var("t_b")),
                )],
            ),
            Node::tile_store("out", vec![Expr::u32(0)], "sum"),
        ],
    );

    let err = vyre_reference::ReferenceRequest::standard(
        &program,
        &[
            Value::from(encode_f32(&a_data)),
            Value::from(encode_f32(&b_data)),
        ],
    )
    .outputs()
    .expect_err("reference_eval must reject 3-element input against 4-element output");

    let err_msg = err.to_string();
    assert!(
        err_msg.contains("does not divide output length 4"),
        "expected divisibility error, got: {err_msg}"
    );
}

// ---------------------------------------------------------------------------
// Absent and ill-formed tile access descriptions
// ---------------------------------------------------------------------------

use vyre_test_support::tile_programs::tile_access_program;

/// The refusal a tile access program produces.
fn tile_access_refusal(program: &Program) -> String {
    vyre_reference::ReferenceRequest::standard(
        program,
        &[Value::from(encode_f32(&[1.0, 2.0, 3.0, 4.0]))],
    )
    .outputs()
    .expect_err("the oracle must refuse this tile access")
    .to_string()
}

/// A tile load reads at the origin it is given, never at one it completes.
///
/// WHY: a coordinate the origin did not state read as zero, so a program that
/// computed one coordinate too few loaded the tile from the start of the buffer
/// and the oracle certified whatever was there. Too many coordinates is the same
/// defect from the other side: the origin describes a displacement along an axis
/// the tile does not have. A rank-0 tile names one element and states the one
/// index of it, so an empty origin leaves that index absent.
///
/// Validation owns this today, as V138, and the oracle carries the same rule so
/// the access cannot be made by any route that reaches the interpreter without
/// it. The case asserts the program does not execute and names one origin
/// coordinate per tile dimension, which is what both rules state, so it goes
/// red if either layer stops refusing while the other is bypassed.
///
/// This covers the description of the access. It says nothing about where the
/// elements land, which the store cases below state.
#[test]
fn a_tile_load_origin_that_does_not_match_the_tile_rank_is_refused() {
    for (extents, origin) in [
        (vec![2u32, 2], vec![Expr::u32(0)]),
        (vec![2u32, 2], Vec::new()),
        (vec![4u32], vec![Expr::u32(0), Expr::u32(0)]),
        (Vec::new(), Vec::new()),
    ] {
        let program = tile_access_program(
            extents.clone(),
            origin,
            vec![Expr::u32(0)],
            Layout::RowMajor,
        );
        let refusal = tile_access_refusal(&program);
        assert!(
            refusal.contains("origin") && refusal.contains("per tile dimension"),
            "a tile of extents {extents:?} must refuse an origin of the wrong rank, got: {refusal}"
        );
    }
}

/// A tile store writes where its origin says, or refuses.
///
/// WHY: the store writes one linear run from the first coordinate and read only
/// that coordinate, so a displacement along any later axis was dropped and the
/// elements landed where a load from the same origin does not read them. An
/// empty origin named no destination at all and wrote from index zero.
#[test]
fn a_tile_store_origin_the_linear_write_cannot_honor_is_refused() {
    let displaced = tile_access_program(
        vec![2, 2],
        vec![Expr::u32(0), Expr::u32(0)],
        vec![Expr::u32(0), Expr::u32(1)],
        Layout::RowMajor,
    );
    let refusal = tile_access_refusal(&displaced);
    assert!(
        refusal.contains("displaces axis 1 by 1"),
        "a store origin displacing a later axis must be refused, got: {refusal}"
    );

    let absent = tile_access_program(
        vec![2, 2],
        vec![Expr::u32(0), Expr::u32(0)],
        Vec::new(),
        Layout::RowMajor,
    );
    let refusal = tile_access_refusal(&absent);
    assert!(
        refusal.contains("states no origin coordinate"),
        "a store with no origin must be refused, got: {refusal}"
    );
}

/// Every element of a loaded tile holds a value the buffer supplied.
///
/// WHY: the element vector was filled with `0.0` and each loaded value was
/// written into it only when the layout's linear index fell inside the tile.
/// A layout that does not cover the tile therefore left a slot holding a float
/// the buffer never held, and every comparison downstream ran against it.
/// `Layout::linear_index` folds coordinates it cannot map to element zero, so a
/// swizzle whose permutation repeats an axis is such a layout.
#[test]
fn a_layout_that_leaves_a_tile_element_unloaded_is_refused() {
    let program = tile_access_program(
        vec![2, 2],
        vec![Expr::u32(0), Expr::u32(0)],
        vec![Expr::u32(0)],
        Layout::Swizzled {
            permutation: vec![0, 0],
            period: 2,
        },
    );
    let refusal = tile_access_refusal(&program);
    assert!(
        refusal.contains("tile layout leaves element"),
        "a layout that does not cover the tile must be refused, got: {refusal}"
    );
}

/// A tile produced by an elementwise step is a tile the next step can use.
///
/// WHY: the result was bound with no shape, so a matmul, reduce or store that
/// needed one refused a program whose tile was perfectly well formed. The shape
/// is the broadcast target's: the operand whose element count is the output
/// length has the extents the output covers.
#[test]
fn an_elementwise_result_carries_the_shape_of_its_broadcast_target() {
    let program = tile_programs::single_input_tile_program(
        2,
        tile_programs::row_max_difference_prefix()
            .into_iter()
            .chain([
                // Reducing the elementwise result is what needs its shape: the
                // reduce has to know that four elements are two rows of two.
                Node::tile_reduce("row_min", "diff", SubgroupReduceOp::Min, 1),
                Node::tile_store("out", vec![Expr::u32(0)], "row_min"),
            ])
            .collect(),
    );

    let outputs = vyre_reference::ReferenceRequest::standard(
        &program,
        &[Value::from(encode_f32(&[10.0, 20.0, 30.0, 40.0]))],
    )
    .outputs()
    .expect("an elementwise result must be usable as a tile");

    // Rows are [10, 20] and [30, 40], so the row maxima are 20 and 40, the
    // differences are [-10, 0] and [-10, 0], and each row's minimum is -10.
    assert_eq!(decode_f32(&outputs[0].to_bytes()), vec![-10.0, -10.0]);
}
