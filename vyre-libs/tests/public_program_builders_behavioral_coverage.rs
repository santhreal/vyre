//! Behavioral and reference-evaluation integration tests for the public Program builders.
//!
//! Covers:
//! - `binary_broadcast_rhs` (`ElementwiseComposer::binary_broadcast_rhs`)
//! - `f32_binary` (`ElementwiseComposer::f32_binary`)
//! - `level_wave_program_with_buffers_and_op_id` (`vyre_libs::graph::level_wave::level_wave_program_with_buffers_and_op_id`)
//! - `line_index_u8_with_geometry` (`vyre_libs::text::line_index_u8_with_geometry`)
//! - `online_softmax_attention` (`vyre_libs::nn::attention::tiled_online_softmax::online_softmax_attention`)
//! - `ternary` (`ElementwiseComposer::ternary`)
//! - `u32_unary` (`ElementwiseComposer::u32_unary`)
//!
//! Each test executes the generated IR under the pure-Rust reference interpreter
//! (`vyre_reference::reference_eval`), verifying observable outputs against
//! independent references or canonical mathematical witnesses.

#![forbid(unsafe_code)]

use vyre_foundation::ir::{BufferAccess, DataType, Expr, Program};
use vyre_libs::elementwise::ElementwiseComposer;
use vyre_primitives::wire::{
    decode_f32_le_bytes_all, decode_u32_le_bytes_all, pack_f32_slice, pack_u32_slice,
};
use vyre_reference::value::Value;

fn assert_program_valid(program: &Program) {
    let errors = vyre::validate(program);
    assert!(
        errors.is_empty(),
        "Program failed validation: {:?}",
        errors
            .iter()
            .map(|e| e.message().to_string())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// 1. binary_broadcast_rhs
// ---------------------------------------------------------------------------

#[test]
fn binary_broadcast_rhs_u32_modulo_indexing() {
    let count = 8u32;
    let rhs_count = 2u32;
    let program = ElementwiseComposer::binary_broadcast_rhs(
        "vyre-libs::test::binary_broadcast_rhs_modulo",
        "lhs",
        "rhs",
        "out",
        count,
        rhs_count,
        DataType::U32,
        |i| Expr::rem(i.clone(), Expr::u32(2)),
        Expr::add,
    );
    assert_program_valid(&program);

    let lhs: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let rhs: Vec<u32> = vec![1, 2];
    let expected: Vec<u32> = lhs
        .iter()
        .enumerate()
        .map(|(i, &x)| x + rhs[i % (rhs_count as usize)])
        .collect();

    let out_idx = vyre_reference::output_index(&program, "out")
        .expect("output buffer `out` must be declared");

    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32_slice(&lhs)),
            Value::from(pack_u32_slice(&rhs)),
        ],
    )
    .expect("binary_broadcast_rhs reference evaluation must succeed");

    let actual = decode_u32_le_bytes_all(&outputs[out_idx].to_bytes());
    assert_eq!(
        actual, expected,
        "binary_broadcast_rhs must correctly broadcast RHS across modulo lanes"
    );
}

#[test]
fn binary_broadcast_rhs_f32_scalar_broadcast() {
    let count = 5u32;
    let rhs_count = 1u32;
    let program = ElementwiseComposer::binary_broadcast_rhs(
        "vyre-libs::test::binary_broadcast_rhs_scalar",
        "lhs",
        "rhs",
        "out",
        count,
        rhs_count,
        DataType::F32,
        |_i| Expr::u32(0),
        Expr::sub,
    );
    assert_program_valid(&program);

    let lhs: Vec<f32> = vec![10.5, 20.0, 30.25, 40.75, 50.125];
    let rhs: Vec<f32> = vec![0.5];
    let expected: Vec<f32> = lhs.iter().map(|&x| x - rhs[0]).collect();

    let out_idx = vyre_reference::output_index(&program, "out")
        .expect("output buffer `out` must be declared");

    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32_slice(&lhs)),
            Value::from(pack_f32_slice(&rhs)),
        ],
    )
    .expect("binary_broadcast_rhs scalar subtraction must succeed");

    let actual = decode_f32_le_bytes_all(&outputs[out_idx].to_bytes());
    assert_eq!(
        actual, expected,
        "binary_broadcast_rhs must subtract scalar RHS from all LHS lanes"
    );
}

// ---------------------------------------------------------------------------
// 2. f32_binary
// ---------------------------------------------------------------------------

#[test]
fn f32_binary_multiplication_over_signed_and_fractional_floats() {
    let count = 5u32;
    let program = ElementwiseComposer::f32_binary(
        "vyre-libs::test::f32_binary_mul",
        "lhs",
        "rhs",
        "out",
        count,
        Expr::mul,
    );
    assert_program_valid(&program);

    let lhs: Vec<f32> = vec![-2.5, 0.0, 4.0, 1.5, 100.0];
    let rhs: Vec<f32> = vec![2.0, 7.0, -0.5, 3.0, 0.01];
    let expected: Vec<f32> = lhs.iter().zip(&rhs).map(|(&l, &r)| l * r).collect();

    let out_idx = vyre_reference::output_index(&program, "out")
        .expect("output buffer `out` must be declared");

    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32_slice(&lhs)),
            Value::from(pack_f32_slice(&rhs)),
        ],
    )
    .expect("f32_binary multiplication reference evaluation must succeed");

    let actual = decode_f32_le_bytes_all(&outputs[out_idx].to_bytes());
    assert_eq!(
        actual, expected,
        "f32_binary must compute pairwise f32 multiplication"
    );
}

// ---------------------------------------------------------------------------
// 3. level_wave_program_with_buffers_and_op_id
// ---------------------------------------------------------------------------

#[cfg(feature = "graph")]
mod level_wave_tests {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, Node};
    use vyre_libs::graph::level_wave::level_wave_program_with_buffers_and_op_id;

    #[test]
    fn level_wave_program_with_buffers_and_op_id_enforces_depth_wave_dependency_order() {
        // Construct a 3-level DAG over 4 nodes:
        // Depth 0: node 0 (init=3), node 1 (init=7)
        // Depth 1: node 2 computes val[2] = val[0] + val[1] (3 + 7 = 10)
        // Depth 2: node 3 computes val[3] = val[2] * 2 (10 * 2 = 20)
        let lane = Expr::InvocationId { axis: 0 };
        let step_body = vec![
            Node::if_then(
                Expr::eq(lane.clone(), Expr::u32(2)),
                vec![Node::store(
                    "val",
                    Expr::u32(2),
                    Expr::add(
                        Expr::load("val", Expr::u32(0)),
                        Expr::load("val", Expr::u32(1)),
                    ),
                )],
            ),
            Node::if_then(
                Expr::eq(lane, Expr::u32(3)),
                vec![Node::store(
                    "val",
                    Expr::u32(3),
                    Expr::mul(Expr::load("val", Expr::u32(2)), Expr::u32(2)),
                )],
            ),
        ];

        let extra_buffers =
            vec![
                BufferDecl::storage("val", 1, BufferAccess::ReadWrite, DataType::U32).with_count(4),
            ];

        let custom_op_id = "vyre-libs::test::custom_dag_level_wave";
        let program = level_wave_program_with_buffers_and_op_id(
            custom_op_id,
            step_body,
            "depth_buf",
            extra_buffers,
            3, // max_depth: 0, 1, 2
            4, // lane_count
        );
        assert_program_valid(&program);

        let depths: Vec<u32> = vec![0, 0, 1, 2];
        let init_val: Vec<u32> = vec![3, 7, 0, 0];
        let expected_final_val: Vec<u32> = vec![3, 7, 10, 20];

        let val_out_idx = vyre_reference::output_index(&program, "val")
            .expect("output buffer `val` must be declared");

        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(pack_u32_slice(&depths)),
                Value::from(pack_u32_slice(&init_val)),
            ],
        )
        .expect("level_wave_program_with_buffers_and_op_id must execute successfully");

        let actual = decode_u32_le_bytes_all(&outputs[val_out_idx].to_bytes());
        assert_eq!(
            actual, expected_final_val,
            "level_wave_program_with_buffers_and_op_id must sequence execution across depth waves"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. line_index_u8_with_geometry
// ---------------------------------------------------------------------------

#[cfg(feature = "text")]
mod line_index_tests {
    use super::*;
    use vyre_foundation::LaunchGeometry;
    use vyre_libs::text::line_index_u8_with_geometry;
    use vyre_reference::composition_witness::line_index_witness;

    #[test]
    fn line_index_u8_with_geometry_matches_composition_witness() {
        let corpus: &[&[u8]] = &[
            b"alpha\nbeta\r\ngamma\ndelta",
            b"single line without trailing newline",
            b"\n\n\n",
            b"x",
            b"",
        ];

        let geometry = LaunchGeometry {
            workgroup: [256, 1, 1],
            ..Default::default()
        };

        for &source in corpus {
            let n = source.len() as u32;
            let program = line_index_u8_with_geometry("source", "lines", n, &geometry);
            assert_program_valid(&program);

            let expected = line_index_witness(source);
            let lines_idx = vyre_reference::output_index(&program, "lines")
                .expect("lines output buffer must be declared");

            let outputs = vyre_reference::reference_eval(&program, &[Value::from(source.to_vec())])
                .expect("line_index_u8_with_geometry reference_eval must succeed");

            let mut actual = decode_u32_le_bytes_all(&outputs[lines_idx].to_bytes());
            actual.truncate(source.len());

            assert_eq!(
                actual,
                expected,
                "line_index_u8_with_geometry must match canonical line_index_witness for {:?}",
                std::str::from_utf8(source).unwrap_or("<binary>")
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5. online_softmax_attention
// ---------------------------------------------------------------------------

#[cfg(feature = "nn-attention")]
mod attention_tests {
    use super::*;
    use vyre_libs::nn::attention::plan_flash_attention_tiled;
    use vyre_libs::nn::attention::tiled_online_softmax::online_softmax_attention;

    #[test]
    fn online_softmax_attention_canonical_witness_parity() {
        // Plan: s=9, h=1, d=4
        let plan = plan_flash_attention_tiled(9, 1, 4)
            .expect("plan_flash_attention_tiled must succeed for canonical dimensions");

        let program = online_softmax_attention("q", "k", "v", "out", &plan);
        assert_program_valid(&program);

        let q = vec![0.0f32; 9];
        let k = vec![0.0f32; 9];
        let v: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // Zero Q and K make all keys equally likely; each row receives the mean of V = 36/9 = 4.0
        let expected = vec![4.0f32; 9];

        let out_idx = vyre_reference::output_index(&program, "out")
            .expect("output buffer `out` must be declared");

        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(pack_f32_slice(&q)),
                Value::from(pack_f32_slice(&k)),
                Value::from(pack_f32_slice(&v)),
            ],
        )
        .expect("online_softmax_attention reference evaluation must succeed");

        let actual = decode_f32_le_bytes_all(&outputs[out_idx].to_bytes());
        assert_eq!(
            actual, expected,
            "online_softmax_attention must compute the exact expected mean of V under uniform attention"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. ternary
// ---------------------------------------------------------------------------

#[test]
fn ternary_u32_conditional_select() {
    let count = 6u32;
    let program = ElementwiseComposer::ternary(
        "vyre-libs::test::ternary_select",
        "cond",
        "x",
        "y",
        DataType::U32,
        "out",
        DataType::U32,
        count,
        |cond, x, y| Expr::select(Expr::ne(cond, Expr::u32(0)), x, y),
    );
    assert_program_valid(&program);

    let cond: Vec<u32> = vec![1, 0, 1, 0, 0, 1];
    let x: Vec<u32> = vec![10, 20, 30, 40, 50, 60];
    let y: Vec<u32> = vec![100, 200, 300, 400, 500, 600];
    let expected: Vec<u32> = cond
        .iter()
        .zip(&x)
        .zip(&y)
        .map(|((&c, &xv), &yv)| if c != 0 { xv } else { yv })
        .collect();

    let out_idx = vyre_reference::output_index(&program, "out")
        .expect("output buffer `out` must be declared");

    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32_slice(&cond)),
            Value::from(pack_u32_slice(&x)),
            Value::from(pack_u32_slice(&y)),
        ],
    )
    .expect("ternary select reference evaluation must succeed");

    let actual = decode_u32_le_bytes_all(&outputs[out_idx].to_bytes());
    assert_eq!(
        actual, expected,
        "ternary must evaluate conditional select lane-by-lane"
    );
}

#[test]
fn ternary_f32_fused_multiply_add() {
    let count = 4u32;
    let program = ElementwiseComposer::ternary(
        "vyre-libs::test::ternary_fma",
        "a",
        "b",
        "c",
        DataType::F32,
        "out",
        DataType::F32,
        count,
        |a, b, c| Expr::add(Expr::mul(a, b), c),
    );
    assert_program_valid(&program);

    let a: Vec<f32> = vec![2.0, 3.0, -1.0, 0.5];
    let b: Vec<f32> = vec![4.0, -2.0, 5.0, 10.0];
    let c: Vec<f32> = vec![1.5, 6.0, 10.0, -2.0];
    let expected: Vec<f32> = a
        .iter()
        .zip(&b)
        .zip(&c)
        .map(|((&av, &bv), &cv)| (av * bv) + cv)
        .collect();

    let out_idx = vyre_reference::output_index(&program, "out")
        .expect("output buffer `out` must be declared");

    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32_slice(&a)),
            Value::from(pack_f32_slice(&b)),
            Value::from(pack_f32_slice(&c)),
        ],
    )
    .expect("ternary fma reference evaluation must succeed");

    let actual = decode_f32_le_bytes_all(&outputs[out_idx].to_bytes());
    assert_eq!(
        actual, expected,
        "ternary must compute pairwise fused multiply-add on f32 values"
    );
}

// ---------------------------------------------------------------------------
// 7. u32_unary
// ---------------------------------------------------------------------------

#[test]
fn u32_unary_bitwise_not() {
    let count = 5u32;
    let program = ElementwiseComposer::u32_unary(
        "vyre-libs::test::u32_unary_not",
        "input",
        "output",
        count,
        Expr::bitnot,
    );
    assert_program_valid(&program);

    let input: Vec<u32> = vec![0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555, 0x12345678];
    let expected: Vec<u32> = input.iter().map(|&x| !x).collect();

    let out_idx = vyre_reference::output_index(&program, "output")
        .expect("output buffer `output` must be declared");

    let outputs = vyre_reference::reference_eval(&program, &[Value::from(pack_u32_slice(&input))])
        .expect("u32_unary bit_not reference evaluation must succeed");

    let actual = decode_u32_le_bytes_all(&outputs[out_idx].to_bytes());
    assert_eq!(
        actual, expected,
        "u32_unary must compute bitwise NOT on all input lanes"
    );
}

#[test]
fn u32_unary_linear_transform() {
    let count = 5u32;
    let program = ElementwiseComposer::u32_unary(
        "vyre-libs::test::u32_unary_linear",
        "input",
        "output",
        count,
        |x| Expr::add(Expr::mul(x, Expr::u32(5)), Expr::u32(3)),
    );
    assert_program_valid(&program);

    let input: Vec<u32> = vec![0, 1, 10, 100, 1000];
    let expected: Vec<u32> = input.iter().map(|&x| x * 5 + 3).collect();

    let out_idx = vyre_reference::output_index(&program, "output")
        .expect("output buffer `output` must be declared");

    let outputs = vyre_reference::reference_eval(&program, &[Value::from(pack_u32_slice(&input))])
        .expect("u32_unary linear transform reference evaluation must succeed");

    let actual = decode_u32_le_bytes_all(&outputs[out_idx].to_bytes());
    assert_eq!(
        actual, expected,
        "u32_unary must compute linear transformation on all input lanes"
    );
}
#[test]
fn read_write_output_storage_is_backend_allocated() {
    let program = ElementwiseComposer::new("vyre-libs::test::read_write_output", 3)
        .add_input("input", DataType::U32, 3)
        .add_output_storage("output", BufferAccess::ReadWrite, DataType::U32, 3)
        .build_pointwise("output", |i| {
            Expr::add(Expr::load("input", i), Expr::u32(1))
        });
    assert_program_valid(&program);
    assert!(
        program
            .buffer("output")
            .expect("output buffer must be declared")
            .is_backend_allocated_output(),
        "read-write output storage must not become a required host input"
    );

    let outputs =
        vyre_reference::reference_eval(&program, &[Value::from(pack_u32_slice(&[1, 2, 3]))])
            .expect("read-write output storage must execute without a host-provided output seed");
    assert_eq!(
        decode_u32_le_bytes_all(&outputs[0].to_bytes()),
        vec![2, 3, 4]
    );
}
