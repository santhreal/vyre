//! Contracts for the wgpu host input ABI.
//!
//! One answer decides which declarations a caller fills:
//! `BufferDecl::consumes_host_input`. The wgpu backend accepts exactly one input
//! list shape, the canonical one, and refuses any other count by naming both the
//! expected count and the received count.

#![cfg(feature = "device-tests")]

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_primitives::wire::pack_u32_slice;
use vyre_reference::value::Value;

fn sample_mixed_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("in_read", 0, DataType::U32).with_count(4),
            BufferDecl::output("out_buf", 1, DataType::U32).with_count(4),
            BufferDecl::read_write("state_rw", 2, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out_buf",
            Expr::gid_x(),
            Expr::add(
                Expr::load("in_read", Expr::gid_x()),
                Expr::load("state_rw", Expr::gid_x()),
            ),
        )],
    )
}

/// A long-form input list carrying a placeholder for a backend-allocated output
/// is refused with both expected and received counts named, while the canonical
/// list dispatches successfully.
#[test]
fn long_form_with_output_placeholder_is_refused_with_both_counts_named() {
    let backend = WgpuBackend::acquire()
        .expect("Fix: live WGPU backend is required for input ABI contract coverage");
    let program = sample_mixed_program();

    let expected_count = program
        .buffers()
        .iter()
        .filter(|decl| decl.consumes_host_input())
        .count();
    assert_eq!(expected_count, 2, "in_read and state_rw consume host input");

    let canonical_inputs = vec![
        pack_u32_slice(&[1, 2, 3, 4]),
        pack_u32_slice(&[10, 20, 30, 40]),
    ];

    let result = backend.dispatch(&program, &canonical_inputs, &DispatchConfig::default());
    assert!(
        result.is_ok(),
        "canonical input list with exact host-consuming slots must dispatch: {:?}",
        result.err()
    );

    // Long form: one slot per non-shared non-trap binding (3 slots), including
    // a placeholder for the backend-allocated output at slot index 1.
    let mut long_form_inputs = canonical_inputs.clone();
    long_form_inputs.insert(1, vec![0u8; 16]);
    assert_eq!(long_form_inputs.len(), 3);

    let too_long_err = backend
        .dispatch(&program, &long_form_inputs, &DispatchConfig::default())
        .expect_err("long-form input list with placeholder must be refused");
    let msg = too_long_err.to_string();
    assert!(
        msg.contains("expected 2 input buffer(s) from Program declarations but received 3"),
        "rejection must state both expected 2 and received 3, got: {msg}"
    );

    // Short form: one slot instead of two.
    let short_form_inputs = vec![canonical_inputs[0].clone()];
    let too_short_err = backend
        .dispatch(&program, &short_form_inputs, &DispatchConfig::default())
        .expect_err("short input list must be refused");
    let short_msg = too_short_err.to_string();
    assert!(
        short_msg.contains("expected 2 input buffer(s) from Program declarations but received 1"),
        "rejection must state both expected 2 and received 1, got: {short_msg}"
    );
}

/// The reference interpreter and the wgpu backend agree on which input counts
/// they accept for the same program, derived from `BufferDecl::consumes_host_input`.
#[test]
fn reference_interpreter_and_wgpu_backend_agree_on_accepted_input_counts() {
    let backend = WgpuBackend::acquire()
        .expect("Fix: live WGPU backend is required for input ABI contract coverage");

    let test_programs = vec![
        // 1 output only -> 0 host inputs
        Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
            [1, 1, 1],
            vec![Node::store("out", Expr::gid_x(), Expr::u32(42))],
        ),
        // 1 read, 1 output -> 1 host input
        Program::wrapped(
            vec![
                BufferDecl::read("in", 0, DataType::U32).with_count(4),
                BufferDecl::output("out", 1, DataType::U32).with_count(4),
            ],
            [1, 1, 1],
            vec![Node::store("out", Expr::gid_x(), Expr::load("in", Expr::gid_x()))],
        ),
        // 1 read, 1 output, 1 read-write, 1 workgroup -> 2 host inputs
        Program::wrapped(
            vec![
                BufferDecl::read("in", 0, DataType::U32).with_count(4),
                BufferDecl::output("out", 1, DataType::U32).with_count(4),
                BufferDecl::read_write("acc", 2, DataType::U32).with_count(4),
                BufferDecl::workgroup("shared", 3, DataType::U32),
            ],
            [1, 1, 1],
            vec![
                Node::store("acc", Expr::gid_x(), Expr::load("in", Expr::gid_x())),
                Node::store("out", Expr::gid_x(), Expr::load("acc", Expr::gid_x())),
            ],
        ),
        // 2 read, 1 uniform, 1 output -> 3 host inputs
        Program::wrapped(
            vec![
                BufferDecl::read("in1", 0, DataType::U32).with_count(4),
                BufferDecl::read("in2", 1, DataType::U32).with_count(4),
                BufferDecl::storage("params", 2, BufferAccess::Uniform, DataType::U32).with_count(1),
                BufferDecl::output("out", 3, DataType::U32).with_count(4),
            ],
            [1, 1, 1],
            vec![Node::store(
                "out",
                Expr::gid_x(),
                Expr::add(
                    Expr::load("in1", Expr::gid_x()),
                    Expr::load("in2", Expr::gid_x()),
                ),
            )],
        ),
    ];

    for program in &test_programs {
        let expected_count = program
            .buffers()
            .iter()
            .filter(|decl| decl.consumes_host_input())
            .count();

        // Exact inputs
        let wgpu_inputs = vec![pack_u32_slice(&[1, 2, 3, 4]); expected_count];
        let ref_inputs: Vec<Value> = wgpu_inputs
            .iter()
            .cloned()
            .map(Value::from)
            .collect();

        assert!(
            vyre_reference::reference_eval(program, &ref_inputs).is_ok(),
            "reference_eval must accept exactly {expected_count} inputs"
        );
        assert!(
            backend
                .dispatch(program, &wgpu_inputs, &DispatchConfig::default())
                .is_ok(),
            "wgpu backend must accept exactly {expected_count} inputs"
        );

        // One extra input
        let mut wgpu_extra = wgpu_inputs.clone();
        wgpu_extra.push(pack_u32_slice(&[0, 0, 0, 0]));
        let ref_extra: Vec<Value> = wgpu_extra
            .iter()
            .cloned()
            .map(Value::from)
            .collect();

        let ref_extra_err = vyre_reference::reference_eval(program, &ref_extra)
            .expect_err("reference_eval must reject extra input");
        assert!(
            ref_extra_err.to_string().contains("unused input"),
            "reference error must report unused input, got: {ref_extra_err}"
        );

        let wgpu_extra_err = backend
            .dispatch(program, &wgpu_extra, &DispatchConfig::default())
            .expect_err("wgpu backend must reject extra input");
        let wgpu_msg = wgpu_extra_err.to_string();
        assert!(
            wgpu_msg.contains(&format!(
                "expected {expected_count} input buffer(s) from Program declarations but received {}",
                expected_count + 1
            )),
            "wgpu error must state expected {expected_count} and received {}, got: {wgpu_msg}",
            expected_count + 1
        );

        // One fewer input (if expected_count > 0)
        if expected_count > 0 {
            let wgpu_under = wgpu_inputs[..expected_count - 1].to_vec();
            let ref_under: Vec<Value> = wgpu_under
                .iter()
                .cloned()
                .map(Value::from)
                .collect();

            let ref_under_err = vyre_reference::reference_eval(program, &ref_under)
                .expect_err("reference_eval must reject missing input");
            assert!(
                ref_under_err.to_string().contains("missing input"),
                "reference error must report missing input, got: {ref_under_err}"
            );

            let wgpu_under_err = backend
                .dispatch(program, &wgpu_under, &DispatchConfig::default())
                .expect_err("wgpu backend must reject missing input");
            let wgpu_under_msg = wgpu_under_err.to_string();
            assert!(
                wgpu_under_msg.contains(&format!(
                    "expected {expected_count} input buffer(s) from Program declarations but received {}",
                    expected_count - 1
                )),
                "wgpu error must state expected {expected_count} and received {}, got: {wgpu_under_msg}",
                expected_count - 1
            );
        }
    }
}
