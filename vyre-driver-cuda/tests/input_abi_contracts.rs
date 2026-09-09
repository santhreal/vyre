//! Contracts for the CUDA host input ABI (Row 125).
//!
//! One answer decides which declarations a caller fills:
//! `BufferDecl::consumes_host_input`. The CUDA backend accepts exactly one input
//! list shape, the canonical one, and refuses any other count by naming both the
//! expected count and the received count.

use vyre_driver::BindingPlan;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, MemoryKind, Node, Program};
use vyre_reference::value::Value;

fn sample_mixed_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("in_read", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(4)
                .with_kind(MemoryKind::Global),
            BufferDecl::storage("out_write", 1, BufferAccess::WriteOnly, DataType::U32)
                .with_count(4)
                .with_kind(MemoryKind::Global),
            BufferDecl::storage("state_rw", 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(4)
                .with_kind(MemoryKind::Global),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out_write",
            Expr::u32(0),
            Expr::add(
                Expr::load("in_read", Expr::u32(0)),
                Expr::load("state_rw", Expr::u32(0)),
            ),
        )],
    )
}

fn sample_test_programs() -> Vec<Program> {
    vec![
        // 0 inputs, 1 output
        Program::wrapped(
            vec![
                BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                    .with_count(4)
                    .with_kind(MemoryKind::Global),
            ],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
        ),
        // 1 input, 1 output
        Program::wrapped(
            vec![
                BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4)
                    .with_kind(MemoryKind::Global),
                BufferDecl::storage("out", 1, BufferAccess::WriteOnly, DataType::U32)
                    .with_count(4)
                    .with_kind(MemoryKind::Global),
            ],
            [1, 1, 1],
            vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::load("in", Expr::u32(0)),
            )],
        ),
        // 2 inputs, 1 output
        Program::wrapped(
            vec![
                BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4)
                    .with_kind(MemoryKind::Global),
                BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4)
                    .with_kind(MemoryKind::Global),
                BufferDecl::storage("out", 2, BufferAccess::WriteOnly, DataType::U32)
                    .with_count(4)
                    .with_kind(MemoryKind::Global),
            ],
            [1, 1, 1],
            vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::add(Expr::load("a", Expr::u32(0)), Expr::load("b", Expr::u32(0))),
            )],
        ),
        // 1 read-only, 1 read-write, 1 write-only (2 host inputs)
        sample_mixed_program(),
        // Shared-tier storage buffer -> 1 host input
        Program::wrapped(
            vec![
                BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4)
                    .with_kind(MemoryKind::Global),
                BufferDecl::storage("shared_tier", 1, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4)
                    .with_kind(MemoryKind::Shared),
                BufferDecl::storage("out", 2, BufferAccess::WriteOnly, DataType::U32)
                    .with_count(4)
                    .with_kind(MemoryKind::Global),
            ],
            [1, 1, 1],
            vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::load("in", Expr::u32(0)),
            )],
        ),
    ]
}

fn canonical_inputs_for_program(program: &Program) -> Vec<Vec<u8>> {
    program
        .buffers()
        .iter()
        .filter(|decl| decl.consumes_host_input())
        .map(|decl| {
            let bytes = decl.static_byte_len().ok().flatten().unwrap_or(16);
            vec![0u8; bytes]
        })
        .collect()
}

/// A contract test proving that host input list length validation derives the
/// expected slot count from the program's declarations at runtime:
/// an input list one slot too long is refused by name, one slot too short is
/// refused by name, and the exact list is accepted.
#[test]
fn host_input_abi_refusal_contracts_non_device() {
    for program in &sample_test_programs() {
        let expected_count = program
            .buffers()
            .iter()
            .filter(|decl| decl.consumes_host_input())
            .count();

        let canonical_inputs = canonical_inputs_for_program(program);

        // Exact input list builds clean
        let plan = BindingPlan::from_program(program, &canonical_inputs);
        assert!(
            plan.is_ok(),
            "exact input count {expected_count} must build clean binding plan: {:?}",
            plan.err()
        );

        // Input list one slot too long is refused naming both counts
        let mut too_long = canonical_inputs.clone();
        too_long.push(vec![0u8; 16]);
        let too_long_err = BindingPlan::from_program(program, &too_long)
            .expect_err("input list one slot too long must be refused");
        let msg = too_long_err.to_string();
        assert!(
            msg.contains(&format!(
                "expected {expected_count} input buffer(s) from Program declarations but received {}",
                expected_count + 1
            )),
            "rejection must state expected {expected_count} and received {}, got: {msg}",
            expected_count + 1
        );

        // Input list one slot too short is refused naming both counts
        if expected_count > 0 {
            let too_short = canonical_inputs[..expected_count - 1].to_vec();
            let too_short_err = BindingPlan::from_program(program, &too_short)
                .expect_err("input list one slot too short must be refused");
            let short_msg = too_short_err.to_string();
            assert!(
                short_msg.contains(&format!(
                    "expected {expected_count} input buffer(s) from Program declarations but received {}",
                    expected_count - 1
                )),
                "rejection must state expected {expected_count} and received {}, got: {short_msg}",
                expected_count - 1
            );
        }
    }
}

/// A long-form input list carrying a placeholder for a backend-allocated output
/// is refused with both expected and received counts named, while the canonical
/// list dispatches successfully on the device.
#[cfg(feature = "device-tests")]
#[test]
fn long_form_with_output_placeholder_is_refused_with_both_counts_named() {
    use vyre_driver::{DispatchConfig, VyreBackend};
    use vyre_driver_cuda::CudaBackend;

    let backend = CudaBackend::acquire()
        .expect("Fix: live CUDA backend is required for input ABI contract coverage");
    let program = sample_mixed_program();

    let expected_count = program
        .buffers()
        .iter()
        .filter(|decl| decl.consumes_host_input())
        .count();
    assert_eq!(expected_count, 2, "in_read and state_rw consume host input");

    let canonical_inputs = canonical_inputs_for_program(&program);

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

/// The reference interpreter and the CUDA backend agree on which input counts
/// they accept for the same program, derived from `BufferDecl::consumes_host_input`.
#[cfg(feature = "device-tests")]
#[test]
fn reference_interpreter_and_cuda_backend_agree_on_accepted_input_counts() {
    use vyre_driver::{DispatchConfig, VyreBackend};
    use vyre_driver_cuda::CudaBackend;

    let backend = CudaBackend::acquire()
        .expect("Fix: live CUDA backend is required for input ABI contract coverage");

    for program in &sample_test_programs() {
        let expected_count = program
            .buffers()
            .iter()
            .filter(|decl| decl.consumes_host_input())
            .count();

        // Exact inputs
        let cuda_inputs = canonical_inputs_for_program(program);
        let ref_inputs: Vec<Value> = cuda_inputs.iter().cloned().map(Value::from).collect();

        assert!(
            vyre_reference::reference_eval(program, &ref_inputs).is_ok(),
            "reference_eval must accept exactly {expected_count} inputs"
        );
        assert!(
            backend
                .dispatch(program, &cuda_inputs, &DispatchConfig::default())
                .is_ok(),
            "cuda backend must accept exactly {expected_count} inputs"
        );

        // One extra input
        let mut cuda_extra = cuda_inputs.clone();
        cuda_extra.push(vec![0u8; 4]);
        let ref_extra: Vec<Value> = cuda_extra.iter().cloned().map(Value::from).collect();

        let ref_extra_err = vyre_reference::reference_eval(program, &ref_extra)
            .expect_err("reference_eval must reject extra input");
        assert!(
            ref_extra_err.to_string().contains("unused input"),
            "reference error must report unused input, got: {ref_extra_err}"
        );

        let cuda_extra_err = backend
            .dispatch(program, &cuda_extra, &DispatchConfig::default())
            .expect_err("cuda backend must reject extra input");
        let cuda_msg = cuda_extra_err.to_string();
        assert!(
            cuda_msg.contains(&format!(
                "expected {expected_count} input buffer(s) from Program declarations but received {}",
                expected_count + 1
            )),
            "cuda error must state expected {expected_count} and received {}, got: {cuda_msg}",
            expected_count + 1
        );

        // One fewer input (if expected_count > 0)
        if expected_count > 0 {
            let cuda_under = cuda_inputs[..expected_count - 1].to_vec();
            let ref_under: Vec<Value> = cuda_under.iter().cloned().map(Value::from).collect();

            let ref_under_err = vyre_reference::reference_eval(program, &ref_under)
                .expect_err("reference_eval must reject missing input");
            assert!(
                ref_under_err.to_string().contains("missing input"),
                "reference error must report missing input, got: {ref_under_err}"
            );

            let cuda_under_err = backend
                .dispatch(program, &cuda_under, &DispatchConfig::default())
                .expect_err("cuda backend must reject missing input");
            let cuda_under_msg = cuda_under_err.to_string();
            assert!(
                cuda_under_msg.contains(&format!(
                    "expected {expected_count} input buffer(s) from Program declarations but received {}",
                    expected_count - 1
                )),
                "cuda error must state expected {expected_count} and received {}, got: {cuda_under_msg}",
                expected_count - 1
            );
        }
    }
}
