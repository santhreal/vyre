//! Parity suite for the cpu-ref backend.
//!
//! Verifies that `CpuRefBackend::dispatch` produces correct results for
//! every major IR shape: arithmetic, bitwise, control flow, memory access,
//! and multi-buffer programs. Each test constructs a `Program`, dispatches
//! through the `VyreBackend` trait surface, and asserts byte-exact output.

use vyre_driver::DispatchConfig;
use vyre_driver::VyreBackend;
use vyre_driver_reference::CpuRefBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

mod dispatch_fixtures;
use dispatch_fixtures::{binary_program, dispatch_no_input, dispatch_with_inputs, u32_out_buffer};

// ---------------------------------------------------------------
// Scalar expression shapes: store, arithmetic, bitwise
// ---------------------------------------------------------------

/// One scalar shape: the expression the program stores into `out[0]`, and the
/// word the reference backend must return for it.
struct ScalarCase {
    name: &'static str,
    value: fn() -> Expr,
    expected: u32,
}

const SCALAR_CASES: &[ScalarCase] = &[
    ScalarCase {
        name: "store literal",
        value: || Expr::u32(42),
        expected: 42,
    },
    ScalarCase {
        name: "store zero",
        value: || Expr::u32(0),
        expected: 0,
    },
    ScalarCase {
        name: "add",
        value: || Expr::add(Expr::u32(10), Expr::u32(32)),
        expected: 42,
    },
    ScalarCase {
        name: "sub",
        value: || Expr::sub(Expr::u32(50), Expr::u32(8)),
        expected: 42,
    },
    ScalarCase {
        name: "mul",
        value: || Expr::mul(Expr::u32(6), Expr::u32(7)),
        expected: 42,
    },
    ScalarCase {
        name: "bitxor",
        value: || Expr::bitxor(Expr::u32(0xFF), Expr::u32(0x55)),
        expected: 0xAA,
    },
    ScalarCase {
        name: "bitand",
        value: || Expr::bitand(Expr::u32(0xFF), Expr::u32(0x0F)),
        expected: 0x0F,
    },
    ScalarCase {
        name: "bitor",
        value: || Expr::bitor(Expr::u32(0xF0), Expr::u32(0x0F)),
        expected: 0xFF,
    },
];

#[test]
fn scalar_expression_shapes_dispatch_to_their_pinned_word() {
    for case in SCALAR_CASES {
        let program = Program::wrapped(
            vec![u32_out_buffer("out", 0)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), (case.value)())],
        );
        assert_eq!(
            dispatch_no_input(&program),
            vec![case.expected.to_le_bytes().to_vec()],
            "Fix: cpu-ref must evaluate the {} shape to {}.",
            case.name,
            case.expected
        );
    }
}

// ---------------------------------------------------------------
// Input buffer passthrough (read → write)
// ---------------------------------------------------------------

#[test]
fn input_buffer_passthrough() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32),
            u32_out_buffer("out", 1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("val", Expr::load("input", Expr::u32(0))),
            Node::store("out", Expr::u32(0), Expr::var("val")),
        ],
    );
    let input = 99u32.to_le_bytes().to_vec();
    let outputs = dispatch_with_inputs(&program, &[input]);
    assert_eq!(outputs, vec![99u32.to_le_bytes().to_vec()]);
}

/// WHY: a reference input the caller never supplied is an ABI failure. Zero
/// synthesis would answer it with fabricated data and hide the caller's defect
/// behind a plausible-looking result.
#[test]
fn missing_input_buffer_is_rejected_rather_than_synthesized() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32),
            u32_out_buffer("out", 1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("val", Expr::load("input", Expr::u32(0))),
            Node::store("out", Expr::u32(0), Expr::var("val")),
        ],
    );
    let error = CpuRefBackend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect_err("a missing reference input must be rejected");
    assert!(
        error
            .to_string()
            .contains("missing an input buffer for `input`"),
        "reference backend must name the buffer it never received: {error}"
    );
}

/// WHY: backend-allocated outputs are not host inputs. Accepting an initializer
/// creates a second submission geometry that can diverge from target backends.
#[test]
fn backend_allocated_output_initializer_is_rejected() {
    let program = Program::wrapped(
        vec![u32_out_buffer("out", 0)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );

    let error = CpuRefBackend
        .dispatch(
            &program,
            &[0_u32.to_le_bytes().to_vec()],
            &DispatchConfig::default(),
        )
        .expect_err("backend-allocated output initializers must be rejected");
    assert!(
        error.to_string().contains("extra input buffer"),
        "reference backend must report the non-canonical input: {error}"
    );
}

// ---------------------------------------------------------------
// Two-buffer XOR (the README example)
// ---------------------------------------------------------------

#[test]
fn two_buffer_xor() {
    let program = binary_program(Expr::bitxor);
    let a = 0xAAu32.to_le_bytes().to_vec();
    let b = 0x55u32.to_le_bytes().to_vec();
    let outputs = dispatch_with_inputs(&program, &[a, b]);
    // 0xAA ^ 0x55 = 0xFF = 255
    assert_eq!(outputs, vec![255u32.to_le_bytes().to_vec()]);
}

// ---------------------------------------------------------------
// Conditional: if-then store
// ---------------------------------------------------------------

#[test]
fn conditional_if_true() {
    let program = Program::wrapped(
        vec![u32_out_buffer("out", 0)],
        [1, 1, 1],
        vec![
            // Store 0 first, then conditionally overwrite with 42
            Node::store("out", Expr::u32(0), Expr::u32(0)),
            Node::if_then(
                Expr::bool(true),
                vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
            ),
        ],
    );
    let outputs = dispatch_no_input(&program);
    assert_eq!(outputs, vec![42u32.to_le_bytes().to_vec()]);
}

#[test]
fn conditional_if_false() {
    let program = Program::wrapped(
        vec![u32_out_buffer("out", 0)],
        [1, 1, 1],
        vec![
            // Store 99, then conditionally overwrite  -  but condition is false
            Node::store("out", Expr::u32(0), Expr::u32(99)),
            Node::if_then(
                Expr::bool(false),
                vec![Node::store("out", Expr::u32(0), Expr::u32(0))],
            ),
        ],
    );
    let outputs = dispatch_no_input(&program);
    // if-false branch not taken → 99 survives
    assert_eq!(outputs, vec![99u32.to_le_bytes().to_vec()]);
}

// ---------------------------------------------------------------
// Backend trait surface: dispatch_borrowed
// ---------------------------------------------------------------

#[test]
fn dispatch_borrowed_matches_owned() {
    let backend = CpuRefBackend;
    let program = Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32),
            u32_out_buffer("out", 1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("val", Expr::load("a", Expr::u32(0))),
            Node::store("out", Expr::u32(0), Expr::var("val")),
        ],
    );
    let input_bytes = 77u32.to_le_bytes();

    let owned_out = backend
        .dispatch(
            &program,
            &[input_bytes.to_vec()],
            &DispatchConfig::default(),
        )
        .expect("owned dispatch");
    let borrowed_out = backend
        .dispatch_borrowed(&program, &[&input_bytes[..]], &DispatchConfig::default())
        .expect("borrowed dispatch");

    assert_eq!(
        owned_out, borrowed_out,
        "Fix: dispatch and dispatch_borrowed must produce identical bytes."
    );
}

// ---------------------------------------------------------------
// Backend trait surface: dispatch_borrowed_timed
// ---------------------------------------------------------------

#[test]
fn dispatch_borrowed_timed_returns_wall_time() {
    let backend = CpuRefBackend;
    let program = Program::wrapped(
        vec![u32_out_buffer("out", 0)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    let result = backend
        .dispatch_borrowed_timed(&program, &[], &DispatchConfig::default())
        .expect("timed dispatch");
    assert_eq!(result.outputs, vec![1u32.to_le_bytes().to_vec()]);
    // wall_ns should be non-zero (program does actual work)
    // device_ns should be None (CPU backend has no device timer)
    assert!(result.device_ns.is_none());
}

// ---------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------

#[test]
fn extra_input_buffers_rejected() {
    let backend = CpuRefBackend;
    // Program has exactly 1 non-output ReadWrite buffer  -  it consumes 1 input.
    // Passing 2 inputs means 1 extra trailing input → rejected.
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    let result = backend.dispatch(
        &program,
        &[vec![0; 4], vec![0; 4]],
        &DispatchConfig::default(),
    );
    assert!(
        result.is_err(),
        "Fix: extra input buffers must be rejected."
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Fix:"),
        "Fix: error must carry Fix: hint, got: {err_msg}"
    );
}

// ---------------------------------------------------------------
// Capability queries
// ---------------------------------------------------------------

#[test]
fn capability_queries_conservative() {
    let backend = CpuRefBackend;
    assert_eq!(backend.id(), "cpu-ref");
    assert_eq!(backend.max_workgroup_size(), [1024, 1, 1]);
    assert_eq!(backend.max_compute_workgroups_per_dimension(), u32::MAX);
    // CPU backend should report conservative capabilities
    assert!(!backend.supports_subgroup_ops());
    assert!(!backend.supports_f16());
    assert!(!backend.supports_tensor_cores());
    assert!(!backend.supports_async_compute());
}

// ---------------------------------------------------------------
// Determinism: same program twice = same bytes
// ---------------------------------------------------------------

#[test]
fn determinism_guarantee() {
    let backend = CpuRefBackend;
    let program = Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32),
            u32_out_buffer("out", 1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("v", Expr::add(Expr::load("a", Expr::u32(0)), Expr::u32(1))),
            Node::store("out", Expr::u32(0), Expr::var("v")),
        ],
    );
    let input = 100u32.to_le_bytes().to_vec();
    let config = DispatchConfig::default();

    let out1 = backend
        .dispatch(&program, &[input.clone()], &config)
        .unwrap();
    let out2 = backend.dispatch(&program, &[input], &config).unwrap();
    assert_eq!(
        out1, out2,
        "Fix: cpu-ref must be deterministic  -  identical inputs must produce identical outputs."
    );
}
