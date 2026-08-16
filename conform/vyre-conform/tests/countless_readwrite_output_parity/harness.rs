//! The XOR program every case below dispatches, and the three paths it runs on.

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_conform::production::ProductionSession;
use vyre_reference::value::Value;

/// Element count for the main cases. Matches the `[64, 1, 1]` workgroup exactly.
pub(crate) const N: u32 = 64;

/// Build the XOR program whose single writable buffer is declared by `out`.
///
/// The store is gated on `idx < len` so the boundary cases (one element, zero
/// elements) do not depend on a backend absorbing out-of-range stores. The
/// expected bytes are a pure function of the input, so any disagreement between
/// backends is a backend defect and not an ambiguity in the program.
pub(crate) fn xor_program(out: BufferDecl, len: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(len.max(1)),
            out,
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::u32(len)),
                vec![Node::store(
                    "out",
                    Expr::var("idx"),
                    Expr::bitxor(Expr::load("a", Expr::var("idx")), Expr::u32(0x0F)),
                )],
            ),
        ],
    )
}

/// Input bytes for `a`: the words `0..len`.
pub(crate) fn a_bytes(len: u32) -> Vec<u8> {
    (0..len).flat_map(u32::to_le_bytes).collect()
}

/// The bytes every path must return for [`xor_program`] over `len` elements.
pub(crate) fn expected_bytes(len: u32) -> Vec<u8> {
    (0..len).flat_map(|i| (i ^ 0x0F).to_le_bytes()).collect()
}

/// Inputs in the ABI every path shares: one slice per buffer the backend does
/// NOT allocate itself, in `Program::buffers()` order.
///
/// `vyre_reference::is_reference_input` is that predicate. Deriving the input
/// vector from it rather than hand-rolling it is what keeps the three backends
/// comparable: handing the reference a different seed length than the GPUs is
/// exactly the mistake that made one backend look correct and another look
/// truncated when both were applying the same rule.
pub(crate) fn inputs_for(program: &Program, seed_bytes: usize, len: u32) -> Vec<Vec<u8>> {
    program
        .buffers()
        .iter()
        .filter(|decl| vyre_reference::is_reference_input(decl))
        .map(|decl| match decl.name() {
            "a" => a_bytes(len),
            _ => vec![0u8; seed_bytes],
        })
        .collect()
}

pub(crate) fn run_reference(program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
    let values: Vec<Value> = inputs.iter().map(|b| Value::from(b.as_slice())).collect();
    vyre_reference::reference_eval(program, &values)
        .map(|outputs| outputs.into_iter().map(|value| value.to_bytes()).collect())
        .map_err(|error| error.to_string())
}

pub(crate) fn run_target(
    backend_id: &str,
    program: &Program,
    inputs: &[Vec<u8>],
) -> Result<Vec<Vec<u8>>, String> {
    let registration =
        vyre_driver::backend_registration(backend_id).map_err(|error| error.to_string())?;
    let production =
        ProductionSession::compile(program, registration).map_err(|error| error.to_string())?;
    let borrowed = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    production
        .submit(&borrowed)
        .map_err(|error| error.to_string())
}

pub(crate) fn run_cuda(program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
    run_target(vyre_driver_cuda::CUDA_BACKEND_ID, program, inputs)
}

pub(crate) fn run_wgpu(program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
    run_target(vyre_driver_wgpu::WGPU_BACKEND_ID, program, inputs)
}

/// Assert the three paths return byte-identical first output buffers.
///
/// Compares against `expected` as well as against each other, so a shared wrong
/// answer cannot pass.
pub(crate) fn assert_all_three_return(
    program: &Program,
    inputs: &[Vec<u8>],
    expected: &[u8],
    case: &str,
) {
    let reference = run_reference(program, inputs)
        .unwrap_or_else(|error| panic!("{case}: reference must dispatch, got: {error}"));
    let cuda = run_cuda(program, inputs)
        .unwrap_or_else(|error| panic!("{case}: CUDA must dispatch, got: {error}"));
    let wgpu = run_wgpu(program, inputs)
        .unwrap_or_else(|error| panic!("{case}: WGPU must dispatch, got: {error}"));

    assert_eq!(
        reference[0].len(),
        expected.len(),
        "{case}: reference returned {} bytes, expected {}",
        reference[0].len(),
        expected.len()
    );
    assert_eq!(reference[0], expected, "{case}: reference bytes");
    assert_eq!(
        cuda[0].len(),
        expected.len(),
        "{case}: CUDA returned {} bytes, expected {}",
        cuda[0].len(),
        expected.len()
    );
    assert_eq!(cuda[0], expected, "{case}: CUDA bytes");
    assert_eq!(
        wgpu[0].len(),
        expected.len(),
        "{case}: WGPU returned {} bytes, expected {}. A zero length here is the original defect.",
        wgpu[0].len(),
        expected.len()
    );
    assert_eq!(wgpu[0], expected, "{case}: WGPU bytes");
    assert_eq!(
        reference[0], cuda[0],
        "{case}: reference and CUDA must agree"
    );
    assert_eq!(cuda[0], wgpu[0], "{case}: CUDA and WGPU must agree");
}

/// Assert every path refuses `program`, and that each message names the remedy.
pub(crate) fn assert_all_three_refuse(
    program: &Program,
    inputs: &[Vec<u8>],
    case: &str,
) -> [String; 3] {
    let reference = run_reference(program, inputs)
        .err()
        .unwrap_or_else(|| panic!("{case}: reference must refuse, it returned Ok"));
    let cuda = run_cuda(program, inputs)
        .err()
        .unwrap_or_else(|| panic!("{case}: CUDA must refuse, it returned Ok"));
    let wgpu = run_wgpu(program, inputs)
        .err()
        .unwrap_or_else(|| panic!("{case}: WGPU must refuse, it returned Ok"));
    [reference, cuda, wgpu]
}
