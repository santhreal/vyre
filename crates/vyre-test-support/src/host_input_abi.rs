//! The host input ABI contract every concrete driver answers on a device.
//!
//! `BufferDecl::consumes_host_input` is the single definition of which
//! declarations a caller fills. `vyre-driver` proves the neutral half against
//! `BindingPlan`; this is the device half, where a live backend and the
//! reference interpreter must accept and refuse the same input counts for the
//! same program. Both halves read one program set and one canonical input
//! builder, stated here, so a backend suite states which backend it runs and
//! nothing else.

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, MemoryKind, Node, Program};
use vyre_reference::value::Value;

/// Element count every host-fed buffer in this set declares.
const ELEMENTS: u32 = 4;

/// What this contract needs of a backend: one dispatch that either runs or
/// states why it refused.
///
/// Every backend registered through `VyreBackend` gets this for free. A driver
/// whose device entry point is an inherent method states the bridge itself,
/// which is one line of its own and not a second copy of the contract.
pub trait HostInputDispatch {
    /// Run `program` over `inputs`, reporting a refusal as the text a caller
    /// reads.
    ///
    /// # Errors
    ///
    /// Returns the refusal text when the backend declines the dispatch.
    fn dispatch_host_inputs(&self, program: &Program, inputs: &[Vec<u8>]) -> Result<(), String>;
}

impl<B: VyreBackend + ?Sized> HostInputDispatch for B {
    fn dispatch_host_inputs(&self, program: &Program, inputs: &[Vec<u8>]) -> Result<(), String> {
        self.dispatch(program, inputs, &DispatchConfig::default())
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

/// A program declaring one read, one output and one read-write buffer, so two
/// of its three declarations consume host input.
#[must_use]
pub fn mixed_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("in_read", 0, DataType::U32).with_count(ELEMENTS),
            BufferDecl::output("out_write", 1, DataType::U32).with_count(ELEMENTS),
            BufferDecl::read_write("state_rw", 2, DataType::U32).with_count(ELEMENTS),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out_write",
            Expr::gid_x(),
            Expr::add(
                Expr::load("in_read", Expr::gid_x()),
                Expr::load("state_rw", Expr::gid_x()),
            ),
        )],
    )
}

/// Programs spanning the host-input counts a caller meets: none, one, several,
/// and the tiers that declare a buffer no caller fills.
#[must_use]
pub fn host_input_programs() -> Vec<Program> {
    vec![
        // One output only: no declaration consumes host input.
        Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(ELEMENTS)],
            [1, 1, 1],
            vec![Node::store("out", Expr::gid_x(), Expr::u32(42))],
        ),
        // One read and one output: one host input.
        Program::wrapped(
            vec![
                BufferDecl::read("in", 0, DataType::U32).with_count(ELEMENTS),
                BufferDecl::output("out", 1, DataType::U32).with_count(ELEMENTS),
            ],
            [1, 1, 1],
            vec![Node::store(
                "out",
                Expr::gid_x(),
                Expr::load("in", Expr::gid_x()),
            )],
        ),
        // A workgroup buffer sits beside a read and a read-write: two host inputs.
        Program::wrapped(
            vec![
                BufferDecl::read("in", 0, DataType::U32).with_count(ELEMENTS),
                BufferDecl::output("out", 1, DataType::U32).with_count(ELEMENTS),
                BufferDecl::read_write("acc", 2, DataType::U32).with_count(ELEMENTS),
                BufferDecl::workgroup("shared", 3, DataType::U32),
            ],
            [1, 1, 1],
            vec![
                Node::store("acc", Expr::gid_x(), Expr::load("in", Expr::gid_x())),
                Node::store("out", Expr::gid_x(), Expr::load("acc", Expr::gid_x())),
            ],
        ),
        // A uniform joins two reads: three host inputs.
        Program::wrapped(
            vec![
                BufferDecl::read("in1", 0, DataType::U32).with_count(ELEMENTS),
                BufferDecl::read("in2", 1, DataType::U32).with_count(ELEMENTS),
                BufferDecl::storage("params", 2, BufferAccess::Uniform, DataType::U32)
                    .with_count(1),
                BufferDecl::output("out", 3, DataType::U32).with_count(ELEMENTS),
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
        mixed_program(),
        // A shared-tier declaration is device memory, so no caller fills it.
        Program::wrapped(
            vec![
                BufferDecl::read("in", 0, DataType::U32).with_count(ELEMENTS),
                BufferDecl::storage("shared_tier", 1, BufferAccess::ReadOnly, DataType::U32)
                    .with_kind(MemoryKind::Shared)
                    .with_count(ELEMENTS),
                BufferDecl::output("out", 2, DataType::U32).with_count(ELEMENTS),
            ],
            [1, 1, 1],
            vec![Node::store(
                "out",
                Expr::gid_x(),
                Expr::load("in", Expr::gid_x()),
            )],
        ),
    ]
}

/// How many of `program`'s declarations a caller fills.
#[must_use]
pub fn host_input_count(program: &Program) -> usize {
    program
        .buffers()
        .iter()
        .filter(|decl| decl.consumes_host_input())
        .count()
}

/// One zeroed buffer per declaration of `program` that consumes host input,
/// each sized by that declaration.
#[must_use]
pub fn canonical_inputs(program: &Program) -> Vec<Vec<u8>> {
    program
        .buffers()
        .iter()
        .filter(|decl| decl.consumes_host_input())
        .map(|decl| {
            let bytes = decl.static_byte_len().ok().flatten().unwrap_or(16);
            vec![0_u8; bytes]
        })
        .collect()
}

/// The refusal wording a wrong input count produces.
fn both_counts(expected: usize, received: usize) -> String {
    format!("expected {expected} input buffer(s) from Program declarations but received {received}")
}

/// Assert `backend` and the reference interpreter accept and refuse the same
/// input counts for every program in [`host_input_programs`].
///
/// The backend refusal names both counts; the reference refusal names which
/// side of the count it is on.
///
/// # Panics
///
/// Panics when the two disagree on any program, or when either accepts a count
/// the declarations do not describe.
pub fn assert_backend_agrees_with_reference_on_input_counts(backend: &impl HostInputDispatch) {
    for program in &host_input_programs() {
        let expected = host_input_count(program);
        let exact = canonical_inputs(program);
        assert_eq!(
            exact.len(),
            expected,
            "Fix: the canonical input builder must fill one slot per consuming declaration."
        );

        let reference_exact: Vec<Value> = exact.iter().cloned().map(Value::from).collect();
        vyre_reference::ReferenceRequest::standard(program, &reference_exact)
            .outputs()
            .unwrap_or_else(|error| {
                panic!("Fix: the oracle must accept {expected} inputs: {error}")
            });
        backend
            .dispatch_host_inputs(program, &exact)
            .unwrap_or_else(|error| {
                panic!("Fix: the backend must accept {expected} inputs: {error}")
            });

        let mut over = exact.clone();
        over.push(vec![0_u8; 4]);
        assert_reference_refuses(program, &over, "unused input");
        assert_backend_refuses(backend, program, &over, expected);

        if expected > 0 {
            let under = exact[..expected - 1].to_vec();
            assert_reference_refuses(program, &under, "missing input");
            assert_backend_refuses(backend, program, &under, expected);
        }
    }
}

/// Assert the reference interpreter refuses `inputs` for `program` and says
/// which side of the declared count the list is on.
fn assert_reference_refuses(program: &Program, inputs: &[Vec<u8>], reason: &str) {
    let values: Vec<Value> = inputs.iter().cloned().map(Value::from).collect();
    let refusal = vyre_reference::ReferenceRequest::standard(program, &values)
        .outputs()
        .err()
        .unwrap_or_else(|| panic!("Fix: the oracle must refuse {} inputs.", inputs.len()));
    let text = refusal.to_string();
    assert!(
        text.contains(reason),
        "Fix: the oracle refusal must report {reason}, got: {text}"
    );
}

/// Assert `backend` refuses `inputs` for `program` and names both counts.
fn assert_backend_refuses(
    backend: &impl HostInputDispatch,
    program: &Program,
    inputs: &[Vec<u8>],
    expected: usize,
) {
    let text = backend
        .dispatch_host_inputs(program, inputs)
        .err()
        .unwrap_or_else(|| {
            panic!(
                "Fix: the backend must refuse {} inputs where {expected} are declared.",
                inputs.len()
            )
        });
    assert!(
        text.contains(&both_counts(expected, inputs.len())),
        "Fix: the backend refusal must state expected {expected} and received {}, got: {text}",
        inputs.len()
    );
}

/// Assert `backend` refuses a long-form input list carrying a placeholder for
/// the output it allocates itself, and a short list missing a declared input,
/// while the canonical list dispatches.
///
/// A placeholder slot is the shape a caller writes when it counts bindings
/// instead of host inputs, so it is the case the count check exists for.
///
/// # Panics
///
/// Panics when either wrong list dispatches, when a refusal omits a count, or
/// when the canonical list is refused.
pub fn assert_long_form_placeholder_is_refused(backend: &impl HostInputDispatch) {
    let program = mixed_program();
    let expected = host_input_count(&program);
    assert_eq!(
        expected, 2,
        "Fix: in_read and state_rw are the two declarations a caller fills."
    );

    let exact = canonical_inputs(&program);
    backend
        .dispatch_host_inputs(&program, &exact)
        .unwrap_or_else(|error| panic!("Fix: the canonical input list must dispatch: {error}"));

    // A placeholder for the backend-allocated output sits at binding index 1.
    let mut long_form = exact.clone();
    long_form.insert(1, vec![0_u8; 16]);
    assert_backend_refuses(backend, &program, &long_form, expected);

    let short_form = vec![exact[0].clone()];
    assert_backend_refuses(backend, &program, &short_form, expected);
}
