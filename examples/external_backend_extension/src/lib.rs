//! Out-of-tree dispatch backend registration.
//!
//! A crate outside the vyre workspace implements [`vyre_driver::VyreBackend`],
//! submits one [`vyre_driver::BackendRegistration`] and one
//! [`vyre_driver::BackendCapability`], and the registry serves it to
//! [`vyre_driver::acquire`] exactly as it serves a driver crate shipped in the
//! workspace. `examples/external_ir_extension` registers a compile-only facet
//! whose factory has no device; this crate registers the dispatch facet.
//!
//! Program execution stays inside `vyre-reference`. This backend translates
//! buffers and delegates, so there is no second host interpreter.

#![forbid(unsafe_code)]

use std::collections::HashSet;

use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, OpId, Program};
use vyre_driver::{
    core_supported_ops, dialect_only_supported_ops, sealed, BackendCapability, BackendError,
    BackendPrecedence, BackendRegistration, DispatchConfig, VyreBackend,
};
use vyre_foundation::operation::TargetId;
use vyre_reference::value::Value;

/// Stable backend id this crate submits into the registry.
pub const BACKEND_ID: &str = "example-external-backend";
/// Validated target identity for the registration.
pub const TARGET_ID: TargetId = TargetId::expect_valid(BACKEND_ID);

/// Backend that evaluates a program through `vyre-reference`.
///
/// The caller passes one input row per declared buffer, in
/// `Program::buffers()` order, workgroup scratch excluded. A row count that
/// disagrees with the program is rejected rather than padded, because a
/// synthesized row is an output nobody wrote.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExternalBackend;

impl sealed::Sealed for ExternalBackend {}

impl VyreBackend for ExternalBackend {
    fn id(&self) -> &'static str {
        BACKEND_ID
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn supported_ops(&self) -> &HashSet<OpId> {
        core_supported_ops()
    }

    fn dispatch_borrowed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        let bound = program
            .buffers()
            .iter()
            .filter(|buffer| buffer.access() != BufferAccess::Workgroup)
            .count();
        if inputs.len() != bound {
            return Err(BackendError::new(format!(
                "{BACKEND_ID} received {} input row(s) for {bound} bound buffer(s). Fix: pass one row per declared buffer in Program::buffers() order, workgroup scratch excluded.",
                inputs.len()
            )));
        }
        let values: Vec<Value> = inputs.iter().map(|row| Value::from(*row)).collect();
        vyre_reference::reference_eval(program, &values)
            .map(|outputs| outputs.iter().map(Value::to_bytes).collect())
            .map_err(|error| {
                BackendError::new(format!(
                    "{BACKEND_ID} could not evaluate the program: {error}. Fix: validate the Program and the input buffer ABI before dispatch."
                ))
            })
    }
}

fn acquire_external_backend() -> Result<Box<dyn VyreBackend>, BackendError> {
    Ok(Box::new(ExternalBackend))
}

/// Program the probe dispatches: `out[i] = in[i] + 1` over four `u32`.
#[must_use]
pub fn build_probe_program() -> Program {
    let index = Expr::var("i");
    Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(4),
        ],
        [4, 1, 1],
        vec![
            Node::let_bind("i", Expr::InvocationId { axis: 0 }),
            Node::if_then(
                Expr::lt(index.clone(), Expr::buf_len("in")),
                vec![Node::store(
                    "out",
                    index.clone(),
                    Expr::add(Expr::load("in", index), Expr::u32(1)),
                )],
            ),
        ],
    )
}

/// Dispatches [`build_probe_program`] through the registered backend.
///
/// # Errors
///
/// Returns the registry error when this crate's registration did not reach the
/// registry, and the backend error when dispatch fails.
pub fn dispatch_probe(words: &[u32]) -> Result<Vec<u32>, BackendError> {
    let backend = vyre_driver::acquire(BACKEND_ID)?;
    let input: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let output = vec![0u8; input.len()];
    let outputs = backend.dispatch_borrowed(
        &build_probe_program(),
        &[input.as_slice(), output.as_slice()],
        &DispatchConfig::default(),
    )?;
    let last = outputs.last().ok_or_else(|| {
        BackendError::new(format!(
            "{BACKEND_ID} returned no output rows. Fix: keep the writable buffer in the program so the reference evaluator reports it."
        ))
    })?;
    Ok(last
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

inventory::submit! {
    BackendRegistration {
        id: BACKEND_ID,
        target_id: TARGET_ID,
        payload_format: None,
        reference_oracle: true,
        factory: acquire_external_backend,
        supported_ops: core_supported_ops,
        semantic_operations: dialect_only_supported_ops,
        target_compiler: None,
        materializer: None,
    }
}

inventory::submit! {
    BackendCapability {
        id: BACKEND_ID,
        dispatches: true,
    }
}

inventory::submit! {
    BackendPrecedence {
        id: BACKEND_ID,
        rank: 950,
    }
}
