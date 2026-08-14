//! PTX fixture builders for the CUDA NVRTC compile/execute gate.
//!
//! Descriptor scaffolding comes from `vyre_lower::descriptor_builder`. What
//! stays here is the one thing every fixture in this gate shares: verify, then
//! emit PTX, and treat either failing as a fixture defect.

use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::descriptor_builder::{
    binop, body, descriptor, global_ro, global_rw, global_wo, lit, load_global, op, store_global,
};
use vyre_lower::{KernelDescriptor, KernelOp, KernelOpKind, LiteralValue};

/// `LocalInvocationId` gives a fixture a value constant folding cannot
/// eliminate, so the emitted PTX still contains the op under test.
fn invocation_id(result: u32) -> KernelOp {
    op(KernelOpKind::LocalInvocationId, [0], result)
}

fn emit_ptx(desc: &KernelDescriptor) -> String {
    let verified = vyre_lower::verify_descriptor(desc)
        .unwrap_or_else(|failure| panic!("fixture `{}` failed verification: {failure:?}", desc.id));
    vyre_emit_ptx::emit(&verified)
        .unwrap_or_else(|error| panic!("fixture `{}` failed PTX emission: {error:?}", desc.id))
}

// `fixtures.rs` is itself loaded by `#[path]`, so an unadorned `mod` here
// resolves under `nvrtc_compile_gate/fixtures/`, a directory that has never
// existed: these three declarations named nothing and the target did not build.
#[path = "scalar_op_fixtures.rs"]
mod scalar_op_fixtures;
#[path = "vector_load_fixtures.rs"]
mod vector_load_fixtures;
#[path = "vector_store_fixtures.rs"]
mod vector_store_fixtures;

pub(crate) use scalar_op_fixtures::*;
pub(crate) use vector_load_fixtures::*;
pub(crate) use vector_store_fixtures::*;
