//! Shared plumbing for the emit probes that drive a whole `Program` through
//! lowering into PTX.
//!
//! A probe owns its buffers, its body and its emit options. The region wrapper
//! it is measured inside, and the two-stage lower-then-emit pipeline it reaches
//! PTX through, are stated here once: a probe that wrapped its body differently
//! would no longer be comparing what its name claims, and a probe that reported
//! a lowering refusal as an emit refusal would attribute the failure to the
//! wrong stage.

use std::sync::Arc;

use vyre_emit_ptx::PtxEmitOptions;
use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferDecl, Node, Program};

/// Wrap `body` in one region named `generator` over `buffers`, dispatched across
/// 256 invocations. Everything outside `body` is fixed, so a difference in emit
/// outcome between two calls is attributable to the body alone.
pub(crate) fn region_program(
    generator: &str,
    buffers: Vec<BufferDecl>,
    body: Vec<Node>,
) -> Program {
    Program::wrapped(
        buffers,
        [256, 1, 1],
        vec![Node::Region {
            generator: Ident::from(generator),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}

/// Lower `program` and emit it under `options`, naming which of the two stages
/// refused so a probe cannot mistake a lowering refusal for an emit refusal.
pub(crate) fn lower_and_emit(program: &Program, options: PtxEmitOptions) -> Result<String, String> {
    let descriptor = vyre_lower::lower_verified(program)
        .map(|lowered| lowered.descriptor)
        .map_err(|error| format!("lower: {error:?}"))?;
    vyre_emit_ptx::emit_with_options(&descriptor, options).map_err(|error| format!("{error:?}"))
}
