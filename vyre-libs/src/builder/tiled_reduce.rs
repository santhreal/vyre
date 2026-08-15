//! The reduce-then-publish program skeleton shared by tiled reductions.
//!
//! [`reduce_mean`](crate::math::reduce_mean), [`rms_norm`](crate::nn::rms_norm),
//! [`softmax`](crate::nn::attention::softmax) and
//! [`layer_norm`](crate::nn::norm::layer_norm) build the same program: bind the
//! lane id, run one or more strided accumulation passes, reduce each pass
//! through workgroup scratch, publish the reduced scalars from lane zero of
//! workgroup zero, then stream the normalized output back. Only the accumulator
//! step functions, the published statistics, and the writeback expression
//! differ.
//!
//! The skeleton lives beside the strided accumulate and writeback children it
//! composes, and above every dialect that reaches for it. `math` and `nn` are
//! separately gated dialects, so a skeleton hosted under either one would make
//! an operation in the other fail to build without a feature it does not need,
//! and hosting it under `nn/attention` or `nn/norm` would make one sub-dialect
//! depend on a sibling.

use vyre_foundation::composition::wrap_region;
use vyre_foundation::ir::{BufferDecl, Expr, Node, Program};

/// One reduction pass over the input.
pub(crate) struct ReducePhase {
    /// Strided accumulation child, built with one of the
    /// `strided_accumulate*_child` helpers.
    pub(crate) accumulate: Node,
    /// Workgroup-tree reduction children, one per scratch buffer the
    /// accumulation filled.
    pub(crate) reductions: Vec<Node>,
    /// Statistics lane zero of workgroup zero writes once the reductions have
    /// landed. An empty publish emits neither the guarded store nor the
    /// barrier that would fence it.
    pub(crate) publish: Vec<Node>,
}

/// A tiled reduce-then-publish program.
pub(crate) struct TiledReduceProgram {
    /// Region generator name recorded on the wrapping region.
    pub(crate) generator: &'static str,
    /// Buffer declarations in binding order.
    pub(crate) buffers: Vec<BufferDecl>,
    /// Workgroup size.
    pub(crate) workgroup: [u32; 3],
    /// Reduction passes, run in order.
    pub(crate) phases: Vec<ReducePhase>,
    /// Final strided pass that writes the normalized output. A reduction whose
    /// result is the published scalar itself has no writeback.
    pub(crate) writeback: Option<Node>,
}

/// Assemble a tiled reduce-then-publish program.
pub(crate) fn tiled_reduce_program(spec: TiledReduceProgram) -> Program {
    let TiledReduceProgram {
        generator,
        buffers,
        workgroup,
        phases,
        writeback,
    } = spec;
    let phase_count = phases.len();
    let mut body = vec![Node::let_bind("local", Expr::LocalId { axis: 0 })];
    for (index, phase) in phases.into_iter().enumerate() {
        body.push(phase.accumulate);
        body.push(Node::barrier());
        body.extend(phase.reductions);
        if phase.publish.is_empty() {
            continue;
        }
        body.push(Node::if_then(
            Expr::and(
                Expr::is_first_workgroup(),
                Expr::eq(Expr::var("local"), Expr::u32(0)),
            ),
            phase.publish,
        ));
        // The barrier fences the published scalars for whoever reads them. When
        // the publish is the last thing the program does, nothing reads them
        // and the barrier would be a synchronization every lane pays for a
        // value none of them loads.
        let read_later = index + 1 < phase_count || writeback.is_some();
        if read_later {
            body.push(Node::barrier());
        }
    }
    body.extend(writeback);
    Program::wrapped(buffers, workgroup, vec![wrap_region(generator, body, None)])
}
