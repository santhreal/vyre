//! The reduce-then-normalize program skeleton shared by tiled normalizers.
//!
//! [`softmax`](crate::nn::attention::softmax) and
//! [`layer_norm`](crate::nn::norm::layer_norm) build the same program: bind the
//! lane id, run one or more strided accumulation passes, reduce each pass
//! through workgroup scratch, publish the reduced scalars from lane zero of
//! workgroup zero, then stream the normalized output back. Only the
//! accumulator step functions, the published statistics, and the writeback
//! expression differ.
//!
//! The skeleton lives at the `nn` level rather than under `attention/` or
//! `norm/` because both of those sub-dialects are consumers and neither owns
//! the other: `nn-attention` enables `nn-norm`, so hosting the skeleton in
//! `attention/` would make a normalization op depend on the attention
//! sub-dialect, and hosting it in `norm/` would let any future non-norm
//! consumer reach across a sibling dialect for it. `nn::rms` sits here for the
//! same reason.

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

/// A tiled reduce-then-normalize program.
pub(crate) struct TiledReduceProgram {
    /// Region generator name recorded on the wrapping region.
    pub(crate) generator: &'static str,
    /// Buffer declarations in binding order.
    pub(crate) buffers: Vec<BufferDecl>,
    /// Workgroup size.
    pub(crate) workgroup: [u32; 3],
    /// Reduction passes, run in order.
    pub(crate) phases: Vec<ReducePhase>,
    /// Final strided pass that writes the normalized output.
    pub(crate) writeback: Node,
}

/// Assemble a tiled reduce-then-normalize program.
pub(crate) fn tiled_reduce_program(spec: TiledReduceProgram) -> Program {
    let TiledReduceProgram {
        generator,
        buffers,
        workgroup,
        phases,
        writeback,
    } = spec;
    let mut body = vec![Node::let_bind("local", Expr::LocalId { axis: 0 })];
    for phase in phases {
        body.push(phase.accumulate);
        body.push(Node::barrier());
        body.extend(phase.reductions);
        if !phase.publish.is_empty() {
            body.push(Node::if_then(
                Expr::and(
                    Expr::is_first_workgroup(),
                    Expr::eq(Expr::var("local"), Expr::u32(0)),
                ),
                phase.publish,
            ));
            body.push(Node::barrier());
        }
    }
    body.push(writeback);
    Program::wrapped(buffers, workgroup, vec![wrap_region(generator, body, None)])
}
