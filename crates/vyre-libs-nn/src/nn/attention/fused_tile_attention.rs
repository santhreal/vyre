//! Fused attention using first-class IR Tile values.
//!
//! Holds score tiles and online-softmax statistics in `Register` and
//! `Subgroup` residency across two matrix operations with no intermediate buffer.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Ident, Layout, Node, Program, Residency,
    SubgroupReduceOp, Tile, UnOp,
};

const OP_ID: &str = "vyre-libs::nn::fused_tile_attention";

/// Build a Program that computes fused attention over tiles in register and subgroup residency.
///
/// Tensors are `[seq_len, head_dim]` row-major F32.
/// All intermediate score matrices and softmax statistics reside in Register / Subgroup residency.
#[must_use]
pub fn fused_tile_attention(
    q: &str,
    k: &str,
    v: &str,
    out: &str,
    seq_len: u32,
    head_dim: u32,
) -> Program {
    let elements = seq_len.saturating_mul(head_dim);

    let q_tile = Tile::new(
        DataType::F32,
        vec![seq_len, head_dim],
        Layout::RowMajor,
        Residency::Register,
    );
    let k_tile = Tile::new(
        DataType::F32,
        vec![head_dim, seq_len],
        Layout::ColumnMajor,
        Residency::Subgroup,
    );
    let score_tile = Tile::new(
        DataType::F32,
        vec![seq_len, seq_len],
        Layout::RowMajor,
        Residency::Register,
    );
    let v_tile = Tile::new(
        DataType::F32,
        vec![seq_len, head_dim],
        Layout::RowMajor,
        Residency::Subgroup,
    );
    let out_tile = Tile::new(
        DataType::F32,
        vec![seq_len, head_dim],
        Layout::RowMajor,
        Residency::Register,
    );

    let nodes = vec![
        // 1. Declare intermediate tiles in register residency
        Node::tile_decl("scores", score_tile.clone()),
        Node::tile_decl("o_tile", out_tile.clone()),
        // 2. Load Q and K tiles directly into registers and subgroup fragments
        Node::tile_load(
            "q_tile",
            q_tile,
            q,
            vec![Expr::u32(0), Expr::u32(0)],
            Layout::RowMajor,
        ),
        Node::tile_load(
            "k_tile",
            k_tile,
            k,
            vec![Expr::u32(0), Expr::u32(0)],
            Layout::ColumnMajor,
        ),
        // 3. First matrix multiplication: Scores = Q x K in registers
        Node::tile_matmul("scores", "q_tile", "k_tile"),
        // 4. Row max reduction in registers
        Node::tile_reduce("row_max", "scores", SubgroupReduceOp::Max, 1),
        // 5. Exponentiate scores in registers: exp(score - max)
        Node::tile_elementwise(
            "exp_scores",
            vec![Ident::from("scores"), Ident::from("row_max")],
            vec![Node::let_bind(
                "exp_scores",
                Expr::UnOp {
                    op: UnOp::Exp,
                    operand: Box::new(Expr::sub(Expr::var("scores"), Expr::var("row_max"))),
                },
            )],
        ),
        // 6. Row sum reduction of exponential scores in registers
        Node::tile_reduce("row_sum", "exp_scores", SubgroupReduceOp::Add, 1),
        // 7. Load V tile into subgroup residency
        Node::tile_load(
            "v_tile",
            v_tile,
            v,
            vec![Expr::u32(0), Expr::u32(0)],
            Layout::RowMajor,
        ),
        // 8. Second matrix multiplication: O = exp_scores x V in registers
        Node::tile_matmul("o_tile", "exp_scores", "v_tile"),
        // 9. Normalize output tile by row sum in registers
        Node::tile_elementwise(
            "final_out_tile",
            vec![Ident::from("o_tile"), Ident::from("row_sum")],
            vec![Node::let_bind(
                "final_out_tile",
                Expr::div(Expr::var("o_tile"), Expr::var("row_sum")),
            )],
        ),
        // 10. Store final output directly to output buffer
        Node::tile_store(out, vec![Expr::u32(0), Expr::u32(0)], "final_out_tile"),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(q, 0, BufferAccess::ReadOnly, DataType::F32).with_count(elements),
            BufferDecl::storage(k, 1, BufferAccess::ReadOnly, DataType::F32).with_count(elements),
            BufferDecl::storage(v, 2, BufferAccess::ReadOnly, DataType::F32).with_count(elements),
            BufferDecl::output(out, 3, DataType::F32).with_count(elements),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(OP_ID, nodes)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::validate::{BackendCapabilities, ValidationOptions};

    #[test]
    fn fused_tile_attention_builds_and_validates() {
        let prog = fused_tile_attention("q", "k", "v", "out", 4, 4);
        assert_eq!(prog.buffers().len(), 4);
        // Validates with tensor core support
        let caps = BackendCapabilities {
            supports_tensor_cores: true,
            max_shared_memory_bytes: 65536,
            regs_per_thread_max: 255,
            subgroup_size: 32,
            ..BackendCapabilities::default()
        };
        let res = vyre_foundation::validate::validate_with_options(
            &prog,
            ValidationOptions::default().with_backend_capabilities(caps),
        );
        assert!(res.is_ok(), "Validation failed: {:?}", res.errors);
    }
}
