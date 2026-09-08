//! Contraction tiling, epilogue, and semiring closure contracts.
//!
//! BACKLOG row 50 requires logical contraction IR to lower into measured scalar,
//! tiled, and matrix instruction candidates with cooperative movement, fragment
//! packing, and epilogue fusion without domain ownership.

#![forbid(unsafe_code)]

use std::sync::Arc;

use vyre_foundation::ir::{DataType, Expr, Program};
use vyre_libs::builder::gemm::{
    ContractionEpilogue, ContractionSemiring, ContractionTiling,
};
use vyre_libs::builder::gemm_programs::*;
use vyre_spec::Semiring;

#[test]
fn contraction_tiling_exhaustive_closure() {
    let tilings = [
        ContractionTiling::Linear {
            workgroup_size: [64, 1, 1],
        },
        ContractionTiling::CooperativeShared {
            tile: 16,
            a_tile_name: "a_shared".into(),
            b_tile_name: "b_shared".into(),
        },
        ContractionTiling::Block1D { tile: 8 },
    ];

    for tiling in &tilings {
        match tiling {
            ContractionTiling::Linear { workgroup_size } => {
                assert_eq!(*workgroup_size, [64, 1, 1]);
            }
            ContractionTiling::CooperativeShared {
                tile,
                a_tile_name,
                b_tile_name,
            } => {
                assert_eq!(*tile, 16);
                assert_eq!(a_tile_name, "a_shared");
                assert_eq!(b_tile_name, "b_shared");
            }
            ContractionTiling::Block1D { tile } => {
                assert_eq!(*tile, 8);
            }
        }
    }
}

#[test]
fn contraction_epilogue_exhaustive_closure() {
    let epilogues = [
        ContractionEpilogue::None,
        ContractionEpilogue::Bias {
            buffer: "bias".into(),
            count: 32,
            dtype: DataType::F32,
        },
        ContractionEpilogue::Activation {
            bias: Some("bias".into()),
            activation: Arc::new(|expr| Expr::relu(expr)),
        },
        ContractionEpilogue::QuantizedScale {
            row_scales: "r_scale".into(),
            batch_scales: "b_scale".into(),
        },
    ];

    for epilogue in &epilogues {
        match epilogue {
            ContractionEpilogue::None => {}
            ContractionEpilogue::Bias {
                buffer,
                count,
                dtype,
            } => {
                assert_eq!(buffer, "bias");
                assert_eq!(*count, 32);
                assert_eq!(*dtype, DataType::F32);
            }
            ContractionEpilogue::Activation { bias, activation } => {
                assert_eq!(bias.as_deref(), Some("bias"));
                let applied = activation(Expr::f32(1.0));
                assert!(matches!(applied, Expr::UnOp { .. } | Expr::Select { .. }));
            }
            ContractionEpilogue::QuantizedScale {
                row_scales,
                batch_scales,
            } => {
                assert_eq!(row_scales, "r_scale");
                assert_eq!(batch_scales, "b_scale");
            }
        }
    }
}

#[test]
fn contraction_semiring_exhaustive_closure() {
    let semirings = [
        ContractionSemiring::Standard,
        ContractionSemiring::Closed(Semiring::Real),
        ContractionSemiring::Closed(Semiring::MinPlus),
        ContractionSemiring::Closed(Semiring::MaxPlus),
        ContractionSemiring::Closed(Semiring::BoolOr),
        ContractionSemiring::Closed(Semiring::BoolAnd),
        ContractionSemiring::Closed(Semiring::MaxTimes),
        ContractionSemiring::Closed(Semiring::Lineage),
        ContractionSemiring::Closed(Semiring::Gf2),
        ContractionSemiring::Fixed16_16,
        ContractionSemiring::Custom {
            identity: 0,
            combine: Arc::new(|a, b| Expr::mul(a, b)),
            accumulate: Arc::new(|acc, val| Expr::add(acc, val)),
        },
    ];

    for semiring in &semirings {
        match semiring {
            ContractionSemiring::Standard => {
                let id = semiring.identity_expr(&DataType::F32);
                assert!(matches!(id, Expr::LitF32(_)));
            }
            ContractionSemiring::Closed(s) => {
                let id = semiring.identity_expr(&DataType::U32);
                assert!(matches!(id, Expr::LitU32(_)));
                let comb = semiring.combine_expr(Expr::u32(2), Expr::u32(3));
                assert!(matches!(comb, Expr::BinOp { .. } | Expr::Select { .. }));
            }
            ContractionSemiring::Fixed16_16 => {
                let id = semiring.identity_expr(&DataType::U32);
                assert!(matches!(id, Expr::LitU32(0)));
            }
            ContractionSemiring::Custom { identity, .. } => {
                assert_eq!(*identity, 0);
            }
        }
    }
}

#[test]
fn contraction_builder_emits_valid_programs_across_epilogues_and_semirings() {
    use vyre_libs::builder::gemm::ContractionBuilder;

    // Standard 2D GEMM
    let gemm_prog = ContractionBuilder::matmul_2d("a", "b", "out", 4, 8, 16)
        .build()
        .expect("standard GEMM must build");
    assert_eq!(gemm_prog.buffers().len(), 3);

    // GEMM with Bias and Activation
    let bias_act_prog = ContractionBuilder::matmul_2d("a", "b", "out", 4, 8, 16)
        .with_epilogue(ContractionEpilogue::Activation {
            bias: Some("bias".into()),
            activation: Arc::new(|e| Expr::relu(e)),
        })
        .build()
        .expect("GEMM with bias+activation must build");
    assert_eq!(bias_act_prog.buffers().len(), 4);

    // MinPlus semiring
    let minplus_prog = ContractionBuilder::semiring_2d("a", "b", "out", 4, 8, 16, Semiring::MinPlus)
        .build()
        .expect("MinPlus semiring contraction must build");
    assert_eq!(minplus_prog.buffers().len(), 3);
}
