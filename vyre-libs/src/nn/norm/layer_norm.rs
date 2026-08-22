//! Layer normalization  -  `y_i = (x_i - mean(x)) / sqrt(var(x) + eps)`.
//!
//! Category-A composition with a workgroup-tiled reduction. The public
//! [`layer_norm`] path computes sum and sum-of-squares in one tiled pass,
//! reduces both through workgroup scratch, and writes normalized output.
//!
//! ## API surface
//!
//! - [`LayerNorm`]  -  typed builder with [`TensorRef`]-accepting
//!   inputs and contract checks at [`LayerNorm::build`] time.
//! - [`layer_norm`]  -  name-based convenience constructor.
//!
//! Both paths emit byte-identical IR.

use crate::reduce::workgroup_tree::{self, WorkgroupReductionScope};
use vyre_foundation::composition::trap_program;
#[cfg(test)]
use vyre_foundation::composition::wrap_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

use crate::builder::reduction::{ReductionComposer, ReductionPhase};
use crate::builder::{
    check_same_shape, check_tensors, checked_element_count, strided_accumulate2_child,
    strided_writeback_child, BuildOptions,
};
use crate::plumbing::operand::tensor_ref::{TensorRef, TensorRefError};

const OP_ID: &str = "vyre-libs::nn::layer_norm";
#[cfg(test)]
const LAYER_NORM_REFERENCE_OP_ID: &str = "vyre-libs::nn::layer_norm_reference";

/// Typed Cat-A builder for [`layer_norm`].
#[derive(Debug, Clone)]
pub struct LayerNorm {
    input: TensorRef,
    output: TensorRef,
    eps: f32,
    options: BuildOptions,
}

impl LayerNorm {
    /// Start a builder. `eps` is the numerical-stability constant
    /// added under the sqrt to guard against zero variance.
    #[must_use]
    pub fn new(input: TensorRef, output: TensorRef, eps: f32) -> Self {
        Self {
            input,
            output,
            eps,
            options: BuildOptions::default(),
        }
    }

    /// Validate + materialize the Program.
    ///
    /// # Errors
    ///
    /// Surfaces the standard [`TensorRefError`] set (dtype, shape,
    /// name-collision, overflow).
    pub fn build(self) -> Result<Program, TensorRefError> {
        check_tensors(
            OP_ID,
            &[(&self.input, DataType::F32), (&self.output, DataType::F32)],
        )?;
        check_same_shape(OP_ID, &self.input, &self.output)?;
        // V7-CORR-008: reject negative or NaN eps so sqrt(var + eps) never
        // poisons the output with NaN. Positive zero is allowed (the
        // caller accepts exact-divide-by-zero risk on zero-variance input).
        if self.eps < 0.0 || self.eps.is_nan() {
            return Err(TensorRefError::ShapeMismatch {
                name: "eps".to_string(),
                found: Vec::new(),
                expected: Vec::new(),
                op: OP_ID,
            });
        }
        // V7-CORR-012/013 parallel: the nonzero floor `checked_element_count`
        // enforces is what keeps the first `Expr::load(input, 0)` in bounds.
        let n = checked_element_count(OP_ID, &self.input)?;
        let workgroup = self.options.workgroup_size.unwrap_or([256, 1, 1]);
        let tile = workgroup[0].max(1).min(n);
        let workgroup = [tile, workgroup[1], workgroup[2]];
        let chunks = n.div_ceil(tile);
        let input_name = self.input.name_str();
        let output_name = self.output.name_str();
        let generator = self.options.region_generator.unwrap_or(OP_ID);

        Ok(layer_norm_tiled_program(&LayerNormTiledSpec {
            input: input_name,
            output: output_name,
            n,
            eps: self.eps,
            tile,
            chunks,
            workgroup,
            generator,
        }))
    }
}

crate::builder::impl_cat_a_builder_options!(LayerNorm);

/// Shape, numerics, and codegen knobs for one tiled layer-norm build.
struct LayerNormTiledSpec<'a> {
    /// Input buffer name.
    input: &'a str,
    /// Output buffer name.
    output: &'a str,
    /// Element count.
    n: u32,
    /// Variance epsilon.
    eps: f32,
    /// Lanes per workgroup, capped at `n`.
    tile: u32,
    /// Strided chunks each lane walks.
    chunks: u32,
    /// Workgroup size.
    workgroup: [u32; 3],
    /// Region generator name.
    generator: &'static str,
}

fn layer_norm_tiled_program(spec: &LayerNormTiledSpec<'_>) -> Program {
    let LayerNormTiledSpec {
        input,
        output,
        n,
        eps,
        tile,
        chunks,
        workgroup,
        generator,
    } = *spec;
    let moments = ReductionPhase {
        accumulate: strided_accumulate2_child(
            OP_ID,
            tile,
            chunks,
            n,
            ("local_sum", Expr::f32(0.0), "ln_sum_scratch", |idx, acc| {
                Expr::add(acc, Expr::load(input, idx))
            }),
            (
                "local_sq_sum",
                Expr::f32(0.0),
                "ln_sq_scratch",
                |idx, acc| {
                    let value = Expr::load(input, idx);
                    Expr::add(acc, Expr::mul(value.clone(), value))
                },
            ),
        ),
        reductions: vec![
            workgroup_tree::sum_f32_child(
                OP_ID,
                tile,
                "ln_sum_scratch",
                WorkgroupReductionScope::FirstWorkgroup,
            ),
            workgroup_tree::sum_f32_child(
                OP_ID,
                tile,
                "ln_sq_scratch",
                WorkgroupReductionScope::FirstWorkgroup,
            ),
        ],
        publish: vec![
            Node::let_bind(
                "mean",
                Expr::div(
                    Expr::load("ln_sum_scratch", Expr::u32(0)),
                    Expr::f32(n as f32),
                ),
            ),
            Node::let_bind(
                "mean_sq",
                Expr::div(
                    Expr::load("ln_sq_scratch", Expr::u32(0)),
                    Expr::f32(n as f32),
                ),
            ),
            Node::let_bind(
                "variance",
                Expr::sub(
                    Expr::var("mean_sq"),
                    Expr::mul(Expr::var("mean"), Expr::var("mean")),
                ),
            ),
            Node::Store {
                buffer: "ln_stats".into(),
                index: Expr::u32(0),
                value: Expr::var("mean"),
            },
            Node::Store {
                buffer: "ln_stats".into(),
                index: Expr::u32(1),
                value: Expr::UnOp {
                    op: UnOp::InverseSqrt,
                    operand: Box::new(Expr::add(Expr::var("variance"), Expr::f32(eps))),
                },
            },
        ],
    };
    let writeback = strided_writeback_child(
        OP_ID,
        tile,
        chunks,
        n,
        output,
        vec![
            Node::let_bind("mean", Expr::load("ln_stats", Expr::u32(0))),
            Node::let_bind("scale", Expr::load("ln_stats", Expr::u32(1))),
        ],
        |idx| {
            Expr::mul(
                Expr::sub(Expr::load(input, idx), Expr::var("mean")),
                Expr::var("scale"),
            )
        },
    );

    ReductionComposer::new(
        generator,
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
            BufferDecl::workgroup("ln_sum_scratch", tile, DataType::F32),
            BufferDecl::workgroup("ln_sq_scratch", tile, DataType::F32),
            BufferDecl::workgroup("ln_stats", 2, DataType::F32),
            BufferDecl::output(output, 1, DataType::F32).with_count(n),
        ],
        workgroup,
    )
    .with_phase(moments)
    .with_writeback(writeback)
    .build()
}

#[cfg(test)]
fn layer_norm_reference_program(input: &str, output: &str, n: u32, eps: f32) -> Program {
    use vyre_foundation::ir::BinOp;

    let n_expr = Expr::u32(n);
    let n_f32 = Expr::f32(n as f32);

    let sum_loop = Node::loop_for(
        "i",
        Expr::u32(0),
        n_expr.clone(),
        vec![Node::assign(
            "sum_val",
            Expr::add(Expr::var("sum_val"), Expr::load(input, Expr::var("i"))),
        )],
    );

    let var_loop = Node::loop_for(
        "i",
        Expr::u32(0),
        n_expr.clone(),
        vec![
            Node::let_bind(
                "centered",
                Expr::BinOp {
                    op: BinOp::Sub,
                    left: Box::new(Expr::load(input, Expr::var("i"))),
                    right: Box::new(Expr::var("mean")),
                },
            ),
            Node::assign(
                "var_sum",
                Expr::add(
                    Expr::var("var_sum"),
                    Expr::mul(Expr::var("centered"), Expr::var("centered")),
                ),
            ),
        ],
    );

    let write_loop = Node::loop_for(
        "i",
        Expr::u32(0),
        n_expr,
        vec![Node::Store {
            buffer: output.into(),
            index: Expr::var("i"),
            value: Expr::BinOp {
                op: BinOp::Div,
                left: Box::new(Expr::BinOp {
                    op: BinOp::Sub,
                    left: Box::new(Expr::load(input, Expr::var("i"))),
                    right: Box::new(Expr::var("mean")),
                }),
                right: Box::new(Expr::var("inv_denom")),
            },
        }],
    );

    let body = vec![
        Node::let_bind("sum_val", Expr::f32(0.0)),
        sum_loop,
        Node::let_bind(
            "mean",
            Expr::BinOp {
                op: BinOp::Div,
                left: Box::new(Expr::var("sum_val")),
                right: Box::new(n_f32.clone()),
            },
        ),
        Node::let_bind("var_sum", Expr::f32(0.0)),
        var_loop,
        Node::let_bind(
            "variance",
            Expr::BinOp {
                op: BinOp::Div,
                left: Box::new(Expr::var("var_sum")),
                right: Box::new(n_f32),
            },
        ),
        Node::let_bind(
            "inv_denom",
            Expr::UnOp {
                op: UnOp::Sqrt,
                operand: Box::new(Expr::add(Expr::var("variance"), Expr::f32(eps))),
            },
        ),
        write_loop,
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
            BufferDecl::output(output, 1, DataType::F32).with_count(n),
        ],
        [1, 1, 1],
        vec![wrap_region(LAYER_NORM_REFERENCE_OP_ID, body, None)],
    )
}

/// Build a Program that layer-normalizes `input` into `output` across
/// `n` F32 elements through [`LayerNorm`]. Invalid inputs lower to a trap.
#[must_use]
pub fn layer_norm(input: &str, output: &str, n: u32, eps: f32) -> Program {
    LayerNorm::new(
        TensorRef::f32_1d(input, n),
        TensorRef::f32_1d(output, n),
        eps,
    )
    .build()
    .unwrap_or_else(|err| {
        trap_program(
            OP_ID,
            Some((output, DataType::F32)),
            format!("Fix: layer_norm build failed: {err}"),
        )
    })
}

const EXPECTED_LAYER_NORM_OUTPUT_BYTES: [u8; 16] = [0u8; 16];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::nn::layer_norm",
        || layer_norm("input", "output", 4, 1e-5),
        Some(|| {
            let input = [2.0f32, 2.0, 2.0, 2.0];
            vec![vec![
                vyre_primitives::wire::pack_f32_slice(&input),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_LAYER_NORM_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
    .with_tolerance(vyre_foundation::operation::TolerancePolicy::f32_ulp(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::assert_tiled_matches_reference;
    use crate::fixture_bytes::eval_f32;
    use crate::fixture_bytes::try_eval_bytes;

    #[test]
    fn builder_rejects_dtype_mismatch() {
        let err = LayerNorm::new(
            TensorRef::u32_1d("in", 4),
            TensorRef::f32_1d("out", 4),
            1e-5,
        )
        .build()
        .unwrap_err();
        assert!(matches!(err, TensorRefError::DtypeMismatch { .. }));
    }

    #[test]
    fn builder_rejects_shape_mismatch() {
        let err = LayerNorm::new(
            TensorRef::f32_1d("in", 4),
            TensorRef::f32_1d("out", 8),
            1e-5,
        )
        .build()
        .unwrap_err();
        assert!(matches!(err, TensorRefError::ShapeMismatch { .. }));
    }

    #[test]
    fn free_function_and_builder_produce_equal_programs_by_default() {
        let free = layer_norm("in", "out", 4, 1e-5);
        let built = LayerNorm::new(
            TensorRef::f32_1d("in", 4),
            TensorRef::f32_1d("out", 4),
            1e-5,
        )
        .build()
        .unwrap();
        assert_eq!(
            free.to_wire().unwrap(),
            built.to_wire().unwrap(),
            "free `layer_norm` and builder `LayerNorm::build` must be byte-identical"
        );
    }

    #[test]
    fn layer_norm_small_tensor_clamps_reduction_tile_to_live_lanes() {
        let program = layer_norm("input", "output", 4, 1e-5);
        assert_eq!(

            program.workgroup_size(),
            [4, 1, 1],
            "Fix: layer_norm must not emit a 256-lane scratch reduction for a 4-element tensor; CUDA may not initialize dead lanes before reduction."
        );
        let scratch = program
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "ln_sum_scratch")
            .expect("Fix: layer_norm must keep its sum scratch buffer.");
        assert_eq!(
            scratch.count(),
            4,
            "Fix: layer_norm scratch size must track the clamped live-lane tile."
        );
    }

    #[test]
    fn tiled_layer_norm_matches_scalar_reference_across_multiple_tiles() {
        let n = 777_u32;
        let eps = 1.0e-5_f32;
        let input = (0..n)
            .map(|i| ((i as f32) * 0.031).sin() * 2.5 + (i % 13) as f32 * 0.0625)
            .collect::<Vec<_>>();
        assert_tiled_matches_reference(
            "layer_norm",
            &input,
            1.0e-4,
            &layer_norm("input", "output", n, eps),
            &layer_norm_reference_program("input", "output", n, eps),
        );
    }

    // Adversarial float tests: expose tolerance misconfiguration gaps.

    #[test]
    fn layer_norm_very_small_variance_eps_dominates() {
        // All elements equal → var = 0, eps dominates.
        // output = (x - mean) / sqrt(eps) = 0 / sqrt(eps) = 0.
        let n = 4u32;
        let eps = 1e-5_f32;
        let input = [3.0f32; 4];
        let program = layer_norm("input", "output", n, eps);
        let out = eval_f32("layer_norm", &program, &[&input[..]], 4);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.abs() <= 1.0e-4,
                "layer_norm zero-variance output at {i} must be ~0, got {v}"
            );
        }
    }

    #[test]
    fn layer_norm_very_large_variance() {
        // Large magnitude elements: mean ≈ 0, var ≈ 1e40, sqrt(var) ≈ 1e20.
        // output ≈ x / 1e20, which should stay finite and in [-1, 1] roughly.
        let n = 4u32;
        let eps = 1e-5_f32;
        let input = [1e20f32, -1e20, 1e20, -1e20];
        let program = layer_norm("input", "output", n, eps);
        let out = eval_f32("layer_norm", &program, &[&input[..]], 4);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "layer_norm large-variance output at {i} must be finite, got {v}"
            );
            assert!(
                v.abs() <= 2.0,
                "layer_norm large-variance output at {i} should be roughly normalized, got {v}"
            );
        }
    }

    #[test]
    fn layer_norm_single_element() {
        // Single element: mean = x, var = 0.
        // output = (x - x) / sqrt(eps) = 0.
        let input = [5.0f32];
        let program = layer_norm("input", "output", 1, 1e-5);
        let out = eval_f32("layer_norm", &program, &[&input[..]], 1);
        assert!(
            out[0].abs() <= 1.0e-4,
            "layer_norm single element must be ~0, got {}",
            out[0]
        );
    }

    #[test]
    fn layer_norm_empty_tensor_traps() {
        // n=0 is rejected by the builder.
        let err = LayerNorm::new(
            TensorRef::f32_1d("input", 0),
            TensorRef::f32_1d("output", 0),
            1e-5,
        )
        .build()
        .expect_err("layer_norm n=0 must be rejected at build time");
        assert!(
            matches!(err, TensorRefError::ShapeMismatch { .. }),
            "layer_norm n=0 shape error: {err:?}"
        );
        let program = layer_norm("input", "output", 0, 1e-5);
        let eval_err = try_eval_bytes(&program, vec![vec![0u8; 4], vec![0u8; 4]])
            .expect_err("layer_norm n=0 must trap instead of producing output");
        let msg = eval_err.to_string();
        assert!(
            msg.contains("trap") || msg.contains("Fix:"),
            "layer_norm n=0 eval error: {msg}"
        );
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn layer_norm_output_mean_is_zero(input in prop::collection::vec(-1e10f32..1e10f32, 2..32)) {
            let n = input.len() as u32;
            let program = layer_norm("input", "output", n, 1e-5);
            let out = eval_f32("layer_norm", &program, &[&input[..]], input.len());
            let mean = out.iter().sum::<f32>() / out.len() as f32;
            prop_assert!(
                mean.abs() <= 1.0e-3 || mean.is_nan(),
                "layer_norm output mean must be ~0, got {mean}"
            );
        }
    }
}
