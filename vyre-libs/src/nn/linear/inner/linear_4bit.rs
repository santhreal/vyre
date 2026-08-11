//! Fused `linear_4bit` constructor  -  unpack-on-demand 4-bit quantized linear.
//!
//! Instead of materializing an unpacked f32 weight buffer, this kernel loads
//! the packed u32 weight, extracts the correct nibble inside the inner `k`
//! loop, and accumulates directly. This eliminates the 8× memory expansion
//! of a separate unpack dispatch.

use crate::region::wrap_anonymous;
use crate::{
    plan_matmul_kernel, F32MatmulMode, MatmulFallbackReason, MatmulKernelCapabilities,
    MatmulKernelPath, MatmulKernelPlan, MatrixShape,
};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_spec::{QuantizationScale, QuantizationZeroPoint};

const INT4_LINEAR_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
const AFFINE_GROUPED_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
const AFFINE_GROUPED_LANES_PER_OUTPUT: u32 = 32;
const AFFINE_GROUPED_WARPS_PER_WORKGROUP: u32 =
    AFFINE_GROUPED_WORKGROUP_SIZE[0] / AFFINE_GROUPED_LANES_PER_OUTPUT;
const AFFINE_GROUPED_OUTPUTS_PER_WORKGROUP: u32 = AFFINE_GROUPED_WARPS_PER_WORKGROUP;
const AFFINE_GROUPED_OP_ID: &str = "vyre-libs::nn::linear_4bit_affine_grouped";
const AFFINE_GROUPED_WEIGHT_TILE: &str = "linear_4bit_weight_tile";

/// Maximum absolute output drift allowed for grouped INT4 planner evidence tests.
pub const LINEAR_4BIT_AFFINE_GROUPED_OUTPUT_DRIFT_ABS_TOLERANCE: f32 = 1.0e-4;

/// Planner evidence for fused grouped INT4 linear versus dequantized matmul.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedLinear4BitPlannerEvidence {
    /// Input feature dimension.
    pub in_dim: u32,
    /// Output feature dimension.
    pub out_dim: u32,
    /// Quantization group size.
    pub group_size: u32,
    /// Number of quantization groups.
    pub group_count: u32,
    /// Packed INT4 weight bytes.
    pub packed_weight_bytes: u64,
    /// Bytes that a materialized f32 dequantized weight matrix would require.
    pub dequantized_weight_bytes: u64,
    /// Scale plus zero-point sidecar bytes.
    pub sidecar_bytes: u64,
    /// Bias bytes.
    pub bias_bytes: u64,
    /// Output bytes.
    pub output_bytes: u64,
    /// Dequantized weight bytes avoided by the fused path.
    pub dequant_bytes_elided: u64,
    /// Equivalent matmul planner M dimension.
    pub matmul_m: u32,
    /// Equivalent matmul planner K dimension.
    pub matmul_k: u32,
    /// Equivalent matmul planner N dimension.
    pub matmul_n: u32,
    /// Equivalent matmul planner K tile.
    pub matmul_tile: u32,
    /// Selected shared matmul planner path.
    pub matmul_selected_path: &'static str,
    /// Candidate tensor-core path from the shared matmul planner, when any.
    pub matmul_candidate_path: Option<&'static str>,
    /// Shared matmul planner fallback reason, when the selected path is cooperative.
    pub matmul_fallback_reason: Option<&'static str>,
    /// Whether the shared matmul planner selected a tensor-core path.
    pub tensor_core_eligible: bool,
    /// Maximum absolute output drift accepted by evidence tests.
    pub output_drift_abs_tolerance: f32,
}

/// Typed metadata for fused grouped INT4 linear.
///
/// The actual packed weight buffer is still addressed as `u32` words because
/// the kernel extracts eight nibbles per word. This spec binds that physical
/// layout to the first-class `DataType::Quantized` contract so call sites do
/// not pass an untyped integer buffer and lose the scale/zero-point semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedLinear4BitSpec {
    /// Input feature dimension.
    pub in_dim: u32,
    /// Output feature dimension.
    pub out_dim: u32,
    /// First-class quantized weight metadata.
    pub weight_type: DataType,
}

impl QuantizedLinear4BitSpec {
    /// Build a grouped affine INT4 metadata spec.
    #[must_use]
    pub fn affine_grouped(in_dim: u32, out_dim: u32, group_size: u32) -> Self {
        Self {
            in_dim,
            out_dim,
            weight_type: DataType::Quantized {
                storage: Box::new(DataType::I4),
                scale: QuantizationScale::PerGroup { group_size },
                zero_point: QuantizationZeroPoint::PerGroup { group_size },
            },
        }
    }

    fn affine_group_size(&self) -> Result<u32, String> {
        match &self.weight_type {
            DataType::Quantized {
                storage,
                scale: QuantizationScale::PerGroup { group_size },
                zero_point:
                    QuantizationZeroPoint::PerGroup {
                        group_size: zp_group_size,
                    },
            } => {
                if storage.as_ref() != &DataType::I4 {
                    return Err(format!(
                        "Fix: grouped INT4 linear requires DataType::Quantized storage I4, got {storage}."
                    ));
                }
                if group_size != zp_group_size {
                    return Err(format!(
                        "Fix: grouped INT4 linear requires scale and zero-point group sizes to match, got scale={group_size}, zero_point={zp_group_size}."
                    ));
                }
                if *group_size == 0 {
                    return Err(
                        "Fix: grouped INT4 linear requires quantized group_size > 0.".to_string()
                    );
                }
                Ok(*group_size)
            }
            other => Err(format!(
                "Fix: grouped INT4 linear requires DataType::Quantized<I4; PerGroup scale; PerGroup zero-point>, got {other}."
            )),
        }
    }
}

/// Build planner evidence for [`linear_4bit_affine_grouped_typed`].
///
/// # Errors
/// Returns `Err` when quantized metadata or dimensions are invalid.
pub fn linear_4bit_affine_grouped_planner_evidence(
    spec: &QuantizedLinear4BitSpec,
) -> Result<QuantizedLinear4BitPlannerEvidence, String> {
    let group_size = spec.affine_group_size()?;
    quantized_linear_4bit_planner_evidence(spec.in_dim, spec.out_dim, group_size)
}

fn quantized_linear_4bit_planner_evidence(
    in_dim: u32,
    out_dim: u32,
    group_size: u32,
) -> Result<QuantizedLinear4BitPlannerEvidence, String> {
    if in_dim == 0 {
        return Err(
            "Fix: linear_4bit_affine_grouped planner evidence requires in_dim > 0.".to_string(),
        );
    }
    if out_dim == 0 {
        return Err(
            "Fix: linear_4bit_affine_grouped planner evidence requires out_dim > 0.".to_string(),
        );
    }
    if group_size == 0 {
        return Err(
            "Fix: linear_4bit_affine_grouped planner evidence requires group_size > 0.".to_string(),
        );
    }
    if in_dim % 8 != 0 {
        return Err(format!(
            "Fix: linear_4bit_affine_grouped planner evidence in_dim={in_dim} is not divisible by 8."
        ));
    }

    let packed_words = (in_dim / 8).checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_4bit_affine_grouped planner evidence packed weights overflow u32.".to_string()
    })?;
    let group_count = in_dim.div_ceil(group_size);
    let sidecar_values = group_count.checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_4bit_affine_grouped planner evidence sidecars overflow u32.".to_string()
    })?;
    let matmul_shape = MatrixShape {
        m: out_dim,
        k: in_dim,
        n: 1,
    };
    let matmul_tile = AFFINE_GROUPED_LANES_PER_OUTPUT;
    let matmul_plan = plan_matmul_kernel(
        &DataType::F32,
        matmul_shape,
        matmul_tile,
        1,
        F32MatmulMode::StrictF32,
        MatmulKernelCapabilities::current_codegen(),
    );
    let dequantized_weight_bytes = u64::from(in_dim)
        .saturating_mul(u64::from(out_dim))
        .saturating_mul(core::mem::size_of::<f32>() as u64);
    let packed_weight_bytes = u64::from(packed_words) * core::mem::size_of::<u32>() as u64;
    let sidecar_bytes = u64::from(sidecar_values)
        .saturating_mul((core::mem::size_of::<f32>() + core::mem::size_of::<u32>()) as u64);
    let output_bytes = u64::from(out_dim) * core::mem::size_of::<f32>() as u64;

    Ok(QuantizedLinear4BitPlannerEvidence {
        in_dim,
        out_dim,
        group_size,
        group_count,
        packed_weight_bytes,
        dequantized_weight_bytes,
        sidecar_bytes,
        bias_bytes: output_bytes,
        output_bytes,
        dequant_bytes_elided: dequantized_weight_bytes,
        matmul_m: matmul_shape.m,
        matmul_k: matmul_shape.k,
        matmul_n: matmul_shape.n,
        matmul_tile,
        matmul_selected_path: matmul_path_label(matmul_plan.selected_path),
        matmul_candidate_path: matmul_plan.candidate_path.map(matmul_path_label),
        matmul_fallback_reason: matmul_fallback_label(&matmul_plan),
        tensor_core_eligible: matmul_plan.selected_path != MatmulKernelPath::Cooperative,
        output_drift_abs_tolerance: LINEAR_4BIT_AFFINE_GROUPED_OUTPUT_DRIFT_ABS_TOLERANCE,
    })
}

fn matmul_path_label(path: MatmulKernelPath) -> &'static str {
    match path {
        MatmulKernelPath::Cooperative => "cooperative",
        MatmulKernelPath::TensorCoreF16M16N8K16 => "tensor_core_f16_m16n8k16",
        MatmulKernelPath::TensorCoreBf16M16N8K16 => "tensor_core_bf16_m16n8k16",
        MatmulKernelPath::TensorCoreTf32M16N8K4 => "tensor_core_tf32_m16n8k4",
    }
}

fn matmul_fallback_label(plan: &MatmulKernelPlan) -> Option<&'static str> {
    match plan.fallback_reason {
        Some(MatmulFallbackReason::StrictF32Requested) => Some("strict_f32_requested"),
        Some(MatmulFallbackReason::UnsupportedDtype) => Some("unsupported_dtype"),
        Some(MatmulFallbackReason::TileSizeMismatch { .. }) => Some("tile_size_mismatch"),
        Some(MatmulFallbackReason::RaggedTileUnsupported) => Some("ragged_tile_unsupported"),
        Some(MatmulFallbackReason::SplitKUnsupported) => Some("split_k_unsupported"),
        Some(MatmulFallbackReason::TensorCoreDtypeUnsupported) => {
            Some("tensor_core_dtype_unsupported")
        }
        None => None,
    }
}

/// Build a Program that computes `out[i] = sum_k x[k] * unpack(w_packed[k,i]) + b[i]`
/// where `w_packed` stores 8 4-bit weights per u32.
///
/// `in_dim` must be divisible by 8 (each output column consumes `in_dim/8` u32s).
///
/// # Errors
/// Returns `Err` when `in_dim == 0` or `in_dim % 8 != 0`.
pub fn linear_4bit(
    x: &str,
    w_packed: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
) -> Result<Program, String> {
    if in_dim == 0 {
        return Err("Fix: linear_4bit in_dim=0 is invalid: empty reduction".to_string());
    }
    if out_dim == 0 {
        return Err("Fix: linear_4bit out_dim=0 is invalid: empty output".to_string());
    }
    if in_dim % 8 != 0 {
        return Err(format!(
            "Fix: linear_4bit in_dim={in_dim} is not divisible by 8; pad weights to a multiple of 8."
        ));
    }
    let u32s_per_col = in_dim / 8;
    let total_u32s = u32s_per_col.checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_4bit in_dim/8 * out_dim overflows u32; reduce dimensions.".to_string()
    })?;

    let i = Expr::var("i");
    let k = Expr::var("k");

    // packed_index = k / 8 * out_dim + i
    let packed_idx = Expr::add(
        Expr::mul(Expr::div(k.clone(), Expr::u32(8)), Expr::u32(out_dim)),
        i.clone(),
    );
    // nibble_shift = (k % 8) * 4
    let shift = Expr::mul(Expr::rem(k.clone(), Expr::u32(8)), Expr::u32(4));
    // unpacked_nibble = (w_packed[packed_idx] >> shift) & 0xF
    let nibble = Expr::bitand(
        Expr::shr(Expr::load(w_packed, packed_idx), shift),
        Expr::u32(0xF),
    );
    // cast to f32 for accumulation
    let weight_f32 = Expr::cast(DataType::F32, nibble);

    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::u32(out_dim)),
            vec![
                Node::let_bind("acc", Expr::load(b, i.clone())),
                Node::loop_for(
                    "k",
                    Expr::u32(0),
                    Expr::u32(in_dim),
                    vec![Node::assign(
                        "acc",
                        Expr::add(
                            Expr::var("acc"),
                            Expr::mul(Expr::load(x, k.clone()), weight_f32.clone()),
                        ),
                    )],
                ),
                Node::Store {
                    buffer: out.into(),
                    index: i,
                    value: Expr::var("acc"),
                },
            ],
        ),
    ];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(x, 0, BufferAccess::ReadOnly, DataType::F32).with_count(in_dim),
            BufferDecl::storage(w_packed, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(total_u32s),
            BufferDecl::storage(b, 2, BufferAccess::ReadOnly, DataType::F32).with_count(out_dim),
            BufferDecl::output(out, 3, DataType::F32).with_count(out_dim),
        ],
        INT4_LINEAR_WORKGROUP_SIZE,
        vec![wrap_anonymous("vyre-libs::nn::linear_4bit", body)],
    ))
}

/// Build a fused affine INT4 linear Program:
///
/// `out[i] = b[i] + sum_k x[k] * ((unpack4(w_packed[k,i]) - zero_point[group,i]) * scale[group,i])`
///
/// This keeps weights packed, applies per-group quantization metadata inside
/// the dot-product loop, and avoids a separate dequantize materialization
/// dispatch. `w_packed` stores 8 4-bit weights per u32 using the same
/// column-interleaved layout as [`linear_4bit`]. `scale` is f32, `zero_point`
/// is u32 with values expected in `0..=15`, and both sidecar buffers are
/// indexed as `group * out_dim + i`.
///
/// For bounded group counts the emitted IR hoists and lane-predicates each
/// scale/zero-point load, broadcasts the values once per `(group, output)`,
/// and expresses dequantization plus accumulation as fused multiply-adds.
/// This removes per-MAC group division, repeated sidecar loads, and redundant
/// affine arithmetic from the inference path.
///
/// # Errors
/// Returns `Err` when dimensions are empty, `group_size == 0`,
/// `in_dim % 8 != 0`, or derived sidecar/storage counts overflow `u32`.
pub fn linear_4bit_affine_grouped(
    x: &str,
    w_packed: &str,
    scale: &str,
    zero_point: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
    group_size: u32,
) -> Result<Program, String> {
    linear_4bit_affine_grouped_batch_impl(
        x, w_packed, scale, zero_point, b, out, in_dim, out_dim, group_size, 1,
    )
}

/// Build a batched fused affine INT4 linear Program with shared weights,
/// quantization sidecars, and bias.
///
/// The activation buffer contains `batch_size` contiguous `in_dim` rows. The
/// output buffer contains the corresponding contiguous `out_dim` rows.
///
/// # Errors
/// Returns `Err` under the same conditions as [`linear_4bit_affine_grouped`],
/// when `batch_size == 0`, or when a batched element count overflows `u32`.
pub fn linear_4bit_affine_grouped_batched(
    x: &str,
    w_packed: &str,
    scale: &str,
    zero_point: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
    group_size: u32,
    batch_size: u32,
) -> Result<Program, String> {
    linear_4bit_affine_grouped_batch_impl(
        x, w_packed, scale, zero_point, b, out, in_dim, out_dim, group_size, batch_size,
    )
}

fn linear_4bit_affine_grouped_batch_impl(
    x: &str,
    w_packed: &str,
    scale: &str,
    zero_point: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
    group_size: u32,
    batch_size: u32,
) -> Result<Program, String> {
    if in_dim == 0 {
        return Err(
            "Fix: linear_4bit_affine_grouped in_dim=0 is invalid: empty reduction".to_string(),
        );
    }
    if out_dim == 0 {
        return Err(
            "Fix: linear_4bit_affine_grouped out_dim=0 is invalid: empty output".to_string(),
        );
    }
    if group_size == 0 {
        return Err(
            "Fix: linear_4bit_affine_grouped group_size=0 is invalid: group size must be > 0"
                .to_string(),
        );
    }
    if batch_size == 0 {
        return Err(
            "Fix: linear_4bit_affine_grouped batch_size=0 is invalid: batch size must be > 0"
                .to_string(),
        );
    }
    let input_count = in_dim.checked_mul(batch_size).ok_or_else(|| {
        "Fix: linear_4bit_affine_grouped in_dim*batch_size overflows u32; reduce dimensions."
            .to_string()
    })?;
    let logical_output_count = out_dim.checked_mul(batch_size).ok_or_else(|| {
        "Fix: linear_4bit_affine_grouped out_dim*batch_size overflows u32; reduce dimensions."
            .to_string()
    })?;
    if in_dim % 8 != 0 {
        return Err(format!(
            "Fix: linear_4bit_affine_grouped in_dim={in_dim} is not divisible by 8; pad weights to a multiple of 8."
        ));
    }
    let u32s_per_col = in_dim / 8;
    let total_u32s = u32s_per_col.checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_4bit_affine_grouped in_dim/8 * out_dim overflows u32; reduce dimensions."
            .to_string()
    })?;
    let group_count = in_dim.div_ceil(group_size);
    let sidecar_count = group_count.checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_4bit_affine_grouped group_count*out_dim overflows u32; reduce dimensions."
            .to_string()
    })?;
    if batch_size >= AFFINE_GROUPED_WARPS_PER_WORKGROUP
        && batch_size % AFFINE_GROUPED_WARPS_PER_WORKGROUP == 0
        && in_dim == AFFINE_GROUPED_WORKGROUP_SIZE[0]
        && group_size >= AFFINE_GROUPED_LANES_PER_OUTPUT
        && group_size % AFFINE_GROUPED_LANES_PER_OUTPUT == 0
        && ![x, w_packed, scale, zero_point, b, out].contains(&AFFINE_GROUPED_WEIGHT_TILE)
    {
        return linear_4bit_affine_grouped_weight_reuse(
            x,
            w_packed,
            scale,
            zero_point,
            b,
            out,
            in_dim,
            out_dim,
            group_size,
            batch_size,
            input_count,
            total_u32s,
            sidecar_count,
            logical_output_count,
        );
    }

    let tile = AFFINE_GROUPED_LANES_PER_OUTPUT;
    let chunks = in_dim.div_ceil(tile);
    let out_idx = Expr::var("out_idx");
    let local = Expr::var("local");
    let lane = Expr::var("lane");
    let k = Expr::var("k");
    let activation_idx = Expr::add(
        Expr::mul(Expr::var("batch_idx"), Expr::u32(in_dim)),
        k.clone(),
    );
    let lane_in_word = Expr::var("lane_in_word");
    let word_leader_lane = Expr::var("word_leader_lane");
    let word_leader_k = Expr::var("word_leader_k");
    let packed_idx = Expr::add(
        Expr::mul(
            Expr::div(word_leader_k.clone(), Expr::u32(8)),
            Expr::u32(out_dim),
        ),
        out_idx.clone(),
    );
    let shift = Expr::mul(lane_in_word.clone(), Expr::u32(4));
    let nibble = Expr::bitand(Expr::shr(Expr::var("packed_word"), shift), Expr::u32(0xF));
    let group = Expr::div(k.clone(), Expr::u32(group_size));
    let chunk_sidecar_idx = Expr::add(Expr::mul(group, Expr::u32(out_dim)), out_idx.clone());
    let weight_f32 = Expr::fma(
        Expr::cast(DataType::F32, nibble),
        Expr::var("group_scale"),
        Expr::var("group_zero_offset"),
    );

    let mut per_output = vec![Node::let_bind("local_acc", Expr::f32(0.0))];
    if group_size > tile && group_size % tile == 0 {
        let group_chunks = group_size / tile;
        let build_chunk = |k_expr: Expr, word_leader_k_expr: Expr| {
            Node::Block(vec![
                Node::let_bind("k", k_expr),
                Node::let_bind("lane_in_word", Expr::bitand(lane.clone(), Expr::u32(7))),
                Node::let_bind(
                    "word_leader_lane",
                    Expr::bitand(lane.clone(), Expr::u32(0xffff_fff8)),
                ),
                Node::let_bind("word_leader_k", word_leader_k_expr),
                Node::let_bind(
                    "packed_word_lane",
                    Expr::select(
                        Expr::and(
                            Expr::eq(lane_in_word.clone(), Expr::u32(0)),
                            Expr::lt(word_leader_k.clone(), Expr::u32(in_dim)),
                        ),
                        Expr::load(w_packed, packed_idx.clone()),
                        Expr::u32(0),
                    ),
                ),
                Node::let_bind(
                    "packed_word",
                    Expr::subgroup_shuffle(Expr::var("packed_word_lane"), word_leader_lane.clone()),
                ),
                Node::if_then(
                    Expr::lt(k.clone(), Expr::u32(in_dim)),
                    vec![Node::assign(
                        "local_acc",
                        Expr::fma(
                            Expr::load(x, activation_idx.clone()),
                            weight_f32.clone(),
                            Expr::var("local_acc"),
                        ),
                    )],
                ),
            ])
        };
        let build_group = |group_base_expr: Expr, sidecar_idx_expr: Expr, chunk_scan: Node| {
            Node::Block(vec![
                Node::let_bind("group_base", group_base_expr),
                Node::let_bind("sidecar_idx", sidecar_idx_expr),
                Node::let_bind(
                    "scale_lane",
                    Expr::select(
                        Expr::eq(lane.clone(), Expr::u32(0)),
                        Expr::load(scale, Expr::var("sidecar_idx")),
                        Expr::f32(0.0),
                    ),
                ),
                Node::let_bind(
                    "zero_point_lane",
                    Expr::select(
                        Expr::eq(lane.clone(), Expr::u32(0)),
                        Expr::load(zero_point, Expr::var("sidecar_idx")),
                        Expr::u32(0),
                    ),
                ),
                Node::let_bind(
                    "group_scale",
                    Expr::subgroup_shuffle(Expr::var("scale_lane"), Expr::u32(0)),
                ),
                Node::let_bind(
                    "group_zero_point",
                    Expr::subgroup_shuffle(Expr::var("zero_point_lane"), Expr::u32(0)),
                ),
                Node::let_bind(
                    "negative_group_scale",
                    Expr::fma(Expr::f32(-1.0), Expr::var("group_scale"), Expr::f32(-0.0)),
                ),
                Node::let_bind(
                    "group_zero_offset",
                    Expr::fma(
                        Expr::cast(DataType::F32, Expr::var("group_zero_point")),
                        Expr::var("negative_group_scale"),
                        Expr::f32(-0.0),
                    ),
                ),
                chunk_scan,
            ])
        };
        let runtime_chunk = build_chunk(
            Expr::add(
                Expr::var("group_base"),
                Expr::add(
                    Expr::mul(Expr::var("group_chunk"), Expr::u32(tile)),
                    lane.clone(),
                ),
            ),
            Expr::add(
                Expr::var("group_base"),
                Expr::add(
                    Expr::mul(Expr::var("group_chunk"), Expr::u32(tile)),
                    word_leader_lane.clone(),
                ),
            ),
        );
        per_output.push(Node::loop_for(
            "group_idx",
            Expr::u32(0),
            Expr::u32(group_count),
            vec![build_group(
                Expr::mul(Expr::var("group_idx"), Expr::u32(group_size)),
                Expr::add(
                    Expr::mul(Expr::var("group_idx"), Expr::u32(out_dim)),
                    out_idx.clone(),
                ),
                Node::loop_for(
                    "group_chunk",
                    Expr::u32(0),
                    Expr::u32(group_chunks),
                    vec![runtime_chunk],
                ),
            )],
        ));
    } else {
        per_output.push(Node::loop_for(
            "chunk",
            Expr::u32(0),
            Expr::u32(chunks),
            vec![
                Node::let_bind(
                    "k",
                    Expr::add(Expr::mul(Expr::var("chunk"), Expr::u32(tile)), lane.clone()),
                ),
                Node::let_bind("lane_in_word", Expr::bitand(lane.clone(), Expr::u32(7))),
                Node::let_bind(
                    "word_leader_lane",
                    Expr::bitand(lane.clone(), Expr::u32(0xffff_fff8)),
                ),
                Node::let_bind(
                    "word_leader_k",
                    Expr::add(
                        Expr::mul(Expr::var("chunk"), Expr::u32(tile)),
                        word_leader_lane.clone(),
                    ),
                ),
                Node::let_bind(
                    "packed_word_lane",
                    Expr::select(
                        Expr::and(
                            Expr::eq(lane_in_word.clone(), Expr::u32(0)),
                            Expr::lt(word_leader_k.clone(), Expr::u32(in_dim)),
                        ),
                        Expr::load(w_packed, packed_idx),
                        Expr::u32(0),
                    ),
                ),
                Node::let_bind(
                    "packed_word",
                    Expr::subgroup_shuffle(Expr::var("packed_word_lane"), word_leader_lane),
                ),
                Node::let_bind("sidecar_idx", chunk_sidecar_idx),
                Node::let_bind("group_scale", Expr::load(scale, Expr::var("sidecar_idx"))),
                Node::let_bind(
                    "group_zero_point",
                    Expr::load(zero_point, Expr::var("sidecar_idx")),
                ),
                Node::let_bind(
                    "negative_group_scale",
                    Expr::fma(Expr::f32(-1.0), Expr::var("group_scale"), Expr::f32(-0.0)),
                ),
                Node::let_bind(
                    "group_zero_offset",
                    Expr::fma(
                        Expr::cast(DataType::F32, Expr::var("group_zero_point")),
                        Expr::var("negative_group_scale"),
                        Expr::f32(-0.0),
                    ),
                ),
                Node::if_then(
                    Expr::lt(k.clone(), Expr::u32(in_dim)),
                    vec![Node::assign(
                        "local_acc",
                        Expr::fma(
                            Expr::load(x, activation_idx.clone()),
                            weight_f32,
                            Expr::var("local_acc"),
                        ),
                    )],
                ),
            ],
        ));
    }
    per_output.push(Node::let_bind(
        "warp_sum",
        Expr::subgroup_add(Expr::var("local_acc")),
    ));
    per_output.push(Node::if_then(
        Expr::eq(lane.clone(), Expr::u32(0)),
        vec![Node::Store {
            buffer: out.into(),
            index: Expr::var("linear_out_idx"),
            value: Expr::add(Expr::load(b, out_idx.clone()), Expr::var("warp_sum")),
        }],
    ));

    let body = vec![
        Node::let_bind("local", Expr::LocalId { axis: 0 }),
        Node::let_bind(
            "warp",
            Expr::div(local.clone(), Expr::u32(AFFINE_GROUPED_LANES_PER_OUTPUT)),
        ),
        Node::let_bind(
            "lane",
            Expr::rem(local.clone(), Expr::u32(AFFINE_GROUPED_LANES_PER_OUTPUT)),
        ),
        Node::let_bind(
            "linear_out_idx",
            Expr::add(
                Expr::mul(
                    Expr::WorkgroupId { axis: 0 },
                    Expr::u32(AFFINE_GROUPED_OUTPUTS_PER_WORKGROUP),
                ),
                Expr::var("warp"),
            ),
        ),
        Node::let_bind(
            "batch_idx",
            Expr::div(Expr::var("linear_out_idx"), Expr::u32(out_dim)),
        ),
        Node::let_bind(
            "out_idx",
            Expr::rem(Expr::var("linear_out_idx"), Expr::u32(out_dim)),
        ),
        Node::if_then(
            Expr::lt(Expr::var("linear_out_idx"), Expr::u32(logical_output_count)),
            per_output,
        ),
    ];
    let output_workgroups = logical_output_count.div_ceil(AFFINE_GROUPED_OUTPUTS_PER_WORKGROUP);
    let padded_output_count = output_workgroups
        .checked_mul(AFFINE_GROUPED_WORKGROUP_SIZE[0])
        .ok_or_else(|| {
            "Fix: linear_4bit_affine_grouped output workgroups overflow u32; reduce dimensions."
                .to_string()
        })?;
    let output_byte_len = (logical_output_count as usize)
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| {
            "Fix: linear_4bit_affine_grouped output byte length overflows usize; reduce dimensions."
                .to_string()
        })?;

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(x, 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count(input_count),
            BufferDecl::storage(w_packed, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(total_u32s),
            BufferDecl::storage(scale, 2, BufferAccess::ReadOnly, DataType::F32)
                .with_count(sidecar_count),
            BufferDecl::storage(zero_point, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(sidecar_count),
            BufferDecl::storage(b, 4, BufferAccess::ReadOnly, DataType::F32).with_count(out_dim),
            BufferDecl::output(out, 5, DataType::F32)
                .with_count(padded_output_count)
                .with_output_byte_range(0..output_byte_len),
        ],
        AFFINE_GROUPED_WORKGROUP_SIZE,
        vec![wrap_anonymous(AFFINE_GROUPED_OP_ID, body)],
    ))
}

#[allow(clippy::too_many_arguments)]
fn linear_4bit_affine_grouped_weight_reuse(
    x: &str,
    w_packed: &str,
    scale: &str,
    zero_point: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
    group_size: u32,
    batch_size: u32,
    input_count: u32,
    total_u32s: u32,
    sidecar_count: u32,
    logical_output_count: u32,
) -> Result<Program, String> {
    let batch_tiles = batch_size / AFFINE_GROUPED_WARPS_PER_WORKGROUP;
    let output_workgroups = batch_tiles.checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_4bit_affine_grouped batched output workgroups overflow u32; reduce dimensions."
            .to_string()
    })?;
    let padded_output_count = output_workgroups
        .checked_mul(AFFINE_GROUPED_WORKGROUP_SIZE[0])
        .ok_or_else(|| {
            "Fix: linear_4bit_affine_grouped batched padded output count overflows u32; reduce dimensions."
                .to_string()
        })?;
    let output_byte_len = (logical_output_count as usize)
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| {
            "Fix: linear_4bit_affine_grouped batched output byte length overflows usize; reduce dimensions."
                .to_string()
        })?;
    let tile = AFFINE_GROUPED_LANES_PER_OUTPUT;
    let chunks = in_dim / tile;
    let local = Expr::var("local");
    let lane = Expr::var("lane");
    let weight_k = Expr::var("weight_k");
    let out_idx = Expr::var("out_idx");
    let lane_in_word = Expr::var("lane_in_word");
    let word_leader_lane = Expr::var("word_leader_lane");
    let word_leader_k = Expr::var("word_leader_k");
    let packed_idx = Expr::add(
        Expr::mul(Expr::div(word_leader_k, Expr::u32(8)), Expr::u32(out_dim)),
        out_idx.clone(),
    );
    let shift = Expr::mul(lane_in_word.clone(), Expr::u32(4));
    let nibble = Expr::bitand(Expr::shr(Expr::var("packed_word"), shift), Expr::u32(0xF));
    let sidecar_idx = Expr::add(
        Expr::mul(
            Expr::div(weight_k.clone(), Expr::u32(group_size)),
            Expr::u32(out_dim),
        ),
        out_idx.clone(),
    );
    let body = vec![
        Node::let_bind("local", Expr::LocalId { axis: 0 }),
        Node::let_bind(
            "warp",
            Expr::div(local.clone(), Expr::u32(AFFINE_GROUPED_LANES_PER_OUTPUT)),
        ),
        Node::let_bind(
            "lane",
            Expr::rem(local.clone(), Expr::u32(AFFINE_GROUPED_LANES_PER_OUTPUT)),
        ),
        Node::let_bind(
            "out_idx",
            Expr::rem(Expr::WorkgroupId { axis: 0 }, Expr::u32(out_dim)),
        ),
        Node::let_bind(
            "batch_tile",
            Expr::div(Expr::WorkgroupId { axis: 0 }, Expr::u32(out_dim)),
        ),
        Node::let_bind(
            "batch_idx",
            Expr::add(
                Expr::mul(
                    Expr::var("batch_tile"),
                    Expr::u32(AFFINE_GROUPED_WARPS_PER_WORKGROUP),
                ),
                Expr::var("warp"),
            ),
        ),
        Node::let_bind(
            "linear_out_idx",
            Expr::add(
                Expr::mul(Expr::var("batch_idx"), Expr::u32(out_dim)),
                out_idx.clone(),
            ),
        ),
        Node::let_bind("weight_k", local.clone()),
        Node::let_bind("lane_in_word", Expr::bitand(lane.clone(), Expr::u32(7))),
        Node::let_bind(
            "word_leader_lane",
            Expr::bitand(lane.clone(), Expr::u32(0xffff_fff8)),
        ),
        Node::let_bind(
            "word_leader_k",
            Expr::bitand(local.clone(), Expr::u32(0xffff_fff8)),
        ),
        Node::let_bind(
            "packed_word_lane",
            Expr::select(
                Expr::eq(lane_in_word.clone(), Expr::u32(0)),
                Expr::load(w_packed, packed_idx),
                Expr::u32(0),
            ),
        ),
        Node::let_bind(
            "packed_word",
            Expr::subgroup_shuffle(Expr::var("packed_word_lane"), word_leader_lane),
        ),
        Node::let_bind("sidecar_idx", sidecar_idx),
        Node::let_bind(
            "scale_lane",
            Expr::select(
                Expr::eq(lane.clone(), Expr::u32(0)),
                Expr::load(scale, Expr::var("sidecar_idx")),
                Expr::f32(0.0),
            ),
        ),
        Node::let_bind(
            "zero_point_lane",
            Expr::select(
                Expr::eq(lane.clone(), Expr::u32(0)),
                Expr::load(zero_point, Expr::var("sidecar_idx")),
                Expr::u32(0),
            ),
        ),
        Node::let_bind(
            "group_scale",
            Expr::subgroup_shuffle(Expr::var("scale_lane"), Expr::u32(0)),
        ),
        Node::let_bind(
            "group_zero_point",
            Expr::subgroup_shuffle(Expr::var("zero_point_lane"), Expr::u32(0)),
        ),
        Node::let_bind(
            "negative_group_scale",
            Expr::fma(Expr::f32(-1.0), Expr::var("group_scale"), Expr::f32(-0.0)),
        ),
        Node::let_bind(
            "group_zero_offset",
            Expr::fma(
                Expr::cast(DataType::F32, Expr::var("group_zero_point")),
                Expr::var("negative_group_scale"),
                Expr::f32(-0.0),
            ),
        ),
        Node::let_bind(
            "weight_value",
            Expr::fma(
                Expr::cast(DataType::F32, nibble),
                Expr::var("group_scale"),
                Expr::var("group_zero_offset"),
            ),
        ),
        Node::store(
            AFFINE_GROUPED_WEIGHT_TILE,
            local.clone(),
            Expr::var("weight_value"),
        ),
        Node::barrier(),
        Node::let_bind("local_acc", Expr::f32(0.0)),
        Node::loop_for(
            "chunk",
            Expr::u32(0),
            Expr::u32(chunks),
            vec![
                Node::let_bind(
                    "dot_k",
                    Expr::add(Expr::mul(Expr::var("chunk"), Expr::u32(tile)), lane.clone()),
                ),
                Node::assign(
                    "local_acc",
                    Expr::fma(
                        Expr::load(
                            x,
                            Expr::add(
                                Expr::mul(Expr::var("batch_idx"), Expr::u32(in_dim)),
                                Expr::var("dot_k"),
                            ),
                        ),
                        Expr::load(AFFINE_GROUPED_WEIGHT_TILE, Expr::var("dot_k")),
                        Expr::var("local_acc"),
                    ),
                ),
            ],
        ),
        Node::let_bind("warp_sum", Expr::subgroup_add(Expr::var("local_acc"))),
        Node::if_then(
            Expr::eq(lane, Expr::u32(0)),
            vec![Node::store(
                out,
                Expr::var("linear_out_idx"),
                Expr::add(Expr::load(b, out_idx), Expr::var("warp_sum")),
            )],
        ),
    ];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(x, 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count(input_count),
            BufferDecl::storage(w_packed, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(total_u32s),
            BufferDecl::storage(scale, 2, BufferAccess::ReadOnly, DataType::F32)
                .with_count(sidecar_count),
            BufferDecl::storage(zero_point, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(sidecar_count),
            BufferDecl::storage(b, 4, BufferAccess::ReadOnly, DataType::F32).with_count(out_dim),
            BufferDecl::workgroup(AFFINE_GROUPED_WEIGHT_TILE, in_dim, DataType::F32),
            BufferDecl::output(out, 5, DataType::F32)
                .with_count(padded_output_count)
                .with_output_byte_range(0..output_byte_len),
        ],
        AFFINE_GROUPED_WORKGROUP_SIZE,
        vec![wrap_anonymous(AFFINE_GROUPED_OP_ID, body)],
    ))
}

/// Build [`linear_4bit_affine_grouped`] from first-class quantized metadata.
///
/// # Errors
/// Returns `Err` when the spec is not `Quantized<I4; PerGroup; PerGroup>`,
/// when scale/zero-point group sizes differ, or when dimensions are invalid.
pub fn linear_4bit_affine_grouped_typed(
    spec: &QuantizedLinear4BitSpec,
    x: &str,
    w_packed: &str,
    scale: &str,
    zero_point: &str,
    b: &str,
    out: &str,
) -> Result<Program, String> {
    let group_size = spec.affine_group_size()?;
    linear_4bit_affine_grouped(
        x,
        w_packed,
        scale,
        zero_point,
        b,
        out,
        spec.in_dim,
        spec.out_dim,
        group_size,
    )
}

/// Build [`linear_4bit_affine_grouped_batched`] from first-class quantized
/// metadata.
///
/// # Errors
/// Returns `Err` when quantized metadata, dimensions, or `batch_size` are
/// invalid.
pub fn linear_4bit_affine_grouped_batched_typed(
    spec: &QuantizedLinear4BitSpec,
    batch_size: u32,
    x: &str,
    w_packed: &str,
    scale: &str,
    zero_point: &str,
    b: &str,
    out: &str,
) -> Result<Program, String> {
    let group_size = spec.affine_group_size()?;
    linear_4bit_affine_grouped_batched(
        x,
        w_packed,
        scale,
        zero_point,
        b,
        out,
        spec.in_dim,
        spec.out_dim,
        group_size,
        batch_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::byte_pack::f32_bytes;
    use crate::test_support::byte_pack::u32_bytes;
    use vyre_reference::value::Value;

    fn expr_contains_subgroup_shuffle(expr: &Expr) -> bool {
        match expr {
            Expr::Load { index, .. }
            | Expr::Cast { value: index, .. }
            | Expr::SubgroupReduce { value: index, .. }
            | Expr::SubgroupBallot { cond: index }
            | Expr::UnOp { operand: index, .. } => expr_contains_subgroup_shuffle(index),
            Expr::BinOp { left, right, .. }
            | Expr::SubgroupShuffle {
                value: left,
                lane: right,
            } => {
                matches!(expr, Expr::SubgroupShuffle { .. })
                    || expr_contains_subgroup_shuffle(left)
                    || expr_contains_subgroup_shuffle(right)
            }
            Expr::Select {
                cond,
                true_val,
                false_val,
            } => {
                expr_contains_subgroup_shuffle(cond)
                    || expr_contains_subgroup_shuffle(true_val)
                    || expr_contains_subgroup_shuffle(false_val)
            }
            Expr::Fma { a, b, c } => {
                expr_contains_subgroup_shuffle(a)
                    || expr_contains_subgroup_shuffle(b)
                    || expr_contains_subgroup_shuffle(c)
            }
            Expr::Atomic {
                index,
                expected,
                value,
                ..
            } => {
                expr_contains_subgroup_shuffle(index)
                    || expected
                        .as_deref()
                        .is_some_and(expr_contains_subgroup_shuffle)
                    || expr_contains_subgroup_shuffle(value)
            }
            Expr::Call { args, .. } => args.iter().any(expr_contains_subgroup_shuffle),
            Expr::LitU32(_)
            | Expr::LitI32(_)
            | Expr::LitF32(_)
            | Expr::LitBool(_)
            | Expr::Var(_)
            | Expr::BufLen { .. }
            | Expr::InvocationId { .. }
            | Expr::WorkgroupId { .. }
            | Expr::LocalId { .. }
            | Expr::SubgroupLocalId
            | Expr::SubgroupSize
            | Expr::Opaque(_) => false,
            _ => false,
        }
    }

    fn nodes_contain_subgroup_shuffle(nodes: &[Node]) -> bool {
        nodes.iter().any(|node| match node {
            Node::Let { value, .. } | Node::Assign { value, .. } => {
                expr_contains_subgroup_shuffle(value)
            }
            Node::Store { index, value, .. } => {
                expr_contains_subgroup_shuffle(index) || expr_contains_subgroup_shuffle(value)
            }
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                expr_contains_subgroup_shuffle(cond)
                    || nodes_contain_subgroup_shuffle(then)
                    || nodes_contain_subgroup_shuffle(otherwise)
            }
            Node::Loop { from, to, body, .. } => {
                expr_contains_subgroup_shuffle(from)
                    || expr_contains_subgroup_shuffle(to)
                    || nodes_contain_subgroup_shuffle(body)
            }
            Node::AsyncLoad { offset, size, .. } | Node::AsyncStore { offset, size, .. } => {
                expr_contains_subgroup_shuffle(offset) || expr_contains_subgroup_shuffle(size)
            }
            Node::Trap { address, .. } => expr_contains_subgroup_shuffle(address),
            Node::Block(body) => nodes_contain_subgroup_shuffle(body),
            Node::Region { body, .. } => nodes_contain_subgroup_shuffle(body),
            Node::IndirectDispatch { .. }
            | Node::AsyncWait { .. }
            | Node::AllReduce { .. }
            | Node::AllGather { .. }
            | Node::ReduceScatter { .. }
            | Node::Broadcast { .. }
            | Node::Return
            | Node::Barrier { .. }
            | Node::Resume { .. }
            | Node::Opaque(_) => false,
            _ => false,
        })
    }

    fn collect_loop_vars(nodes: &[Node], vars: &mut Vec<String>) {
        for node in nodes {
            match node {
                Node::If {
                    then, otherwise, ..
                } => {
                    collect_loop_vars(then, vars);
                    collect_loop_vars(otherwise, vars);
                }
                Node::Loop { var, body, .. } => {
                    vars.push(var.to_string());
                    collect_loop_vars(body, vars);
                }
                Node::Block(body) => collect_loop_vars(body, vars),
                Node::Region { body, .. } => collect_loop_vars(body, vars),
                _ => {}
            }
        }
    }

    fn affine_cpu_reference(
        x: &[f32],
        packed: &[u32],
        scale: &[f32],
        zero_point: &[u32],
        bias: &[f32],
        in_dim: u32,
        out_dim: u32,
        group_size: u32,
    ) -> Vec<f32> {
        (0..out_dim as usize)
            .map(|out| {
                let mut acc = bias[out];
                for k in 0..in_dim as usize {
                    let word = packed[(k / 8) * out_dim as usize + out];
                    let nibble = ((word >> ((k % 8) * 4)) & 0xF) as f32;
                    let sidecar_idx = (k / group_size as usize) * out_dim as usize + out;
                    acc += x[k] * (nibble - zero_point[sidecar_idx] as f32) * scale[sidecar_idx];
                }
                acc
            })
            .collect()
    }

    #[test]
    fn linear_4bit_matches_unpack_then_linear() {
        // in_dim = 8, out_dim = 2
        // x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let x = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // Weights: 2 output columns, each with 8 nibbles (2 u32s)
        // Column 0 nibbles: [1, 2, 3, 4, 5, 6, 7, 8] → packed as:
        //   u32[0] = 0x_8_7_6_5_4_3_2_1 (little-endian byte order, but nibble order within u32)
        //   Actually in our unpack: nibble for k=0 is bits[3:0], k=1 is bits[7:4], etc.
        //   So u32[0] = (8<<28)|(7<<24)|(6<<20)|(5<<16)|(4<<12)|(3<<8)|(2<<4)|1
        let col0 = 0x8765_4321u32;
        // Column 1 nibbles: [0, 0, 0, 0, 0, 0, 0, 0]
        let col1 = 0x0000_0000u32;
        let w = u32_bytes(&[col0, col1]);
        // bias = [0.0, 0.0]
        let b = f32_bytes(&[0.0, 0.0]);
        let out_size = 2usize * 4;

        let program = linear_4bit("x", "w", "b", "out", 8, 2).unwrap();
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(x),
                Value::from(w),
                Value::from(b),
                Value::from(vec![0u8; out_size]),
            ],
        )
        .expect("Fix: reference eval must succeed");

        let out_vals: Vec<f32> =
            vyre_primitives::wire::decode_f32_le_bytes_all(&outputs[0].to_bytes());

        // Column 0: sum_k x[k] * nibble[k] = 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 + 8*8 = 204
        assert!(
            (out_vals[0] - 204.0).abs() < 1e-4,
            "expected 204.0, got {}",
            out_vals[0]
        );
        // Column 1: all zero nibbles → 0
        assert!(
            (out_vals[1] - 0.0).abs() < 1e-4,
            "expected 0.0, got {}",
            out_vals[1]
        );
    }

    #[test]
    fn linear_4bit_rejects_indivisible_in_dim() {
        let err = linear_4bit("x", "w", "b", "out", 7, 4).unwrap_err();
        assert!(
            err.contains("divisible by 8"),
            "error must mention divisibility: {err}"
        );
    }

    #[test]
    fn linear_4bit_affine_grouped_applies_scale_and_zero_point_in_loop() {
        let x = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let w = u32_bytes(&[0x8765_4321u32, 0x0000_0000u32]);
        let scale = f32_bytes(&[0.5, 1.0, 2.0, 1.0]);
        let zero_point = u32_bytes(&[1, 0, 4, 0]);
        let b = f32_bytes(&[0.0, 3.0]);

        let program = linear_4bit_affine_grouped("x", "w", "scale", "zp", "b", "out", 8, 2, 4)
            .expect("Fix: affine grouped int4 linear fixture must build");
        assert_eq!(
            program.workgroup_size(),
            AFFINE_GROUPED_WORKGROUP_SIZE,
            "Fix: grouped INT4 linear must keep the CUDA-measured cooperative release launch shape."
        );
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(x),
                Value::from(w),
                Value::from(scale),
                Value::from(zero_point),
                Value::from(b),
                Value::from(vec![0u8; 8]),
            ],
        )
        .expect("Fix: affine grouped int4 linear must execute");

        let out_vals = vyre_primitives::wire::decode_f32_le_bytes_all(&outputs[0].to_bytes());

        assert!(
            (out_vals[0] - 150.0).abs() < 1e-4,
            "expected fused affine dequantized dot product 150.0, got {}",
            out_vals[0]
        );
        let evidence = linear_4bit_affine_grouped_planner_evidence(
            &QuantizedLinear4BitSpec::affine_grouped(8, 2, 4),
        )
        .expect("Fix: planner evidence fixture must build");
        assert!(
            (out_vals[0] - 150.0).abs() <= evidence.output_drift_abs_tolerance,
            "Fix: runtime output drift must stay within planner evidence tolerance."
        );
        assert!(
            (out_vals[1] - 3.0).abs() < 1e-4,
            "expected bias-only second output 3.0, got {}",
            out_vals[1]
        );
    }

    #[test]
    fn linear_4bit_affine_grouped_batched_indexes_independent_activation_rows() {
        let mut x = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        x.extend(f32_bytes(&[0.0; 8]));
        let w = u32_bytes(&[0x8765_4321u32, 0x0000_0000u32]);
        let scale = f32_bytes(&[0.5, 1.0, 2.0, 1.0]);
        let zero_point = u32_bytes(&[1, 0, 4, 0]);
        let b = f32_bytes(&[0.0, 3.0]);
        let program =
            linear_4bit_affine_grouped_batched("x", "w", "scale", "zp", "b", "out", 8, 2, 4, 2)
                .expect("Fix: valid batched affine grouped INT4 fixture must build");

        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(x),
                Value::from(w),
                Value::from(scale),
                Value::from(zero_point),
                Value::from(b),
                Value::from(vec![0u8; 16]),
            ],
        )
        .expect("Fix: batched affine grouped INT4 linear must execute");
        let out_vals = vyre_primitives::wire::decode_f32_le_bytes_all(&outputs[0].to_bytes());

        assert_eq!(out_vals.len(), 4);
        assert!((out_vals[0] - 150.0).abs() < 1e-4);
        assert!((out_vals[1] - 3.0).abs() < 1e-4);
        assert!((out_vals[2] - 0.0).abs() < 1e-4);
        assert!((out_vals[3] - 3.0).abs() < 1e-4);

        let error =
            linear_4bit_affine_grouped_batched("x", "w", "scale", "zp", "b", "out", 8, 2, 4, 0)
                .expect_err("Fix: zero-sized grouped INT4 batches must fail closed");
        assert!(error.contains("batch_size=0"));
    }

    /// WHY: the resident throughput path shares one dequantized weight tile across eight
    /// independent batch rows. Row/output remapping must not alias activations or results.
    #[test]
    fn linear_4bit_affine_grouped_batched_reuses_weights_across_independent_rows() {
        let mut activations = Vec::with_capacity(8 * 256);
        for batch in 0..8 {
            activations.extend(std::iter::repeat_n((batch + 1) as f32, 256));
        }
        let packed = vec![0x1111_1111_u32; 32 * 8];
        let scale = vec![1.0_f32; 4 * 8];
        let zero_point = vec![0_u32; 4 * 8];
        let bias = (0..8).map(|value| value as f32).collect::<Vec<_>>();
        let program =
            linear_4bit_affine_grouped_batched("x", "w", "scale", "zp", "b", "out", 256, 8, 64, 8)
                .expect("Fix: valid cross-batch grouped INT4 fixture must build");

        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&activations)),
                Value::from(u32_bytes(&packed)),
                Value::from(f32_bytes(&scale)),
                Value::from(u32_bytes(&zero_point)),
                Value::from(f32_bytes(&bias)),
                Value::from(vec![0_u8; 8 * 8 * core::mem::size_of::<f32>()]),
            ],
        )
        .expect("Fix: cross-batch grouped INT4 weight reuse must execute");
        let values = vyre_primitives::wire::decode_f32_le_bytes_all(&outputs[0].to_bytes());

        assert_eq!(values.len(), 8 * 8);
        for batch in 0..8 {
            for output in 0..8 {
                let expected = 256.0 * (batch + 1) as f32 + output as f32;
                assert!(
                    (values[batch * 8 + output] - expected).abs() < 1.0e-4,
                    "Fix: batch {batch} output {output} must remain independent: expected {expected}, got {}",
                    values[batch * 8 + output]
                );
            }
        }
    }

    #[test]
    fn batched_weight_reuse_does_not_shadow_caller_buffer_names() {
        let program = linear_4bit_affine_grouped_batched(
            AFFINE_GROUPED_WEIGHT_TILE,
            "w",
            "scale",
            "zp",
            "b",
            "out",
            256,
            8,
            64,
            8,
        )
        .expect("Fix: caller-owned buffer names must remain valid on the batched builder");

        assert_eq!(
            program
                .buffers()
                .iter()
                .filter(|buffer| buffer.name() == AFFINE_GROUPED_WEIGHT_TILE)
                .count(),
            1
        );
        assert!(
            program
                .buffers()
                .iter()
                .all(|buffer| buffer.access() != BufferAccess::Workgroup),
            "Fix: an internal weight tile must not shadow the caller-owned input buffer"
        );
    }

    #[test]
    fn linear_4bit_affine_grouped_broadcasts_packed_weight_words() {
        let program =
            linear_4bit_affine_grouped("x", "w", "scale", "zp", "b", "out", 256, 4096, 64)
                .expect("Fix: grouped INT4 affine release fixture must build");

        assert!(
            nodes_contain_subgroup_shuffle(program.entry()),
            "Fix: grouped INT4 release kernel must broadcast each packed u32 weight word across its 8 nibble lanes instead of reloading it per MAC."
        );
    }

    #[test]
    fn linear_4bit_affine_grouped_hoists_sidecars_for_aligned_release_groups() {
        let aligned =
            linear_4bit_affine_grouped("x", "w", "scale", "zp", "b", "out", 256, 4096, 64)
                .expect("Fix: aligned grouped INT4 release fixture must build");
        let mut aligned_loops = Vec::new();
        collect_loop_vars(aligned.entry(), &mut aligned_loops);
        assert!(
            aligned_loops.iter().any(|var| var == "group_idx")
                && aligned_loops.iter().any(|var| var == "group_chunk"),
            "Fix: release-aligned grouped INT4 must load and broadcast sidecars once per quantization group, then scan that group's chunks: {aligned_loops:?}"
        );

        let single_tile =
            linear_4bit_affine_grouped("x", "w", "scale", "zp", "b", "out", 32, 8, 32)
                .expect("Fix: single-tile grouped INT4 fixture must build");
        let mut single_tile_loops = Vec::new();
        collect_loop_vars(single_tile.entry(), &mut single_tile_loops);
        assert!(
            single_tile_loops.iter().any(|var| var == "chunk")
                && !single_tile_loops.iter().any(|var| var == "group_idx"),
            "Fix: single-tile and non-tile-aligned quantization groups must retain chunk-indexed sidecar selection for correctness: {single_tile_loops:?}"
        );
    }

    #[test]
    fn linear_4bit_affine_grouped_rejects_zero_group_size() {
        let err =
            linear_4bit_affine_grouped("x", "w", "scale", "zp", "b", "out", 8, 4, 0).unwrap_err();
        assert!(
            err.contains("group_size=0"),
            "error must identify invalid group size: {err}"
        );
    }

    #[test]
    fn typed_affine_grouped_builder_uses_quantized_metadata() {
        let spec = QuantizedLinear4BitSpec::affine_grouped(32, 7, 8);
        let program = linear_4bit_affine_grouped_typed(&spec, "x", "w", "scale", "zp", "b", "out")
            .expect("Fix: valid typed grouped INT4 spec must build");

        assert_eq!(program.buffers()[1].name(), "w");
        assert_eq!(program.buffers()[1].element(), DataType::U32);
        assert_eq!(program.buffers()[1].count(), 28);
        assert!(matches!(
            spec.weight_type,
            DataType::Quantized {
                scale: QuantizationScale::PerGroup { group_size: 8 },
                zero_point: QuantizationZeroPoint::PerGroup { group_size: 8 },
                ..
            }
        ));
    }

    #[test]
    fn typed_affine_grouped_planner_evidence_records_matmul_and_dequant_contract() {
        let spec = QuantizedLinear4BitSpec::affine_grouped(256, 4096, 64);
        let evidence = linear_4bit_affine_grouped_planner_evidence(&spec)
            .expect("Fix: release grouped INT4 evidence must build");

        assert_eq!(evidence.in_dim, 256);
        assert_eq!(evidence.out_dim, 4096);
        assert_eq!(evidence.group_size, 64);
        assert_eq!(evidence.group_count, 4);
        assert_eq!(evidence.packed_weight_bytes, 524_288);
        assert_eq!(evidence.dequantized_weight_bytes, 4_194_304);
        assert_eq!(
            evidence.dequant_bytes_elided,
            evidence.dequantized_weight_bytes
        );
        assert_eq!(evidence.sidecar_bytes, 131_072);
        assert_eq!(evidence.bias_bytes, 16_384);
        assert_eq!(evidence.output_bytes, 16_384);
        assert_eq!(evidence.matmul_m, 4096);
        assert_eq!(evidence.matmul_k, 256);
        assert_eq!(evidence.matmul_n, 1);
        assert_eq!(evidence.matmul_tile, AFFINE_GROUPED_LANES_PER_OUTPUT);
        assert_eq!(evidence.matmul_selected_path, "cooperative");
        assert_eq!(evidence.matmul_candidate_path, None);
        assert_eq!(
            evidence.matmul_fallback_reason,
            Some("strict_f32_requested")
        );
        assert!(!evidence.tensor_core_eligible);
        assert_eq!(
            evidence.output_drift_abs_tolerance,
            LINEAR_4BIT_AFFINE_GROUPED_OUTPUT_DRIFT_ABS_TOLERANCE
        );
    }

    #[test]
    fn typed_affine_grouped_builder_rejects_mismatched_quantized_metadata() {
        let bad_storage = QuantizedLinear4BitSpec {
            in_dim: 32,
            out_dim: 4,
            weight_type: DataType::Quantized {
                storage: Box::new(DataType::I8),
                scale: QuantizationScale::PerGroup { group_size: 8 },
                zero_point: QuantizationZeroPoint::PerGroup { group_size: 8 },
            },
        };
        let error =
            linear_4bit_affine_grouped_typed(&bad_storage, "x", "w", "scale", "zp", "b", "out")
                .unwrap_err();
        assert!(
            error.contains("storage I4"),
            "Fix: storage mismatch should be explicit: {error}"
        );

        let bad_sidecar = QuantizedLinear4BitSpec {
            in_dim: 32,
            out_dim: 4,
            weight_type: DataType::Quantized {
                storage: Box::new(DataType::I4),
                scale: QuantizationScale::PerGroup { group_size: 8 },
                zero_point: QuantizationZeroPoint::PerGroup { group_size: 16 },
            },
        };
        let error =
            linear_4bit_affine_grouped_typed(&bad_sidecar, "x", "w", "scale", "zp", "b", "out")
                .unwrap_err();
        assert!(
            error.contains("group sizes to match"),
            "Fix: sidecar mismatch should be explicit: {error}"
        );
    }

    #[test]
    fn generated_typed_affine_grouped_specs_build_or_reject_by_metadata_contract() {
        let mut accepted = 0usize;
        let mut rejected = 0usize;
        for in_dim in [8u32, 10, 16, 18, 24, 32, 64, 128] {
            for out_dim in [1u32, 2, 3, 7, 16, 31] {
                for group_size in [1u32, 2, 4, 8, 16, 32] {
                    let spec = QuantizedLinear4BitSpec::affine_grouped(in_dim, out_dim, group_size);
                    let result = linear_4bit_affine_grouped_typed(
                        &spec, "x", "w", "scale", "zp", "b", "out",
                    );
                    if in_dim % 8 == 0 {
                        let program = result.expect("Fix: generated valid typed spec must build");
                        let output = &program.buffers()[5];
                        assert!(
                            output.count() >= out_dim,
                            "Fix: grouped INT4 output storage must cover the logical outputs after launch padding."
                        );
                        assert_eq!(
                            output.output_byte_range(),
                            Some(0..(out_dim as usize * core::mem::size_of::<f32>())),
                            "Fix: grouped INT4 output byte range must trim padded launch storage to the logical tensor."
                        );
                        accepted += 1;
                    } else {
                        let error = result.expect_err(
                            "Fix: generated indivisible typed spec must reject before dispatch",
                        );
                        assert!(error.contains("divisible by 8"));
                        rejected += 1;
                    }
                }
            }
        }

        assert!(
            accepted + rejected >= 216,
            "Fix: generated typed quantized specs should cover hundreds of layouts"
        );
    }

    #[test]
    fn generated_affine_grouped_vectors_match_cpu_oracle() {
        let mut checked = 0usize;
        for out_dim in [1u32, 2, 3, 5, 8, 13, 21, 32] {
            for group_size in [1u32, 2, 4, 8, 16, 32] {
                for seed in 0..48u32 {
                    let in_dim = 32u32;
                    let group_count = in_dim.div_ceil(group_size);
                    let x = (0..in_dim)
                        .map(|k| ((k.wrapping_mul(3).wrapping_add(seed)) % 19) as f32)
                        .collect::<Vec<_>>();
                    let mut packed = vec![0u32; (in_dim / 8 * out_dim) as usize];
                    for block in 0..(in_dim / 8) {
                        for out in 0..out_dim {
                            let mut word = 0u32;
                            for lane in 0..8 {
                                let k = block * 8 + lane;
                                let nibble = k
                                    .wrapping_mul(7)
                                    .wrapping_add(out.wrapping_mul(11))
                                    .wrapping_add(seed)
                                    & 0xF;
                                word |= nibble << (lane * 4);
                            }
                            packed[(block * out_dim + out) as usize] = word;
                        }
                    }
                    let mut scale = vec![0.0f32; (group_count * out_dim) as usize];
                    let mut zero_point = vec![0u32; (group_count * out_dim) as usize];
                    for group in 0..group_count {
                        for out in 0..out_dim {
                            let idx = (group * out_dim + out) as usize;
                            scale[idx] = match (group + out + seed) & 3 {
                                0 => 0.25,
                                1 => 0.5,
                                2 => 1.0,
                                _ => 2.0,
                            };
                            zero_point[idx] =
                                group.wrapping_mul(5).wrapping_add(out).wrapping_add(seed) & 0xF;
                        }
                    }
                    let bias = (0..out_dim)
                        .map(|out| ((out + seed) & 7) as f32)
                        .collect::<Vec<_>>();

                    let program = linear_4bit_affine_grouped(
                        "x", "w", "scale", "zp", "b", "out", in_dim, out_dim, group_size,
                    )
                    .expect("Fix: generated affine grouped fixture must build");
                    let outputs = vyre_reference::reference_eval(
                        &program,
                        &[
                            Value::from(f32_bytes(&x)),
                            Value::from(u32_bytes(&packed)),
                            Value::from(f32_bytes(&scale)),
                            Value::from(u32_bytes(&zero_point)),
                            Value::from(f32_bytes(&bias)),
                            Value::from(vec![0u8; out_dim as usize * 4]),
                        ],
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "Fix: generated affine grouped fixture must execute for out_dim={out_dim}, group_size={group_size}, seed={seed}: {error}"
                        )
                    });
                    let actual =
                        vyre_primitives::wire::decode_f32_le_bytes_all(&outputs[0].to_bytes());
                    let expected = affine_cpu_reference(
                        &x,
                        &packed,
                        &scale,
                        &zero_point,
                        &bias,
                        in_dim,
                        out_dim,
                        group_size,
                    );

                    assert_eq!(
                        actual, expected,
                        "generated affine grouped vector mismatch for out_dim={out_dim}, group_size={group_size}, seed={seed}"
                    );
                    checked += out_dim as usize;
                }
            }
        }

        assert!(
            checked >= 24_000,
            "Fix: generated affine grouped coverage should exercise tens of thousands of output vectors, got {checked}"
        );
    }
}

inventory::submit! {
    crate::fixture_catalog::OpEntry {
        semantic_version: 1,
        signature: None,
        tier: vyre_foundation::operation::OperationTier::Library,
        laws: &[],
        tolerance: vyre_foundation::operation::TolerancePolicy::EXACT,
        id: "vyre-libs::nn::linear_4bit",
        build: Some(|| {
            linear_4bit("x", "w", "b", "out", 8, 4).unwrap_or_else(|error| {
                crate::builder::invalid_builder_trap_program(
                    "vyre-libs::nn::linear_4bit",
                    "out",
                    DataType::F32,
                    error,
                )
            })
        }),
        test_inputs: Some(|| {
            let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
            let w: Vec<u32> = vec![0x7654_3210, 0xFEDC_BA98, 0x1111_1111, 0x0000_0000];
            let b: Vec<f32> = vec![0.0; 4];
            vec![vec![
                vyre_primitives::wire::pack_f32_slice(&x),
                vyre_primitives::wire::pack_u32_slice(&w),
                vyre_primitives::wire::pack_f32_slice(&b),
            ]]
        }),
        expected_output: Some(|| {
            let out = [140.0f32, 364.0, 28.0, 0.0];
            vec![vec![vyre_primitives::wire::pack_f32_slice(&out)]]
        }),
        category: Some("nn"),
    }
}

inventory::submit! {
    crate::fixture_catalog::OpEntry {
        semantic_version: 1,
        signature: None,
        tier: vyre_foundation::operation::OperationTier::Library,
        laws: &[],
        tolerance: vyre_foundation::operation::TolerancePolicy::EXACT,
        id: "vyre-libs::nn::linear_4bit_affine_grouped",
        build: Some(|| {
            linear_4bit_affine_grouped("x", "w", "scale", "zp", "b", "out", 8, 2, 4)
                .unwrap_or_else(|error| {
                    crate::builder::invalid_builder_trap_program(
                        "vyre-libs::nn::linear_4bit_affine_grouped",
                        "out",
                        DataType::F32,
                        error,
                    )
                })
        }),
        test_inputs: Some(|| {
            let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let w = [0x8765_4321u32, 0x0000_0000u32];
            let scale = [0.5f32, 1.0, 2.0, 1.0];
            let zp = [1u32, 0, 4, 0];
            let b = [0.0f32, 3.0];
            vec![vec![
                vyre_primitives::wire::pack_f32_slice(&x),
                vyre_primitives::wire::pack_u32_slice(&w),
                vyre_primitives::wire::pack_f32_slice(&scale),
                vyre_primitives::wire::pack_u32_slice(&zp),
                vyre_primitives::wire::pack_f32_slice(&b),
            ]]
        }),
        expected_output: Some(|| {
            let out = [150.0f32, 3.0];
            vec![vec![vyre_primitives::wire::pack_f32_slice(&out)]]
        }),
        category: Some("nn"),
    }
}
