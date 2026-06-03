//! Fused `linear_4bit` constructor  -  unpack-on-demand 4-bit quantized linear.
//!
//! Instead of materializing an unpacked f32 weight buffer, this kernel loads
//! the packed u32 weight, extracts the correct nibble inside the inner `k`
//! loop, and accumulates directly. This eliminates the 8× memory expansion
//! of a separate unpack dispatch.

use crate::region::wrap_anonymous;
use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_spec::{QuantizationScale, QuantizationZeroPoint};

const INT4_LINEAR_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
const AFFINE_GROUPED_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
const AFFINE_GROUPED_LANES_PER_OUTPUT: u32 = 32;
const AFFINE_GROUPED_OUTPUTS_PER_WARP: u32 = 1;
const AFFINE_GROUPED_WARPS_PER_WORKGROUP: u32 =
    AFFINE_GROUPED_WORKGROUP_SIZE[0] / AFFINE_GROUPED_LANES_PER_OUTPUT;
const AFFINE_GROUPED_OUTPUTS_PER_WORKGROUP: u32 =
    AFFINE_GROUPED_WARPS_PER_WORKGROUP * AFFINE_GROUPED_OUTPUTS_PER_WARP;
const AFFINE_GROUPED_OP_ID: &str = "vyre-libs::nn::linear_4bit_affine_grouped";

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
/// For bounded group counts the emitted IR hoists scale/zero-point loads once
/// per `(group, output)` and emits one tight `k` loop per group. This removes
/// per-MAC group division and repeated sidecar loads on the inference path.
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

    let tile = AFFINE_GROUPED_LANES_PER_OUTPUT;
    let chunks = in_dim.div_ceil(tile);
    let out_idx = Expr::var("out_idx");
    let local = Expr::var("local");
    let lane = Expr::var("lane");
    let k = Expr::var("k");
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
    let weight_f32 = Expr::mul(
        Expr::sub(
            Expr::cast(DataType::F32, nibble),
            Expr::cast(DataType::F32, Expr::var("group_zero_point")),
        ),
        Expr::var("group_scale"),
    );

    let mut per_output = vec![Node::let_bind("local_acc", Expr::f32(0.0))];
    if group_size > tile && group_size % tile == 0 {
        let group_chunks = group_size.div_ceil(tile);
        per_output.push(Node::loop_for(
            "group_idx",
            Expr::u32(0),
            Expr::u32(group_count),
            vec![
                Node::let_bind(
                    "group_base",
                    Expr::mul(Expr::var("group_idx"), Expr::u32(group_size)),
                ),
                Node::let_bind(
                    "sidecar_idx",
                    Expr::add(
                        Expr::mul(Expr::var("group_idx"), Expr::u32(out_dim)),
                        out_idx.clone(),
                    ),
                ),
                Node::let_bind("scale_lane", Expr::f32(0.0)),
                Node::let_bind("zero_point_lane", Expr::u32(0)),
                Node::if_then(
                    Expr::eq(lane.clone(), Expr::u32(0)),
                    vec![
                        Node::assign("scale_lane", Expr::load(scale, Expr::var("sidecar_idx"))),
                        Node::assign(
                            "zero_point_lane",
                            Expr::load(zero_point, Expr::var("sidecar_idx")),
                        ),
                    ],
                ),
                Node::let_bind(
                    "group_scale",
                    Expr::subgroup_shuffle(Expr::var("scale_lane"), Expr::u32(0)),
                ),
                Node::let_bind(
                    "group_zero_point",
                    Expr::subgroup_shuffle(Expr::var("zero_point_lane"), Expr::u32(0)),
                ),
                Node::loop_for(
                    "group_chunk",
                    Expr::u32(0),
                    Expr::u32(group_chunks),
                    vec![
                        Node::let_bind(
                            "k",
                            Expr::add(
                                Expr::var("group_base"),
                                Expr::add(
                                    Expr::mul(Expr::var("group_chunk"), Expr::u32(tile)),
                                    lane.clone(),
                                ),
                            ),
                        ),
                        Node::let_bind("lane_in_word", Expr::bitand(lane.clone(), Expr::u32(7))),
                        Node::let_bind(
                            "word_leader_lane",
                            Expr::bitand(lane.clone(), Expr::u32(0xffff_fff8)),
                        ),
                        Node::let_bind(
                            "word_leader_k",
                            Expr::add(
                                Expr::var("group_base"),
                                Expr::add(
                                    Expr::mul(Expr::var("group_chunk"), Expr::u32(tile)),
                                    word_leader_lane.clone(),
                                ),
                            ),
                        ),
                        Node::let_bind("packed_word_lane", Expr::u32(0)),
                        Node::if_then(
                            Expr::and(
                                Expr::eq(lane_in_word.clone(), Expr::u32(0)),
                                Expr::lt(word_leader_k.clone(), Expr::u32(in_dim)),
                            ),
                            vec![Node::assign(
                                "packed_word_lane",
                                Expr::load(w_packed, packed_idx.clone()),
                            )],
                        ),
                        Node::let_bind(
                            "packed_word",
                            Expr::subgroup_shuffle(Expr::var("packed_word_lane"), word_leader_lane),
                        ),
                        Node::if_then(
                            Expr::lt(k.clone(), Expr::u32(in_dim)),
                            vec![Node::assign(
                                "local_acc",
                                Expr::add(
                                    Expr::var("local_acc"),
                                    Expr::mul(Expr::load(x, k.clone()), weight_f32.clone()),
                                ),
                            )],
                        ),
                    ],
                ),
            ],
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
                Node::let_bind("packed_word_lane", Expr::u32(0)),
                Node::if_then(
                    Expr::and(
                        Expr::eq(lane_in_word.clone(), Expr::u32(0)),
                        Expr::lt(word_leader_k.clone(), Expr::u32(in_dim)),
                    ),
                    vec![Node::assign(
                        "packed_word_lane",
                        Expr::load(w_packed, packed_idx),
                    )],
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
                Node::if_then(
                    Expr::lt(k.clone(), Expr::u32(in_dim)),
                    vec![Node::assign(
                        "local_acc",
                        Expr::add(
                            Expr::var("local_acc"),
                            Expr::mul(Expr::load(x, k.clone()), weight_f32),
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
            index: out_idx.clone(),
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
        Node::loop_for(
            "warp_output",
            Expr::u32(0),
            Expr::u32(AFFINE_GROUPED_OUTPUTS_PER_WARP),
            vec![
                Node::let_bind(
                    "out_idx",
                    Expr::add(
                        Expr::add(
                            Expr::mul(
                                Expr::WorkgroupId { axis: 0 },
                                Expr::u32(AFFINE_GROUPED_OUTPUTS_PER_WORKGROUP),
                            ),
                            Expr::mul(
                                Expr::var("warp_output"),
                                Expr::u32(AFFINE_GROUPED_WARPS_PER_WORKGROUP),
                            ),
                        ),
                        Expr::var("warp"),
                    ),
                ),
                Node::if_then(Expr::lt(out_idx.clone(), Expr::u32(out_dim)), per_output),
            ],
        ),
    ];
    let output_workgroups = out_dim.div_ceil(AFFINE_GROUPED_OUTPUTS_PER_WORKGROUP);
    let padded_output_count = output_workgroups
        .checked_mul(AFFINE_GROUPED_WORKGROUP_SIZE[0])
        .ok_or_else(|| {
            "Fix: linear_4bit_affine_grouped output workgroups overflow u32; reduce dimensions."
                .to_string()
        })?;
    let output_byte_len = (out_dim as usize)
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| {
            "Fix: linear_4bit_affine_grouped output byte length overflows usize; reduce dimensions."
                .to_string()
        })?;

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(x, 0, BufferAccess::ReadOnly, DataType::F32).with_count(in_dim),
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
            | Expr::SubgroupAdd { value: index }
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
        assert!(
            (out_vals[1] - 3.0).abs() < 1e-4,
            "expected bias-only second output 3.0, got {}",
            out_vals[1]
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
        assert!(
            !aligned_loops.iter().any(|var| var == "chunk"),
            "Fix: release-aligned grouped INT4 must not use the per-chunk sidecar broadcast path: {aligned_loops:?}"
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
    crate::harness::OpEntry {
        id: "vyre-libs::nn::linear_4bit",
        build: || {
            linear_4bit("x", "w", "b", "out", 8, 4).unwrap_or_else(|error| {
                crate::builder::invalid_output_program(
                    "vyre-libs::nn::linear_4bit",
                    "out",
                    DataType::F32,
                    error,
                )
            })
        },
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
    crate::harness::OpEntry {
        id: "vyre-libs::nn::linear_4bit_affine_grouped",
        build: || {
            linear_4bit_affine_grouped("x", "w", "scale", "zp", "b", "out", 8, 2, 4)
                .unwrap_or_else(|error| {
                    crate::builder::invalid_output_program(
                        "vyre-libs::nn::linear_4bit_affine_grouped",
                        "out",
                        DataType::F32,
                        error,
                    )
                })
        },
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
