//! Workgroup geometry, buffer identity, and the IR stages shared by every
//! grouped INT4 lowering strategy.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node};

pub(super) const AFFINE_GROUPED_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
pub(super) const AFFINE_GROUPED_LANES_PER_OUTPUT: u32 = 32;
pub(super) const AFFINE_GROUPED_WARPS_PER_WORKGROUP: u32 =
    AFFINE_GROUPED_WORKGROUP_SIZE[0] / AFFINE_GROUPED_LANES_PER_OUTPUT;
pub(super) const AFFINE_GROUPED_OUTPUTS_PER_WORKGROUP: u32 = AFFINE_GROUPED_WARPS_PER_WORKGROUP;
pub(super) const AFFINE_GROUPED_OP_ID: &str = "vyre-libs::nn::linear_4bit_affine_grouped";
pub(super) const AFFINE_GROUPED_WEIGHT_TILE: &str = "linear_4bit_weight_tile";

/// Padded output element count and logical output byte length for a dispatch of
/// `output_workgroups` workgroups producing `logical_output_count` values.
///
/// # Errors
/// Returns `Err` when either derived extent overflows.
pub(super) fn affine_grouped_output_extent(
    output_workgroups: u32,
    logical_output_count: u32,
) -> Result<(u32, usize), String> {
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
    Ok((padded_output_count, output_byte_len))
}

/// The buffer set every affine-grouped lowering binds, in binding order:
/// activations, packed weights, scale, zero point, bias, output.
pub(super) fn affine_grouped_buffers(
    names: [&str; 6],
    input_count: u32,
    total_u32s: u32,
    sidecar_count: u32,
    out_dim: u32,
    padded_output_count: u32,
    output_byte_len: usize,
) -> Vec<BufferDecl> {
    let [x, w_packed, scale, zero_point, b, out] = names;
    vec![
        BufferDecl::storage(x, 0, BufferAccess::ReadOnly, DataType::F32).with_count(input_count),
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
    ]
}

/// `index` folded inside a buffer of `count` elements.
///
/// Every packed-word and sidecar read below sits in the taken arm of a select
/// whose condition already decides whether the value matters: a non-leader lane,
/// a reduction step past the input, or a padded output the workgroup grid rounded
/// up to. A select evaluates both arms, so the read runs for those lanes too and
/// lands past the buffer on a backend that does not bounds-check. Folding the
/// index keeps it inside, where the same select discards what it read.
pub(super) fn bounded_index(index: Expr, count: u32) -> Expr {
    Expr::select(
        Expr::lt(index.clone(), Expr::u32(count)),
        index,
        Expr::u32(0),
    )
}

/// Lane-cooperative packed-word fetch: the leader of each 8-lane group loads the
/// `u32` holding its eight nibbles and shuffles it across the group.
///
/// `bound` gates the leader's load against the reduction length. Callers whose
/// `k` range is exactly the workgroup width pass `None`, which leaves the gate
/// off; `packed_count` still keeps the index inside the weight buffer.
pub(super) fn push_packed_word_fetch(
    nodes: &mut Vec<Node>,
    lane: &Expr,
    w_packed: &str,
    word_leader_k: Expr,
    packed_idx: Expr,
    bound: Option<u32>,
    packed_count: u32,
) {
    let leader = Expr::eq(Expr::var("lane_in_word"), Expr::u32(0));
    let load_when = match bound {
        Some(in_dim) => Expr::and(
            leader,
            Expr::lt(Expr::var("word_leader_k"), Expr::u32(in_dim)),
        ),
        None => leader,
    };
    nodes.extend([
        Node::let_bind("lane_in_word", Expr::bitand(lane.clone(), Expr::u32(7))),
        Node::let_bind(
            "word_leader_lane",
            Expr::bitand(lane.clone(), Expr::u32(0xffff_fff8)),
        ),
        Node::let_bind("word_leader_k", word_leader_k),
        Node::let_bind(
            "packed_word_lane",
            Expr::select(
                load_when,
                Expr::load(w_packed, bounded_index(packed_idx, packed_count)),
                Expr::u32(0),
            ),
        ),
        Node::let_bind(
            "packed_word",
            Expr::subgroup_shuffle(Expr::var("packed_word_lane"), Expr::var("word_leader_lane")),
        ),
    ]);
}

/// Lane-0 sidecar loads, held per lane so the following broadcast has a value to
/// shuffle from.
pub(super) fn push_lane0_sidecar_loads(
    nodes: &mut Vec<Node>,
    lane: &Expr,
    scale: &str,
    zero_point: &str,
    sidecar_count: u32,
) {
    nodes.extend([
        Node::let_bind(
            "scale_lane",
            Expr::select(
                Expr::eq(lane.clone(), Expr::u32(0)),
                Expr::load(
                    scale,
                    bounded_index(Expr::var("sidecar_idx"), sidecar_count),
                ),
                Expr::f32(0.0),
            ),
        ),
        Node::let_bind(
            "zero_point_lane",
            Expr::select(
                Expr::eq(lane.clone(), Expr::u32(0)),
                Expr::load(
                    zero_point,
                    bounded_index(Expr::var("sidecar_idx"), sidecar_count),
                ),
                Expr::u32(0),
            ),
        ),
    ]);
}

/// The group's affine dequantization terms.
///
/// `scale_value` and `zero_point_value` bind the group's sidecar values: a
/// broadcast path passes a lane-0 shuffle of [`push_lane0_sidecar_loads`], a
/// warp-uniform path passes the load directly. Negation is folded into an FMA so
/// dequantization costs two FMAs per group and none per multiply-accumulate.
pub(super) fn push_group_affine_terms(
    nodes: &mut Vec<Node>,
    scale_value: Expr,
    zero_point_value: Expr,
) {
    nodes.extend([
        Node::let_bind("group_scale", scale_value),
        Node::let_bind("group_zero_point", zero_point_value),
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
    ]);
}

/// A lane-0 shuffle of a per-lane sidecar binding.
pub(super) fn broadcast_from_lane0(name: &str) -> Expr {
    Expr::subgroup_shuffle(Expr::var(name), Expr::u32(0))
}

/// The tile-point decomposition every grouped INT4 lowering opens with:
/// the within-tile logical point, the subgroup that owns one output, and its lane.
pub(super) fn lane_decomposition() -> Vec<Node> {
    vec![
        Node::let_bind("local", Expr::LogicalWithinTileId { axis: 0 }),
        Node::let_bind(
            "warp",
            Expr::div(
                Expr::var("local"),
                Expr::u32(AFFINE_GROUPED_LANES_PER_OUTPUT),
            ),
        ),
        Node::let_bind(
            "lane",
            Expr::rem(
                Expr::var("local"),
                Expr::u32(AFFINE_GROUPED_LANES_PER_OUTPUT),
            ),
        ),
    ]
}

/// Index of the packed `u32` word holding `word_leader_k`'s eight nibbles for
/// output column `out_idx`.
pub(super) fn packed_column_index(out_dim: u32, out_idx: Expr) -> Expr {
    Expr::add(
        Expr::mul(
            Expr::div(Expr::var("word_leader_k"), Expr::u32(8)),
            Expr::u32(out_dim),
        ),
        out_idx,
    )
}

/// This lane's 4-bit weight, selected out of the word
/// [`push_packed_word_fetch`] bound.
pub(super) fn packed_nibble() -> Expr {
    Expr::bitand(
        Expr::shr(
            Expr::var("packed_word"),
            Expr::mul(Expr::var("lane_in_word"), Expr::u32(4)),
        ),
        Expr::u32(0xF),
    )
}

/// This lane's dequantized weight, from the nibble and the affine terms
/// [`push_group_affine_terms`] bound.
pub(super) fn dequantized_weight() -> Expr {
    Expr::fma(
        Expr::cast(DataType::F32, packed_nibble()),
        Expr::var("group_scale"),
        Expr::var("group_zero_offset"),
    )
}

/// The warp reduction and the lane-0 biased store that ends every grouped INT4
/// lowering.
pub(super) fn push_warp_reduction_store(
    nodes: &mut Vec<Node>,
    lane: Expr,
    b: &str,
    out: &str,
    out_idx: Expr,
) {
    nodes.extend([
        Node::let_bind("warp_sum", Expr::subgroup_add(Expr::var("local_acc"))),
        Node::if_then(
            Expr::eq(lane, Expr::u32(0)),
            vec![Node::store(
                out,
                Expr::var("linear_out_idx"),
                Expr::add(Expr::load(b, out_idx), Expr::var("warp_sum")),
            )],
        ),
    ]);
}
