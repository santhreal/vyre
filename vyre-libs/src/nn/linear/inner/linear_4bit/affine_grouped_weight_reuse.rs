//! Grouped INT4 lowering that stages the packed weight column into a
//! workgroup tile and reuses it across the batch rows a workgroup owns.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::grouped_layout::{
    AFFINE_GROUPED_LANES_PER_OUTPUT, AFFINE_GROUPED_OP_ID, AFFINE_GROUPED_WARPS_PER_WORKGROUP,
    AFFINE_GROUPED_WEIGHT_TILE, AFFINE_GROUPED_WORKGROUP_SIZE,
};
use crate::region::wrap_anonymous;

#[allow(clippy::too_many_arguments)]
pub(super) fn linear_4bit_affine_grouped_weight_reuse(
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

#[cfg(test)]
mod tests {
    use vyre_foundation::ir::BufferAccess;
    use vyre_reference::value::Value;

    use super::super::affine_grouped::linear_4bit_affine_grouped_batched;
    use super::super::grouped_layout::AFFINE_GROUPED_WEIGHT_TILE;
    use crate::fixture_bytes::{f32_bytes, u32_bytes};

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
}
