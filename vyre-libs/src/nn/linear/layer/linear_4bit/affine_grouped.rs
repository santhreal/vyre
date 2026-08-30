//! Fused affine grouped INT4 linear: public entry points, strategy selection,
//! and the lane-predicated lowering used when weight-tile reuse does not apply.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{Expr, Node, Program};

use super::affine_grouped_weight_reuse::linear_4bit_affine_grouped_weight_reuse;
use super::grouped_layout::{
    affine_grouped_buffers, affine_grouped_output_extent, bounded_index, broadcast_from_lane0,
    dequantized_weight, lane_decomposition, packed_column_index, push_group_affine_terms,
    push_lane0_sidecar_loads, push_packed_word_fetch, push_subgroup_reduction_store,
    AFFINE_GROUPED_LANES_PER_OUTPUT, AFFINE_GROUPED_OP_ID, AFFINE_GROUPED_OUTPUTS_PER_WORKGROUP,
    AFFINE_GROUPED_SUBGROUPS_PER_WORKGROUP, AFFINE_GROUPED_WEIGHT_TILE,
    AFFINE_GROUPED_WORKGROUP_SIZE,
};

/// Build a fused affine INT4 linear Program:
///
/// `out[i] = b[i] + sum_k x[k] * ((unpack4(w_packed[k,i]) - zero_point[group,i]) * scale[group,i])`
///
/// This keeps weights packed, applies per-group quantization metadata inside
/// the dot-product loop, and avoids a separate dequantize materialization
/// dispatch. `w_packed` stores 8 4-bit weights per u32 using the same
/// column-interleaved layout as [`super::linear_4bit`]. `scale` is f32, `zero_point`
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
    if batch_size >= AFFINE_GROUPED_SUBGROUPS_PER_WORKGROUP
        && batch_size % AFFINE_GROUPED_SUBGROUPS_PER_WORKGROUP == 0
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
    let packed_idx = packed_column_index(out_dim, out_idx.clone());
    let group = Expr::div(k.clone(), Expr::u32(group_size));
    let chunk_sidecar_idx = Expr::add(Expr::mul(group, Expr::u32(out_dim)), out_idx.clone());
    let weight_f32 = dequantized_weight();

    let mut per_output = vec![Node::let_bind("local_acc", Expr::f32(0.0))];
    if group_size > tile && group_size % tile == 0 {
        let group_chunks = group_size / tile;
        let build_chunk = |k_expr: Expr, word_leader_k_expr: Expr| {
            let mut chunk = vec![Node::let_bind("k", k_expr)];
            push_packed_word_fetch(
                &mut chunk,
                &lane,
                w_packed,
                word_leader_k_expr,
                packed_idx.clone(),
                Some(in_dim),
                total_u32s,
            );
            chunk.push(Node::if_then(
                Expr::lt(k.clone(), Expr::u32(in_dim)),
                vec![Node::assign(
                    "local_acc",
                    Expr::fma(
                        Expr::load(x, activation_idx.clone()),
                        weight_f32.clone(),
                        Expr::var("local_acc"),
                    ),
                )],
            ));
            Node::Block(chunk)
        };
        let build_group = |group_base_expr: Expr, sidecar_idx_expr: Expr, chunk_scan: Node| {
            let mut group = vec![
                Node::let_bind("group_base", group_base_expr),
                Node::let_bind("sidecar_idx", sidecar_idx_expr),
            ];
            push_lane0_sidecar_loads(&mut group, &lane, scale, zero_point, sidecar_count);
            push_group_affine_terms(
                &mut group,
                broadcast_from_lane0("scale_lane"),
                broadcast_from_lane0("zero_point_lane"),
            );
            group.push(chunk_scan);
            Node::Block(group)
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
                    Expr::var("word_leader_lane"),
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
        let mut chunk = vec![Node::let_bind(
            "k",
            Expr::add(Expr::mul(Expr::var("chunk"), Expr::u32(tile)), lane.clone()),
        )];
        push_packed_word_fetch(
            &mut chunk,
            &lane,
            w_packed,
            Expr::add(
                Expr::mul(Expr::var("chunk"), Expr::u32(tile)),
                Expr::var("word_leader_lane"),
            ),
            packed_idx,
            Some(in_dim),
            total_u32s,
        );
        chunk.push(Node::let_bind("sidecar_idx", chunk_sidecar_idx));
        push_group_affine_terms(
            &mut chunk,
            Expr::load(
                scale,
                bounded_index(Expr::var("sidecar_idx"), sidecar_count),
            ),
            Expr::load(
                zero_point,
                bounded_index(Expr::var("sidecar_idx"), sidecar_count),
            ),
        );
        chunk.push(Node::if_then(
            Expr::lt(k.clone(), Expr::u32(in_dim)),
            vec![Node::assign(
                "local_acc",
                Expr::fma(
                    Expr::load(x, activation_idx.clone()),
                    weight_f32,
                    Expr::var("local_acc"),
                ),
            )],
        ));
        per_output.push(Node::loop_for(
            "chunk",
            Expr::u32(0),
            Expr::u32(chunks),
            chunk,
        ));
    }
    push_subgroup_reduction_store(&mut per_output, lane.clone(), b, out, out_idx.clone());

    let mut body = lane_decomposition();
    body.extend([
        Node::let_bind(
            "linear_out_idx",
            Expr::add(
                Expr::mul(
                    Expr::LogicalTileId { axis: 0 },
                    Expr::u32(AFFINE_GROUPED_OUTPUTS_PER_WORKGROUP),
                ),
                Expr::var("subgroup"),
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
    ]);
    let output_workgroups = logical_output_count.div_ceil(AFFINE_GROUPED_OUTPUTS_PER_WORKGROUP);
    let (padded_output_count, output_byte_len) =
        affine_grouped_output_extent(output_workgroups, logical_output_count)?;

    Ok(Program::wrapped(
        affine_grouped_buffers(
            [x, w_packed, scale, zero_point, b, out],
            input_count,
            total_u32s,
            sidecar_count,
            out_dim,
            padded_output_count,
            output_byte_len,
        ),
        AFFINE_GROUPED_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(AFFINE_GROUPED_OP_ID, body)],
    ))
}

#[cfg(test)]
mod tests {
    use crate::fixture_bytes::eval_bytes;
    use vyre_foundation::ir::{Expr, Node};

    use super::super::grouped_layout::AFFINE_GROUPED_WORKGROUP_SIZE;
    use super::super::planner_evidence::linear_4bit_affine_grouped_planner_evidence;
    use super::super::QuantizedLinear4BitSpec;
    use super::{linear_4bit_affine_grouped, linear_4bit_affine_grouped_batched};
    use crate::fixture_bytes::{f32_bytes, u32_bytes};

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
            | Node::LogicalBarrier { .. }
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

    fn reference_affine_grouped(
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
        let outputs = eval_bytes(
            "affine_grouped",
            &program,
            vec![x, w, scale, zero_point, b, vec![0u8; 8]],
        );

        let out_vals = vyre_primitives::wire::decode_f32_le_bytes_all(&outputs[0]);

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

        let outputs = eval_bytes(
            "affine_grouped",
            &program,
            vec![x, w, scale, zero_point, b, vec![0u8; 16]],
        );
        let out_vals = vyre_primitives::wire::decode_f32_le_bytes_all(&outputs[0]);

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
    fn generated_affine_grouped_vectors_match_reference_oracle() {
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
                    let outputs = eval_bytes(
                        "affine_grouped",
                        &program,
                        vec![
                            f32_bytes(&x),
                            u32_bytes(&packed),
                            f32_bytes(&scale),
                            u32_bytes(&zero_point),
                            f32_bytes(&bias),
                            vec![0u8; out_dim as usize * 4],
                        ],
                    );
                    let actual = vyre_primitives::wire::decode_f32_le_bytes_all(&outputs[0]);
                    let expected = reference_affine_grouped(
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
