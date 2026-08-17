//! CPU oracle for the synthetic release macro patterns: generated input words, per-lane
//! masks, and the match predicates the GPU programs must reproduce.

use super::run_assembly::encode_u32_words;
use super::synthetic_count::SyntheticPattern;
use super::synthetic_programs::pattern_buffers;

pub(super) fn mixed_release_index(index: u32, rounds: u32) -> u32 {
    let mut hash = index ^ 0x9E37_79B9;
    for lane in 0..rounds {
        hash = hash
            .rotate_left(5)
            .wrapping_mul(0x85EB_CA6B)
            .wrapping_add(0xC2B2_AE35 ^ lane);
    }
    hash
}

pub(super) struct SyntheticInputs {
    pub(super) inputs: Vec<Vec<u8>>,
    pub(super) expected: u32,
}

pub(super) struct StringBitmapScatterInputs {
    pub(super) inputs: Vec<Vec<u8>>,
    pub(super) pattern_bitmap: Vec<u32>,
    pub(super) rule_bitmap: Vec<u32>,
}

pub(super) fn string_bitmap_scatter_inputs(records: u32) -> StringBitmapScatterInputs {
    let output_words = records.div_ceil(32) as usize;
    let mut pattern_bitmap = Vec::with_capacity(records as usize);
    let mut rule_bitmap = Vec::with_capacity(records as usize);
    for index in 0..records {
        let row = synthetic_row(SyntheticPattern::StringBitmapScatter, index);
        pattern_bitmap.push(row[0]);
        rule_bitmap.push(row[1]);
    }
    let inputs = vec![
        vec![0u8; output_words * 4],
        encode_u32_words(&pattern_bitmap),
        encode_u32_words(&rule_bitmap),
    ];
    StringBitmapScatterInputs {
        inputs,
        pattern_bitmap,
        rule_bitmap,
    }
}

pub(super) fn string_bitmap_scatter_expected_words(
    pattern_bitmap: &[u32],
    rule_bitmap: &[u32],
    records: u32,
) -> Vec<u32> {
    let mut expected_words = vec![0u32; records.div_ceil(32) as usize];
    for index in 0..records {
        if pattern_bitmap[index as usize] != 0 && rule_bitmap[index as usize] != 0 {
            expected_words[(index / 32) as usize] |= 1u32 << (index & 31);
        }
    }
    expected_words
}

pub(super) fn pattern_input_count(pattern: SyntheticPattern) -> usize {
    pattern_buffers(pattern).len()
}

pub(super) fn synthetic_output_reset_bytes(pattern: SyntheticPattern, _records: u32) -> usize {
    match pattern {
        SyntheticPattern::StringBitmapScatter => 0,
        _ => 4,
    }
}

pub(super) fn synthetic_logical_output_bytes(pattern: SyntheticPattern, records: u32) -> u64 {
    match pattern {
        SyntheticPattern::StringBitmapScatter => u64::from(records.div_ceil(32)) * 4,
        _ => 4,
    }
}

pub(super) fn synthetic_inputs(pattern: SyntheticPattern, records: u32) -> SyntheticInputs {
    let mut columns = (0..pattern_input_count(pattern))
        .map(|_| Vec::with_capacity(records as usize))
        .collect::<Vec<Vec<u32>>>();
    let mut expected = 0u32;
    for index in 0..records {
        let row = synthetic_row(pattern, index);
        expected += u32::from(row_matches(pattern, &row));
        for (column, value) in columns.iter_mut().zip(row) {
            column.push(value);
        }
    }
    let mut inputs = Vec::with_capacity(columns.len());
    inputs.extend(columns.iter().map(|column| encode_u32_words(column)));
    SyntheticInputs { inputs, expected }
}

pub(super) fn synthetic_cpu_count(pattern: SyntheticPattern, records: u32) -> u32 {
    (0..records)
        .map(|index| u32::from(row_matches(pattern, &synthetic_row(pattern, index))))
        .sum()
}

/// Count matching rows out of the host input buffers the dispatch reads.
///
/// `synthetic_cpu_count` regenerates every row from its index, which costs several
/// rotate-multiply rounds per column and is work the dispatched program never does:
/// its inputs are materialized and uploaded before the clock starts. Timing the
/// generator on one side only inflates every speedup this harness reports, so the
/// baseline the contract is judged against reads the same bytes the device reads.
pub(super) fn synthetic_cpu_count_over_inputs(
    pattern: SyntheticPattern,
    inputs: &[Vec<u8>],
    records: u32,
) -> Result<u32, String> {
    let columns = pattern_input_count(pattern);
    if inputs.len() < columns {
        return Err(format!(
            "synthetic CPU baseline needs {columns} input column(s) for this pattern but received {}. Fix: pass the prepared input buffers.",
            inputs.len()
        ));
    }
    let required = records as usize * std::mem::size_of::<u32>();
    for (index, column) in inputs.iter().take(columns).enumerate() {
        if column.len() < required {
            return Err(format!(
                "synthetic CPU baseline input column {index} holds {} bytes but records={records} needs {required}. Fix: generate the column at the benchmarked record count.",
                column.len()
            ));
        }
    }
    let mut row = vec![0u32; columns];
    let mut matches = 0u32;
    for record in 0..records as usize {
        let start = record * std::mem::size_of::<u32>();
        for (slot, column) in row.iter_mut().zip(inputs) {
            *slot = u32::from_le_bytes([
                column[start],
                column[start + 1],
                column[start + 2],
                column[start + 3],
            ]);
        }
        matches += u32::from(row_matches(pattern, &row));
    }
    Ok(matches)
}

/// Name the CPU baseline this harness times, owned by the module that implements it.
///
/// Every release macro workload is judged against the same in-process reference, so
/// the label is derived from the pattern instead of restated per workload, where a
/// case could name an engine that never runs.
pub(super) fn synthetic_baseline_label(pattern: SyntheticPattern) -> &'static str {
    match pattern {
        SyntheticPattern::StringBitmapScatter => {
            "single-threaded scalar CPU reference bitmap materialization over the same host input buffers (string_bitmap_scatter_expected_words)"
        }
        _ => {
            "single-threaded scalar CPU reference predicate count over the same host input buffers (synthetic_cpu_count_over_inputs)"
        }
    }
}

fn synthetic_row(pattern: SyntheticPattern, index: u32) -> Vec<u32> {
    match pattern {
        SyntheticPattern::ConditionEval => vec![
            condition_match_mask(index),
            condition_rule_mask(index),
            condition_metadata_mask(index),
        ],
        SyntheticPattern::StringBitmapScatter => vec![
            string_bitmap_pattern_word(index),
            string_bitmap_rule_word(index),
        ],
        SyntheticPattern::OffsetCountAggregation => vec![
            aggregation_offset_mask(index),
            aggregation_length_mask(index),
            aggregation_count_mask(index),
        ],
        SyntheticPattern::EntropyWindow => vec![
            entropy_byte_class_mask(index),
            entropy_transition_mask(index),
            entropy_rarity_mask(index),
        ],
        SyntheticPattern::QuantifiedLoops => vec![
            quantified_any_mask(index),
            quantified_all_mask(index),
            quantified_threshold_mask(index),
        ],
        SyntheticPattern::AliasReachingDef => vec![
            alias_def_mask(index),
            alias_use_mask(index),
            alias_kill_mask(index),
        ],
        SyntheticPattern::IfdsWitness => vec![
            ifds_frontier_mask(index),
            ifds_transfer_mask(index),
            ifds_witness_mask(index),
        ],
        SyntheticPattern::AstMotifTraversal => vec![
            ast_node_kind_mask(index),
            ast_depth_mask(index),
            ast_motif_mask(index),
        ],
        SyntheticPattern::MegakernelQueuedBatch => vec![
            megakernel_queue_mask(index),
            megakernel_predicate_mask(index),
            megakernel_dispatch_mask(index),
        ],
        SyntheticPattern::EgraphSaturation => {
            vec![
                egraph_opcode_mask(index),
                egraph_lhs_class_mask(index),
                egraph_rhs_class_mask(index),
            ]
        }
    }
}

fn string_bitmap_pattern_word(index: u32) -> u32 {
    let hash = mixed_release_index(index, 24);
    u32::from(index % 29 == 0 || index % 211 == 3 || hash == 0)
}

fn string_bitmap_rule_word(index: u32) -> u32 {
    let mut hash = index.wrapping_add(0x27D4_EB2D);
    for lane in 0..12 {
        hash = hash
            .rotate_right(7)
            .wrapping_mul(0x1656_67B1)
            .wrapping_add(0xD3A2_646C ^ lane);
    }
    u32::from(index % 7 != 0 && hash != u32::MAX)
}

pub(super) const CONDITION_LANES: u32 = 16;

pub(super) const CONDITION_THRESHOLD: u32 = 6;

const CONDITION_LANE_MASK: u32 = (1u32 << CONDITION_LANES) - 1;

fn condition_match_mask(index: u32) -> u32 {
    let mut state = index ^ 0xB529_7A4D;
    let mut mask = 0u32;
    for lane in 0..CONDITION_LANES {
        state = state
            .rotate_left(5)
            .wrapping_mul(0x68E3_1DA4)
            .wrapping_add(lane ^ 0x1B56_C4E9);
        if state & 0x5 != 0 {
            mask |= 1u32 << lane;
        }
    }
    if index % 31 == 0 {
        mask | 0x3F3F
    } else {
        mask & 0x5A5A
    }
}

fn condition_rule_mask(index: u32) -> u32 {
    let rotated = condition_match_mask(index).rotate_left((index & 7) + 1) & CONDITION_LANE_MASK;
    if index % 31 == 0 {
        rotated | 0x3F3F
    } else {
        rotated & 0x33CC
    }
}

fn condition_metadata_mask(index: u32) -> u32 {
    if index % 31 == 0 {
        0x3F3F
    } else {
        0x0F0F ^ (1u32 << (index & (CONDITION_LANES - 1)))
    }
}

fn condition_eval_matches(match_mask: u32, rule_mask: u32, metadata_mask: u32) -> bool {
    let mut condition_hits = 0u32;
    for lane in 0..CONDITION_LANES {
        let bit = 1u32 << lane;
        if match_mask & bit != 0 && rule_mask & bit != 0 && metadata_mask & bit != 0 {
            condition_hits += 1;
        }
    }
    condition_hits >= CONDITION_THRESHOLD
}

pub(super) const AGGREGATION_LANES: u32 = 16;

pub(super) const AGGREGATION_THRESHOLD: u32 = 7;

const AGGREGATION_LANE_MASK: u32 = (1u32 << AGGREGATION_LANES) - 1;

fn aggregation_offset_mask(index: u32) -> u32 {
    let mut state = index ^ 0xC13F_A9A9;
    let mut mask = 0u32;
    for lane in 0..AGGREGATION_LANES {
        state = state
            .rotate_left(11)
            .wrapping_mul(0x9E37_79B1)
            .wrapping_add(lane ^ 0x85EB_CA77);
        if state & 0xD != 0 {
            mask |= 1u32 << lane;
        }
    }
    if index % 43 == 0 {
        mask | 0x7F7F
    } else {
        mask & 0x6DB6
    }
}

fn aggregation_length_mask(index: u32) -> u32 {
    let rotated =
        aggregation_offset_mask(index).rotate_right((index & 7) + 1) & AGGREGATION_LANE_MASK;
    if index % 43 == 0 {
        rotated | 0x7F7F
    } else {
        rotated & 0x3F3C
    }
}

fn aggregation_count_mask(index: u32) -> u32 {
    if index % 43 == 0 {
        0x7F7F
    } else {
        0x1F1F ^ (1u32 << (index & (AGGREGATION_LANES - 1)))
    }
}

fn offset_count_aggregation_matches(offset_mask: u32, length_mask: u32, count_mask: u32) -> bool {
    let mut aggregation_hits = 0u32;
    for lane in 0..AGGREGATION_LANES {
        let bit = 1u32 << lane;
        if offset_mask & bit != 0 && length_mask & bit != 0 && count_mask & bit != 0 {
            aggregation_hits += 1;
        }
    }
    aggregation_hits >= AGGREGATION_THRESHOLD
}

pub(super) const ENTROPY_LANES: u32 = 16;

pub(super) const ENTROPY_THRESHOLD: u32 = 9;

const ENTROPY_LANE_MASK: u32 = (1u32 << ENTROPY_LANES) - 1;

fn entropy_byte_class_mask(index: u32) -> u32 {
    let mut state = index ^ 0xA24B_AED5;
    let mut mask = 0u32;
    for lane in 0..ENTROPY_LANES {
        state = state
            .rotate_left(13)
            .wrapping_mul(0x9FB2_1C65)
            .wrapping_add(lane ^ 0xC2B2_AE3D);
        if state & 0x17 != 0 {
            mask |= 1u32 << lane;
        }
    }
    if index % 47 == 0 {
        mask | 0x7FFF
    } else {
        mask & 0x6B6D
    }
}

fn entropy_transition_mask(index: u32) -> u32 {
    let rotated = entropy_byte_class_mask(index).rotate_left((index & 7) + 1) & ENTROPY_LANE_MASK;
    if index % 47 == 0 {
        rotated | 0x7E7E
    } else {
        rotated & 0x35B5
    }
}

fn entropy_rarity_mask(index: u32) -> u32 {
    if index % 47 == 0 {
        0x7E7E
    } else {
        0x2D2D ^ (1u32 << (index & (ENTROPY_LANES - 1)))
    }
}

fn entropy_window_matches(byte_class_mask: u32, transition_mask: u32, rarity_mask: u32) -> bool {
    let mut entropy_score = 0u32;
    for lane in 0..ENTROPY_LANES {
        let bit = 1u32 << lane;
        if byte_class_mask & bit != 0 && (transition_mask & bit != 0 || rarity_mask & bit != 0) {
            entropy_score += 1;
        }
    }
    entropy_score >= ENTROPY_THRESHOLD
}

pub(super) const QUANTIFIED_LANES: u32 = 16;

pub(super) const QUANTIFIED_THRESHOLD: u32 = 11;

const QUANTIFIED_LANE_MASK: u32 = (1u32 << QUANTIFIED_LANES) - 1;

fn quantified_any_mask(index: u32) -> u32 {
    let mut mask = 0u32;
    let mut state = index ^ 0xA511_E9B3;
    for lane in 0..QUANTIFIED_LANES {
        state = state
            .rotate_left(3)
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(lane ^ 0x7F4A_7C15);
        if state & 0x13 != 0 {
            mask |= 1u32 << lane;
        }
    }
    mask
}

fn quantified_all_mask(index: u32) -> u32 {
    if index % 29 == 0 {
        QUANTIFIED_LANE_MASK
    } else {
        QUANTIFIED_LANE_MASK ^ (1u32 << (index & (QUANTIFIED_LANES - 1)))
    }
}

fn quantified_threshold_mask(index: u32) -> u32 {
    let mut mask = 0u32;
    let mut state = index.wrapping_mul(0x045D_9F3B);
    for lane in 0..QUANTIFIED_LANES {
        state = state.rotate_right(5).wrapping_add(0x27D4_EB2D ^ lane);
        if state.count_ones() >= 14 || (index.wrapping_add(lane) % 5 == 0) {
            mask |= 1u32 << lane;
        }
    }
    mask
}

fn quantified_row_matches(any_mask: u32, all_mask: u32, threshold_mask: u32) -> bool {
    let mut any_seen = false;
    let mut threshold_hits = 0u32;
    for lane in 0..QUANTIFIED_LANES {
        let bit = 1u32 << lane;
        any_seen |= any_mask & bit != 0;
        if all_mask & bit == 0 {
            return false;
        }
        threshold_hits += u32::from(threshold_mask & bit != 0);
    }
    any_seen && threshold_hits >= QUANTIFIED_THRESHOLD
}

pub(super) const ALIAS_LANES: u32 = 16;

pub(super) const ALIAS_THRESHOLD: u32 = 4;

const ALIAS_LANE_MASK: u32 = (1u32 << ALIAS_LANES) - 1;

fn alias_def_mask(index: u32) -> u32 {
    let mut state = index ^ 0x6C8E_9CF5;
    let mut mask = 0u32;
    for lane in 0..ALIAS_LANES {
        state = state
            .rotate_left(7)
            .wrapping_mul(0x7FEB_352D)
            .wrapping_add(lane ^ 0x846C_A68B);
        if state & 0x7 != 0 {
            mask |= 1u32 << lane;
        }
    }
    if index % 37 == 0 {
        mask | 0x00F3
    } else {
        mask & 0x5555
    }
}

fn alias_use_mask(index: u32) -> u32 {
    let shifted = alias_def_mask(index).rotate_left((index & 7) + 1) & ALIAS_LANE_MASK;
    if index % 37 == 0 {
        shifted | 0x00F3
    } else {
        shifted & 0x3333
    }
}

fn alias_kill_mask(index: u32) -> u32 {
    if index % 37 == 0 {
        ALIAS_LANE_MASK ^ 0x00F3
    } else {
        0xAAAA | (1u32 << (index & (ALIAS_LANES - 1)))
    }
}

fn alias_reaching_def_matches(def_mask: u32, use_mask: u32, kill_mask: u32) -> bool {
    let mut reaching_aliases = 0u32;
    for lane in 0..ALIAS_LANES {
        let bit = 1u32 << lane;
        if def_mask & bit != 0 && use_mask & bit != 0 && kill_mask & bit == 0 {
            reaching_aliases += 1;
        }
    }
    reaching_aliases >= ALIAS_THRESHOLD
}

pub(super) const IFDS_LANES: u32 = 16;

pub(super) const IFDS_THRESHOLD: u32 = 5;

const IFDS_LANE_MASK: u32 = (1u32 << IFDS_LANES) - 1;

fn ifds_frontier_mask(index: u32) -> u32 {
    let mut state = index.wrapping_add(0xD1B5_4A35);
    let mut mask = 0u32;
    for lane in 0..IFDS_LANES {
        state = state
            .rotate_left(9)
            .wrapping_mul(0x94D0_49BB)
            .wrapping_add(lane ^ 0x2545_F491);
        if state & 0xB != 0 {
            mask |= 1u32 << lane;
        }
    }
    if index % 41 == 0 {
        mask | 0x1F1F
    } else {
        mask & 0x5A5A
    }
}

fn ifds_transfer_mask(index: u32) -> u32 {
    let rotated = ifds_frontier_mask(index).rotate_right((index & 7) + 1) & IFDS_LANE_MASK;
    if index % 41 == 0 {
        rotated | 0x1F1F
    } else {
        rotated & 0x3C3C
    }
}

fn ifds_witness_mask(index: u32) -> u32 {
    if index % 41 == 0 {
        0x1F1F
    } else {
        0x00F0 ^ (1u32 << (index & (IFDS_LANES - 1)))
    }
}

fn ifds_witness_matches(frontier_mask: u32, transfer_mask: u32, witness_mask: u32) -> bool {
    let mut witness_hits = 0u32;
    for lane in 0..IFDS_LANES {
        let bit = 1u32 << lane;
        if frontier_mask & bit != 0 && transfer_mask & bit != 0 && witness_mask & bit != 0 {
            witness_hits += 1;
        }
    }
    witness_hits >= IFDS_THRESHOLD
}

pub(super) const C_AST_LANES: u32 = 16;

pub(super) const C_AST_THRESHOLD: u32 = 6;

const AST_MOTIF_LANE_MASK: u32 = (1u32 << C_AST_LANES) - 1;

fn ast_node_kind_mask(index: u32) -> u32 {
    let mut state = index ^ 0xDEAD_BEEF;
    let mut mask = 0u32;
    for lane in 0..C_AST_LANES {
        state = state
            .rotate_left(3)
            .wrapping_mul(0x85EB_CA6B)
            .wrapping_add(lane ^ 0x27D4_EB2D);
        if state & 0xB != 0 {
            mask |= 1u32 << lane;
        }
    }
    if index % 53 == 0 {
        mask | 0x3F3F
    } else {
        mask & 0x5B5B
    }
}

fn ast_depth_mask(index: u32) -> u32 {
    let rotated = ast_node_kind_mask(index).rotate_right((index & 7) + 1) & AST_MOTIF_LANE_MASK;
    if index % 53 == 0 {
        rotated | 0x3F3F
    } else {
        rotated & 0x33F0
    }
}

fn ast_motif_mask(index: u32) -> u32 {
    if index % 53 == 0 {
        0x3F3F
    } else {
        0x0FF0 ^ (1u32 << (index & (C_AST_LANES - 1)))
    }
}

fn ast_motif_traversal_matches(node_kind_mask: u32, depth_mask: u32, motif_mask: u32) -> bool {
    let mut ast_hits = 0u32;
    for lane in 0..C_AST_LANES {
        let bit = 1u32 << lane;
        if node_kind_mask & bit != 0 && depth_mask & bit != 0 && motif_mask & bit != 0 {
            ast_hits += 1;
        }
    }
    ast_hits >= C_AST_THRESHOLD
}

pub(super) const MEGAKERNEL_QUEUE_LANES: u32 = 16;

pub(super) const MEGAKERNEL_QUEUE_THRESHOLD: u32 = 6;

const MEGAKERNEL_QUEUE_LANE_MASK: u32 = (1u32 << MEGAKERNEL_QUEUE_LANES) - 1;

fn megakernel_queue_mask(index: u32) -> u32 {
    let mut state = index ^ 0x8CB9_2BA7;
    let mut mask = 0u32;
    for lane in 0..MEGAKERNEL_QUEUE_LANES {
        state = state
            .rotate_left(7)
            .wrapping_mul(0xC2B2_AE35)
            .wrapping_add(lane ^ 0x27D4_EB2F);
        if state & 0x7 != 0 {
            mask |= 1u32 << lane;
        }
    }
    if index % 59 == 0 {
        mask | 0x3F3F
    } else {
        mask & 0x56D6
    }
}

fn megakernel_predicate_mask(index: u32) -> u32 {
    let rotated =
        megakernel_queue_mask(index).rotate_right((index & 7) + 1) & MEGAKERNEL_QUEUE_LANE_MASK;
    if index % 59 == 0 {
        rotated | 0x3F3F
    } else {
        rotated & 0x333C
    }
}

fn megakernel_dispatch_mask(index: u32) -> u32 {
    if index % 59 == 0 {
        0x3F3F
    } else {
        0x0F0F ^ (1u32 << (index & (MEGAKERNEL_QUEUE_LANES - 1)))
    }
}

fn megakernel_queue_matches(queue_mask: u32, predicate_mask: u32, dispatch_mask: u32) -> bool {
    let mut queued_hits = 0u32;
    for lane in 0..MEGAKERNEL_QUEUE_LANES {
        let bit = 1u32 << lane;
        if queue_mask & bit != 0 && predicate_mask & bit != 0 && dispatch_mask & bit != 0 {
            queued_hits += 1;
        }
    }
    queued_hits >= MEGAKERNEL_QUEUE_THRESHOLD
}

pub(super) const EGRAPH_LANES: u32 = 16;

pub(super) const EGRAPH_THRESHOLD: u32 = 7;

const EGRAPH_LANE_MASK: u32 = (1u32 << EGRAPH_LANES) - 1;

fn egraph_opcode_mask(index: u32) -> u32 {
    let mut state = index ^ 0xA409_3822;
    let mut mask = 0u32;
    for lane in 0..EGRAPH_LANES {
        state = state
            .rotate_left(9)
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(lane ^ 0x299F_31D0);
        if state & 0xD != 0 {
            mask |= 1u32 << lane;
        }
    }
    if index % 61 == 0 {
        mask | 0x7F7F
    } else {
        mask & 0x5DB5
    }
}

fn egraph_lhs_class_mask(index: u32) -> u32 {
    let rotated = egraph_opcode_mask(index).rotate_left((index & 7) + 1) & EGRAPH_LANE_MASK;
    if index % 61 == 0 {
        rotated | 0x7F7F
    } else {
        rotated & 0x3F33
    }
}

fn egraph_rhs_class_mask(index: u32) -> u32 {
    if index % 61 == 0 {
        0x7F7F
    } else {
        0x1F1F ^ (1u32 << (index & (EGRAPH_LANES - 1)))
    }
}

fn egraph_saturation_matches(opcode_mask: u32, lhs_class_mask: u32, rhs_class_mask: u32) -> bool {
    let mut rewrite_hits = 0u32;
    for lane in 0..EGRAPH_LANES {
        let bit = 1u32 << lane;
        if opcode_mask & bit != 0 && lhs_class_mask & bit != 0 && rhs_class_mask & bit != 0 {
            rewrite_hits += 1;
        }
    }
    rewrite_hits >= EGRAPH_THRESHOLD
}

fn row_matches(pattern: SyntheticPattern, row: &[u32]) -> bool {
    match pattern {
        SyntheticPattern::ConditionEval => condition_eval_matches(row[0], row[1], row[2]),
        SyntheticPattern::StringBitmapScatter => row[0] != 0 && row[1] != 0,
        SyntheticPattern::OffsetCountAggregation => {
            offset_count_aggregation_matches(row[0], row[1], row[2])
        }
        SyntheticPattern::EntropyWindow => entropy_window_matches(row[0], row[1], row[2]),
        SyntheticPattern::QuantifiedLoops => quantified_row_matches(row[0], row[1], row[2]),
        SyntheticPattern::AliasReachingDef => alias_reaching_def_matches(row[0], row[1], row[2]),
        SyntheticPattern::IfdsWitness => ifds_witness_matches(row[0], row[1], row[2]),
        SyntheticPattern::AstMotifTraversal => ast_motif_traversal_matches(row[0], row[1], row[2]),
        SyntheticPattern::MegakernelQueuedBatch => megakernel_queue_matches(row[0], row[1], row[2]),
        SyntheticPattern::EgraphSaturation => egraph_saturation_matches(row[0], row[1], row[2]),
    }
}

#[cfg(test)]
mod tests {
    use super::super::synthetic_programs::{
        build_synthetic_release_program, string_bitmap_scatter_program,
        string_bitmap_scatter_program_with_batch,
    };
    use super::*;
    use vyre::ir::BufferAccess;

    #[test]
    fn string_bitmap_scatter_inputs_match_program_abi_at_word_boundaries() {
        for records in [1, 31, 32, 33, 255, 256, 257, 1024] {
            let program = string_bitmap_scatter_program(records);
            let generated = string_bitmap_scatter_inputs(records);
            let output_words = records.div_ceil(32) as usize;

            assert_eq!(
                generated.inputs.len(),
                3,
                "records={records} must pass initialized out_flags plus read-only bitmap inputs"
            );
            assert_eq!(generated.inputs[0].len(), output_words * 4);
            assert_eq!(generated.inputs[1].len(), records as usize * 4);
            assert_eq!(generated.inputs[2].len(), records as usize * 4);
            assert_eq!(program.buffers()[0].name.as_ref(), "out_flags");
            assert_eq!(program.buffers()[0].count, records.div_ceil(32));
            assert_eq!(program.buffers()[0].access(), BufferAccess::ReadWrite);
            assert_eq!(
                program.buffers()[0].output_byte_range(),
                Some(0..output_words * 4)
            );
            assert_eq!(program.buffers()[1].name.as_ref(), "pattern_bitmap");
            assert_eq!(program.buffers()[1].count, records);
            assert_eq!(program.buffers()[2].name.as_ref(), "rule_bitmap");
            assert_eq!(program.buffers()[2].count, records);
        }
    }

    #[test]
    fn string_bitmap_scatter_reference_eval_matches_cpu_bitmap_oracle() {
        for records in [1, 17, 32, 33, 127, 257] {
            let program = string_bitmap_scatter_program(records);
            let generated = string_bitmap_scatter_inputs(records);
            let values = generated
                .inputs
                .iter()
                .cloned()
                .map(vyre_reference::value::Value::from)
                .collect::<Vec<_>>();
            let outputs = vyre_reference::reference_eval(&program, &values)
                .expect("Fix: string bitmap scatter must reference-evaluate")
                .into_iter()
                .map(|value| value.to_bytes())
                .collect::<Vec<_>>();

            let mut expected_words = vec![0u32; records.div_ceil(32) as usize];
            for index in 0..records {
                let pattern_word = generated.pattern_bitmap[index as usize];
                let rule_word = generated.rule_bitmap[index as usize];
                assert_eq!(pattern_word, string_bitmap_pattern_word(index));
                assert_eq!(rule_word, string_bitmap_rule_word(index));
                if pattern_word != 0 && rule_word != 0 {
                    expected_words[(index / 32) as usize] |= 1u32 << (index & 31);
                }
            }

            assert_eq!(
                outputs,
                vec![encode_u32_words(&expected_words)],
                "records={records} must scatter the full CPU oracle bitmap"
            );
        }
    }

    /// WHY: throughput batching may share immutable bitmap inputs, but every logical row must
    /// still receive the complete bitmap rather than a partial or aliased output.
    #[test]
    fn string_bitmap_scatter_batched_rows_match_independent_oracles() {
        let records = 257;
        let batch_size = 4;
        let program = string_bitmap_scatter_program_with_batch(records, batch_size);
        let mut generated = string_bitmap_scatter_inputs(records);
        let expected_words = string_bitmap_scatter_expected_words(
            &generated.pattern_bitmap,
            &generated.rule_bitmap,
            records,
        );
        let expected_row = encode_u32_words(&expected_words);
        generated.inputs[0].resize(expected_row.len() * batch_size as usize, 0);
        let values = generated
            .inputs
            .iter()
            .cloned()
            .map(vyre_reference::value::Value::from)
            .collect::<Vec<_>>();

        let outputs = vyre_reference::reference_eval(&program, &values)
            .expect("Fix: batched string bitmap scatter must reference-evaluate")
            .into_iter()
            .map(|value| value.to_bytes())
            .collect::<Vec<_>>();

        assert_eq!(outputs, vec![expected_row.repeat(batch_size as usize)]);
    }
    /// WHY: release.condition_eval.1m uses bitwise popcount reduction instead of an unrolled loop;
    /// reference evaluation must produce byte-exact agreement with the scalar CPU oracle.
    #[test]
    fn condition_eval_reference_eval_matches_cpu_oracle() {
        for records in [1, 17, 32, 33, 64, 127, 256, 1024] {
            let program = build_synthetic_release_program(SyntheticPattern::ConditionEval, records);
            let generated = synthetic_inputs(SyntheticPattern::ConditionEval, records);
            let values = generated
                .inputs
                .iter()
                .cloned()
                .map(vyre_reference::value::Value::from)
                .collect::<Vec<_>>();
            let outputs = vyre_reference::reference_eval(&program, &values)
                .expect("Fix: condition eval program must reference-evaluate")
                .into_iter()
                .map(|value| value.to_bytes())
                .collect::<Vec<_>>();

            assert_eq!(
                outputs,
                vec![encode_u32_words(&[generated.expected])],
                "records={records} condition eval output must match CPU oracle count"
            );
        }
    }

    /// WHY: all triple-mask threshold count programs share the bitwise popcount fast path;
    /// every pattern variant across all predicates must match its CPU oracle.
    #[test]
    fn all_triple_mask_programs_reference_eval_match_cpu_oracles() {
        let patterns = [
            SyntheticPattern::ConditionEval,
            SyntheticPattern::OffsetCountAggregation,
            SyntheticPattern::EntropyWindow,
            SyntheticPattern::AliasReachingDef,
            SyntheticPattern::IfdsWitness,
            SyntheticPattern::AstMotifTraversal,
            SyntheticPattern::MegakernelQueuedBatch,
            SyntheticPattern::EgraphSaturation,
        ];
        for pattern in patterns {
            for records in [1, 31, 64, 128] {
                let program = build_synthetic_release_program(pattern, records);
                let generated = synthetic_inputs(pattern, records);
                let values = generated
                    .inputs
                    .iter()
                    .cloned()
                    .map(vyre_reference::value::Value::from)
                    .collect::<Vec<_>>();
                let outputs = vyre_reference::reference_eval(&program, &values)
                    .unwrap_or_else(|err| {
                        panic!("pattern {pattern:?} records {records} failed reference eval: {err}")
                    })
                    .into_iter()
                    .map(|value| value.to_bytes())
                    .collect::<Vec<_>>();

                assert_eq!(
                    outputs,
                    vec![encode_u32_words(&[generated.expected])],
                    "pattern={pattern:?} records={records} output must match CPU oracle count"
                );
            }
        }
    }
}
