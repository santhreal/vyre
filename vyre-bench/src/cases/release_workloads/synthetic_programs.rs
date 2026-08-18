//! IR program builders for the synthetic release macro patterns.

use super::synthetic_count::{SyntheticPattern, STRING_BITMAP_RESIDENT_BATCH_SIZE};
use super::synthetic_oracle::{
    AGGREGATION_LANES, AGGREGATION_THRESHOLD, ALIAS_LANES, ALIAS_THRESHOLD, CONDITION_LANES,
    CONDITION_THRESHOLD, C_AST_LANES, C_AST_THRESHOLD, EGRAPH_LANES, EGRAPH_THRESHOLD,
    ENTROPY_LANES, ENTROPY_THRESHOLD, IFDS_LANES, IFDS_THRESHOLD, MEGAKERNEL_QUEUE_LANES,
    MEGAKERNEL_QUEUE_THRESHOLD, QUANTIFIED_LANES, QUANTIFIED_THRESHOLD,
};
use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

pub(super) fn build_synthetic_release_program(pattern: SyntheticPattern, records: u32) -> Program {
    match pattern {
        SyntheticPattern::ConditionEval => condition_eval_program(records),
        SyntheticPattern::StringBitmapScatter => string_bitmap_scatter_program(records),
        SyntheticPattern::OffsetCountAggregation => offset_count_aggregation_program(records),
        SyntheticPattern::EntropyWindow => entropy_window_program(records),
        SyntheticPattern::QuantifiedLoops => quantified_condition_loops_program(records),
        SyntheticPattern::AliasReachingDef => alias_reaching_def_program(records),
        SyntheticPattern::IfdsWitness => ifds_witness_program(records),
        SyntheticPattern::AstMotifTraversal => ast_motif_traversal_program(records),
        SyntheticPattern::MegakernelQueuedBatch => megakernel_queue_program(records),
        SyntheticPattern::EgraphSaturation => egraph_saturation_program(records),
    }
}

fn synthetic_count_program(pattern: SyntheticPattern, records: u32) -> Program {
    let mut buffers = vec![BufferDecl::output("out_count", 0, DataType::U32).with_count(1)];
    for (binding, name) in pattern_buffers(pattern).iter().enumerate() {
        buffers.push(
            BufferDecl::storage(
                name,
                (binding + 1) as u32,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(records),
        );
    }
    Program::wrapped(
        buffers,
        [256, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::and(
                    Expr::lt(Expr::var("idx"), Expr::u32(records)),
                    pattern_condition(pattern),
                ),
                vec![Node::let_bind(
                    "_slot",
                    Expr::atomic_add("out_count", Expr::u32(0), Expr::u32(1)),
                )],
            ),
        ],
    )
}

fn condition_eval_program(records: u32) -> Program {
    triple_mask_threshold_count_program(
        records,
        ["match_mask", "rule_mask", "metadata_mask"],
        ["match_word", "rule_word", "metadata_word"],
        "condition_hits",
        TripleMaskPredicate::AllSet,
        CONDITION_LANES,
        CONDITION_THRESHOLD,
    )
}

pub(super) fn string_bitmap_scatter_program(records: u32) -> Program {
    string_bitmap_scatter_program_with_batch(records, 1)
}

pub(super) fn string_bitmap_scatter_release_program(records: u32) -> Program {
    string_bitmap_scatter_program_with_batch(records, STRING_BITMAP_RESIDENT_BATCH_SIZE as u32)
}

pub(super) fn string_bitmap_scatter_program_with_batch(records: u32, batch_size: u32) -> Program {
    let output_words = records.div_ceil(32);
    let total_output_words = output_words * batch_size;
    let record_idx = Expr::var("record_idx");
    let selected = Expr::and(
        Expr::lt(record_idx.clone(), Expr::u32(records)),
        Expr::and(
            Expr::ne(
                Expr::load("pattern_bitmap", record_idx.clone()),
                Expr::u32(0),
            ),
            Expr::ne(Expr::load("rule_bitmap", record_idx.clone()), Expr::u32(0)),
        ),
    );
    Program::wrapped(
        vec![
            BufferDecl::storage("out_flags", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(total_output_words)
                .with_output_byte_range(0..(output_words as usize * 4)),
            BufferDecl::storage("pattern_bitmap", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(records),
            BufferDecl::storage("rule_bitmap", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(records),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("record_idx", Expr::gid_x()),
            Node::let_bind("scatter_word", Expr::subgroup_ballot(selected)),
            Node::if_then(
                Expr::and(
                    Expr::eq(Expr::SubgroupLocalId, Expr::u32(0)),
                    Expr::lt(record_idx.clone(), Expr::u32(records)),
                ),
                vec![Node::loop_for(
                    "scatter_batch",
                    Expr::u32(0),
                    Expr::u32(batch_size),
                    vec![Node::store(
                        "out_flags",
                        Expr::add(
                            Expr::mul(Expr::var("scatter_batch"), Expr::u32(output_words)),
                            Expr::shr(record_idx, Expr::u32(5)),
                        ),
                        Expr::var("scatter_word"),
                    )],
                )],
            ),
        ],
    )
}

#[derive(Clone, Copy)]
enum TripleMaskPredicate {
    AllSet,
    FirstAndEither,
    FirstTwoSetThirdClear,
}

fn triple_mask_threshold_count_program(
    records: u32,
    buffers: [&'static str; 3],
    words: [&'static str; 3],
    _hits: &'static str,
    predicate: TripleMaskPredicate,
    lanes: u32,
    threshold: u32,
) -> Program {
    let combined_word = match predicate {
        TripleMaskPredicate::AllSet => Expr::bitand(
            Expr::var(words[0]),
            Expr::bitand(Expr::var(words[1]), Expr::var(words[2])),
        ),
        TripleMaskPredicate::FirstAndEither => Expr::bitand(
            Expr::var(words[0]),
            Expr::bitor(Expr::var(words[1]), Expr::var(words[2])),
        ),
        TripleMaskPredicate::FirstTwoSetThirdClear => Expr::bitand(
            Expr::var(words[0]),
            Expr::bitand(Expr::var(words[1]), Expr::bitnot(Expr::var(words[2]))),
        ),
    };
    let masked_word = if lanes < 32 {
        Expr::bitand(combined_word, Expr::u32((1u32 << lanes) - 1))
    } else {
        combined_word
    };
    let condition = Expr::ge(Expr::popcount(masked_word), Expr::u32(threshold));

    Program::wrapped(
        vec![
            BufferDecl::output("out_count", 0, DataType::U32).with_count(1),
            BufferDecl::storage(buffers[0], 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(records),
            BufferDecl::storage(buffers[1], 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(records),
            BufferDecl::storage(buffers[2], 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(records),
            BufferDecl::workgroup("warp_scratch", 1024, DataType::U32),
        ],
        [256, 1, 1],
        warp_reduction_count_nodes(
            256,
            records,
            Expr::and(Expr::var("in_bounds"), condition),
            vec![
                Node::let_bind(words[0], Expr::load(buffers[0], Expr::var("safe_idx"))),
                Node::let_bind(words[1], Expr::load(buffers[1], Expr::var("safe_idx"))),
                Node::let_bind(words[2], Expr::load(buffers[2], Expr::var("safe_idx"))),
            ],
        ),
    )
}

fn offset_count_aggregation_program(records: u32) -> Program {
    triple_mask_threshold_count_program(
        records,
        ["offset_mask", "length_mask", "count_mask"],
        ["offset_word", "length_word", "count_word"],
        "aggregation_hits",
        TripleMaskPredicate::AllSet,
        AGGREGATION_LANES,
        AGGREGATION_THRESHOLD,
    )
}

fn entropy_window_program(records: u32) -> Program {
    triple_mask_threshold_count_program(
        records,
        ["byte_class_mask", "transition_mask", "rarity_mask"],
        ["byte_class_word", "transition_word", "rarity_word"],
        "entropy_score",
        TripleMaskPredicate::FirstAndEither,
        ENTROPY_LANES,
        ENTROPY_THRESHOLD,
    )
}

fn quantified_condition_loops_program(records: u32) -> Program {
    let lane_mask = Expr::u32((1u32 << QUANTIFIED_LANES) - 1);
    let any_hit = Expr::ne(
        Expr::bitand(Expr::var("any_word"), lane_mask.clone()),
        Expr::u32(0),
    );
    let all_hit = Expr::eq(
        Expr::bitand(Expr::var("all_word"), lane_mask.clone()),
        lane_mask.clone(),
    );
    let threshold_hits = Expr::popcount(Expr::bitand(Expr::var("threshold_word"), lane_mask));
    let threshold_hit = Expr::ge(threshold_hits, Expr::u32(QUANTIFIED_THRESHOLD));
    let condition = Expr::and(any_hit, Expr::and(all_hit, threshold_hit));

    Program::wrapped(
        vec![
            BufferDecl::output("out_count", 0, DataType::U32).with_count(1),
            BufferDecl::storage("any_mask", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(records),
            BufferDecl::storage("all_mask", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(records),
            BufferDecl::storage("threshold_mask", 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(records),
            BufferDecl::workgroup("warp_scratch", 1024, DataType::U32),
        ],
        [256, 1, 1],
        warp_reduction_count_nodes(
            256,
            records,
            Expr::and(Expr::var("in_bounds"), condition),
            vec![
                Node::let_bind("any_word", Expr::load("any_mask", Expr::var("safe_idx"))),
                Node::let_bind("all_word", Expr::load("all_mask", Expr::var("safe_idx"))),
                Node::let_bind(
                    "threshold_word",
                    Expr::load("threshold_mask", Expr::var("safe_idx")),
                ),
            ],
        ),
    )
}

fn alias_reaching_def_program(records: u32) -> Program {
    triple_mask_threshold_count_program(
        records,
        ["def_mask", "use_mask", "kill_mask"],
        ["def_word", "use_word", "kill_word"],
        "reaching_aliases",
        TripleMaskPredicate::FirstTwoSetThirdClear,
        ALIAS_LANES,
        ALIAS_THRESHOLD,
    )
}

fn ifds_witness_program(records: u32) -> Program {
    triple_mask_threshold_count_program(
        records,
        ["frontier_mask", "transfer_mask", "witness_mask"],
        ["frontier_word", "transfer_word", "witness_word"],
        "witness_hits",
        TripleMaskPredicate::AllSet,
        IFDS_LANES,
        IFDS_THRESHOLD,
    )
}

fn ast_motif_traversal_program(records: u32) -> Program {
    triple_mask_threshold_count_program(
        records,
        ["node_kind_mask", "depth_mask", "motif_mask"],
        ["node_kind_word", "depth_word", "motif_word"],
        "ast_hits",
        TripleMaskPredicate::AllSet,
        C_AST_LANES,
        C_AST_THRESHOLD,
    )
}

fn megakernel_queue_program(records: u32) -> Program {
    triple_mask_threshold_count_program(
        records,
        ["queue_mask", "predicate_mask", "dispatch_mask"],
        ["queue_word", "predicate_word", "dispatch_word"],
        "queued_hits",
        TripleMaskPredicate::AllSet,
        MEGAKERNEL_QUEUE_LANES,
        MEGAKERNEL_QUEUE_THRESHOLD,
    )
}

fn egraph_saturation_program(records: u32) -> Program {
    triple_mask_threshold_count_program(
        records,
        ["opcode_mask", "lhs_class_mask", "rhs_class_mask"],
        ["opcode_word", "lhs_word", "rhs_word"],
        "rewrite_hits",
        TripleMaskPredicate::AllSet,
        EGRAPH_LANES,
        EGRAPH_THRESHOLD,
    )
}

fn pattern_condition(pattern: SyntheticPattern) -> Expr {
    match pattern {
        SyntheticPattern::ConditionEval => Expr::and(
            Expr::gt(load_u32("match_count"), Expr::u32(3)),
            Expr::and(
                Expr::eq(load_u32("rule_bitmap"), Expr::u32(7)),
                Expr::ne(load_u32("metadata_gate"), Expr::u32(0)),
            ),
        ),
        SyntheticPattern::StringBitmapScatter => Expr::and(
            Expr::ne(load_u32("pattern_bitmap"), Expr::u32(0)),
            Expr::ne(load_u32("rule_bitmap"), Expr::u32(0)),
        ),
        SyntheticPattern::OffsetCountAggregation => Expr::and(
            Expr::gt(load_u32("offset"), Expr::u32(128)),
            Expr::and(
                Expr::gt(load_u32("length"), Expr::u32(4)),
                Expr::gt(load_u32("count"), Expr::u32(1)),
            ),
        ),
        SyntheticPattern::EntropyWindow => Expr::gt(load_u32("entropy_x1000"), Expr::u32(7200)),
        SyntheticPattern::QuantifiedLoops => Expr::and(
            Expr::ne(load_u32("any_hit"), Expr::u32(0)),
            Expr::and(
                Expr::ne(load_u32("all_hit"), Expr::u32(0)),
                Expr::gt(load_u32("n_hit"), Expr::u32(2)),
            ),
        ),
        SyntheticPattern::AliasReachingDef => Expr::and(
            Expr::eq(load_u32("def_id"), load_u32("use_id")),
            Expr::ne(load_u32("alias_mask"), Expr::u32(0)),
        ),
        SyntheticPattern::IfdsWitness => Expr::and(
            Expr::ne(load_u32("frontier"), Expr::u32(0)),
            Expr::eq(load_u32("edge_kind"), Expr::u32(1)),
        ),
        SyntheticPattern::AstMotifTraversal => Expr::and(
            Expr::eq(load_u32("node_kind"), Expr::u32(42)),
            Expr::gt(load_u32("depth"), Expr::u32(3)),
        ),
        SyntheticPattern::MegakernelQueuedBatch => Expr::and(
            Expr::eq(load_u32("queue_state"), Expr::u32(1)),
            Expr::ne(load_u32("predicate"), Expr::u32(0)),
        ),
        SyntheticPattern::EgraphSaturation => Expr::and(
            Expr::eq(load_u32("opcode"), Expr::u32(3)),
            Expr::eq(load_u32("lhs_class"), load_u32("rhs_class")),
        ),
    }
}

fn load_u32(name: &'static str) -> Expr {
    Expr::load(name, Expr::var("idx"))
}

pub(super) fn pattern_buffers(pattern: SyntheticPattern) -> &'static [&'static str] {
    match pattern {
        SyntheticPattern::ConditionEval => &["match_mask", "rule_mask", "metadata_mask"],
        SyntheticPattern::StringBitmapScatter => &["pattern_bitmap", "rule_bitmap"],
        SyntheticPattern::OffsetCountAggregation => &["offset_mask", "length_mask", "count_mask"],
        SyntheticPattern::EntropyWindow => &["byte_class_mask", "transition_mask", "rarity_mask"],
        SyntheticPattern::QuantifiedLoops => &["any_mask", "all_mask", "threshold_mask"],
        SyntheticPattern::AliasReachingDef => &["def_mask", "use_mask", "kill_mask"],
        SyntheticPattern::IfdsWitness => &["frontier_mask", "transfer_mask", "witness_mask"],
        SyntheticPattern::AstMotifTraversal => &["node_kind_mask", "depth_mask", "motif_mask"],
        SyntheticPattern::MegakernelQueuedBatch => {
            &["queue_mask", "predicate_mask", "dispatch_mask"]
        }
        SyntheticPattern::EgraphSaturation => &["opcode_mask", "lhs_class_mask", "rhs_class_mask"],
    }
}

pub(crate) fn warp_reduction_count_nodes(
    workgroup_size: u32,
    records: u32,
    is_match: Expr,
    body_nodes: Vec<Node>,
) -> Vec<Node> {
    let mut nodes = vec![
        Node::let_bind("idx", Expr::gid_x()),
        Node::let_bind("lid", Expr::local_x()),
        Node::let_bind("subgroup_sz", Expr::subgroup_size()),
        Node::let_bind("lane_id", Expr::subgroup_local_id()),
        Node::let_bind(
            "warp_id",
            Expr::div(Expr::var("lid"), Expr::var("subgroup_sz")),
        ),
        Node::let_bind(
            "num_warps",
            Expr::div(Expr::u32(workgroup_size), Expr::var("subgroup_sz")),
        ),
        Node::let_bind("in_bounds", Expr::lt(Expr::var("idx"), Expr::u32(records))),
        Node::let_bind(
            "safe_idx",
            Expr::select(Expr::var("in_bounds"), Expr::var("idx"), Expr::u32(0)),
        ),
    ];
    nodes.extend(body_nodes);
    nodes.extend([
        Node::let_bind(
            "warp_matches",
            Expr::popcount(Expr::subgroup_ballot(is_match)),
        ),
        Node::if_then(
            Expr::eq(Expr::var("lane_id"), Expr::u32(0)),
            vec![Node::store(
                "warp_scratch",
                Expr::var("warp_id"),
                Expr::var("warp_matches"),
            )],
        ),
        Node::barrier(),
        Node::let_bind("lane_partial", Expr::u32(0)),
        Node::if_then(
            Expr::eq(Expr::var("warp_id"), Expr::u32(0)),
            vec![Node::loop_for(
                "round",
                Expr::u32(0),
                Expr::div(
                    Expr::add(
                        Expr::var("num_warps"),
                        Expr::sub(Expr::var("subgroup_sz"), Expr::u32(1)),
                    ),
                    Expr::var("subgroup_sz"),
                ),
                vec![
                    Node::let_bind(
                        "scratch_idx",
                        Expr::add(
                            Expr::var("lane_id"),
                            Expr::mul(Expr::var("round"), Expr::var("subgroup_sz")),
                        ),
                    ),
                    Node::if_then(
                        Expr::lt(Expr::var("scratch_idx"), Expr::var("num_warps")),
                        vec![Node::assign(
                            "lane_partial",
                            Expr::add(
                                Expr::var("lane_partial"),
                                Expr::load("warp_scratch", Expr::var("scratch_idx")),
                            ),
                        )],
                    ),
                ],
            )],
        ),
        Node::let_bind("wg_total", Expr::subgroup_add(Expr::var("lane_partial"))),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("lid"), Expr::u32(0)),
                Expr::gt(Expr::var("wg_total"), Expr::u32(0)),
            ),
            vec![Node::let_bind(
                "_slot",
                Expr::atomic_add("out_count", Expr::u32(0), Expr::var("wg_total")),
            )],
        ),
    ]);
    nodes
}
