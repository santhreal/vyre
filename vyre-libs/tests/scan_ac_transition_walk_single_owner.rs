//! One Aho-Corasick transition walk for the whole crate, proved across entry points.
//!
//! WHY this exists: `scan/classic_ac`, `scan/classic_ac/bounded_ranges`,
//! `scan/classic_ac/bounded_ranges/prefilter`, `scan/classic_ac/count_program`,
//! `scan/regex_region_admission`, `scan/regex_anchored_window` and
//! `scan/fused_region_evidence` all emit the same three pieces of IR: the dense
//! `state = transitions[state * 256 + byte]` step, the flat output-link span
//! `out_begin`/`out_end`, and (for the region-attributed builders) the bounded
//! binary search over `region_starts`. Each one used to spell those out by hand.
//! Hand copies drift one at a time: a row-stride change, an output-link layout
//! change or a region-lookup fix lands in the builder someone was editing and
//! silently misses the rest, and no per-builder parity test can see it because
//! every such test only exercises its own emission.
//!
//! These tests are entry-point-crossing on purpose. They pull the walk
//! substructure back out of the SHIPPED program builders and assert the copies
//! are literally the same IR. They do not check that the walk is correct; the
//! per-builder oracle-parity tests own that. They check that there is one of it.
//!
//! Not caught: a builder that emits no AC walk at all (it contributes nothing to
//! compare), a divergence living entirely in the per-record emission after the
//! walk, and a walk written against differently named loop variables.

#![cfg(all(feature = "matching-regex", feature = "matching-dfa"))]

use std::collections::BTreeMap;

use vyre_foundation::ir::{BinOp, Expr, Node, Program};
use vyre_foundation::visit::for_each_node;
use vyre_libs::scan::classic_ac::{
    build_ac_bounded_count_prefilter_program, build_ac_bounded_count_program,
    build_ac_bounded_count_suffix2_prefilter_program,
    build_ac_bounded_count_suffix3_prefilter_program, build_ac_bounded_ranges_prefilter_program,
    build_ac_bounded_ranges_program, build_ac_bounded_ranges_suffix3_prefilter_program,
    classic_ac_compile, classic_ac_program,
    try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_program,
};
use vyre_libs::scan::{
    aho_corasick, anchored_window_extract_program,
    build_regex_dfa_pipeline_with_policy_and_subgroup_coalesce, fused_region_evidence_program,
    regex_admission_by_region_program, RegexReplayPolicy,
};
use vyre_libs::matching::CompiledDfa;

const PATTERNS: [&[u8]; 4] = [b"alpha", b"beta", b"gamma", b"al"];
const PATTERN_COUNT: u32 = 4;
const MAX_REGIONS: u32 = 8;
const MAX_MATCHES: u32 = 64;
const MAX_PATTERN_LEN: u32 = 8;
const LOG2_MAX_REGIONS: u32 = 3;

fn dfa() -> CompiledDfa {
    classic_ac_compile(&PATTERNS).dfa
}

fn state_count(dfa: &CompiledDfa) -> u32 {
    (dfa.output_offsets.len() - 1) as u32
}

fn output_records_len(dfa: &CompiledDfa) -> u32 {
    dfa.output_records.len() as u32
}

/// Every packed-haystack AC program the crate ships, keyed by the builder that
/// produced it so a failure names the drifted call site instead of an index.
fn packed_walk_programs(dfa: &CompiledDfa) -> BTreeMap<&'static str, Program> {
    let presence_words = PATTERN_COUNT.div_ceil(32).max(1);
    let mut programs = BTreeMap::new();
    programs.insert(
        "build_ac_bounded_ranges_program",
        build_ac_bounded_ranges_program(dfa, PATTERN_COUNT, MAX_MATCHES),
    );
    programs.insert(
        "build_ac_bounded_ranges_prefilter_program",
        build_ac_bounded_ranges_prefilter_program(dfa, PATTERN_COUNT, MAX_MATCHES),
    );
    programs.insert(
        "build_ac_bounded_ranges_suffix3_prefilter_program",
        build_ac_bounded_ranges_suffix3_prefilter_program(dfa, PATTERN_COUNT, MAX_MATCHES),
    );
    programs.insert(
        "try_build_ac_bounded_ranges_suffix3_presence_program",
        try_build_ac_bounded_ranges_suffix3_presence_program(dfa, PATTERN_COUNT)
            .expect("presence program fits the u32 buffer-count ABI"),
    );
    programs.insert(
        "try_build_ac_bounded_ranges_suffix3_presence_by_region_program",
        try_build_ac_bounded_ranges_suffix3_presence_by_region_program(
            dfa,
            PATTERN_COUNT,
            MAX_REGIONS,
        )
        .expect("region presence program fits the u32 buffer-count ABI"),
    );
    programs.insert(
        "try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program",
        try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program(
            dfa,
            PATTERN_COUNT,
            MAX_REGIONS,
            MAX_MATCHES,
        )
        .expect("fused region program fits the u32 buffer-count ABI"),
    );
    programs.insert(
        "build_ac_bounded_count_program",
        build_ac_bounded_count_program(dfa),
    );
    programs.insert(
        "build_ac_bounded_count_prefilter_program",
        build_ac_bounded_count_prefilter_program(dfa),
    );
    programs.insert(
        "build_ac_bounded_count_suffix2_prefilter_program",
        build_ac_bounded_count_suffix2_prefilter_program(dfa),
    );
    programs.insert(
        "build_ac_bounded_count_suffix3_prefilter_program",
        build_ac_bounded_count_suffix3_prefilter_program(dfa),
    );
    programs.insert(
        "regex_admission_by_region_program",
        regex_admission_by_region_program(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "region_starts",
            "region_base",
            "haystack_len",
            "presence",
            state_count(dfa),
            output_records_len(dfa),
            MAX_REGIONS,
            presence_words,
            MAX_PATTERN_LEN,
            LOG2_MAX_REGIONS,
        ),
    );
    programs.insert(
        "anchored_window_extract_program",
        anchored_window_extract_program(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "candidates",
            "candidate_count",
            "haystack_len",
            "match_count",
            "matches",
            state_count(dfa),
            output_records_len(dfa),
            32,
            MAX_MATCHES,
            MAX_PATTERN_LEN,
        ),
    );
    programs.insert(
        "fused_region_evidence_program",
        fused_region_evidence_program(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "region_starts",
            "region_base",
            "position_mask",
            "admission_mask",
            "haystack_len",
            "presence",
            "match_count",
            "matches",
            "admission",
            state_count(dfa),
            output_records_len(dfa),
            MAX_REGIONS,
            PATTERN_COUNT,
            presence_words,
            MAX_MATCHES,
            MAX_PATTERN_LEN,
            LOG2_MAX_REGIONS,
        ),
    );
    programs.insert(
        "regex_exact_ranges_program (via build_regex_dfa_pipeline_with_policy_and_subgroup_coalesce)",
        build_regex_dfa_pipeline_with_policy_and_subgroup_coalesce(
            &["alpha", "beta", "gamma", "al"],
            MAX_MATCHES,
            4096,
            RegexReplayPolicy::default(),
            false,
        )
        .expect("the regex pipeline compiles this literal set")
        .program,
    );
    programs
}

/// The two unpacked-haystack walks: one AC byte per invocation loaded straight
/// out of an unpacked buffer, no u32 word to peel.
fn unbounded_programs(dfa: &CompiledDfa) -> BTreeMap<&'static str, Program> {
    let mut programs = BTreeMap::new();
    programs.insert("classic_ac_program", unbounded_classic_program(dfa));
    programs.insert(
        "aho_corasick",
        aho_corasick(
            "haystack",
            "transitions",
            "accept",
            "matches",
            256,
            state_count(dfa),
        ),
    );
    programs
}

fn unbounded_classic_program(dfa: &CompiledDfa) -> Program {
    classic_ac_program(
        "haystack",
        "transitions",
        "output_offsets",
        "output_records",
        "match_count",
        "matches",
        256,
        state_count(dfa),
        output_records_len(dfa),
        MAX_MATCHES,
    )
}

/// Depth-first statement walk over a program body, in emission order.
///
/// Descent is `for_each_node`, the crate-wide exhaustive traversal owner. The
/// hand-rolled walk this replaces enumerated the nesting variants itself and
/// ended in `_ => {}`, which is the failure mode this whole file exists to
/// prevent: a `Node` variant that gains a body is classified as a leaf, the walk
/// silently stops returning the transition step and output-link span nested
/// inside it, and the comparisons below go vacuous on the builders that use it
/// rather than red.
fn all_nodes(program: &Program) -> Vec<&Node> {
    let mut out = Vec::new();
    for_each_node(program.entry(), |node| out.push(node));
    out
}

/// Canonical rendering of a construct within one program: every distinct
/// spelling it emits, sorted and joined. One walk means one line.
fn canonical(mut renderings: Vec<String>) -> Option<String> {
    renderings.sort_unstable();
    renderings.dedup();
    if renderings.is_empty() {
        None
    } else {
        Some(renderings.join("\n"))
    }
}

/// The `state = ...` assignment, the AC transition step itself.
fn transition_step(program: &Program) -> Option<String> {
    canonical(
        all_nodes(program)
            .into_iter()
            .filter_map(|node| match node {
                Node::Assign { name, value } if *name == "state" => Some(format!("{value:?}")),
                _ => None,
            })
            .collect(),
    )
}

/// The `let out_begin` / `let out_end` pair that reads the flat output links.
fn output_link_span(program: &Program) -> Option<String> {
    canonical(
        all_nodes(program)
            .into_iter()
            .filter_map(|node| match node {
                Node::Let { name, value } if *name == "out_begin" || *name == "out_end" => {
                    Some(format!("{}={value:?}", &**name))
                }
                _ => None,
            })
            .collect(),
    )
}

/// The bounded `region_starts` binary search plus the `rs_base` row offset it
/// produces.
fn region_search(program: &Program) -> Option<String> {
    canonical(
        all_nodes(program)
            .into_iter()
            .filter_map(|node| match node {
                Node::Loop { var, body, .. } if *var == "rs_step" => {
                    Some(format!("rs_step={body:?}"))
                }
                Node::Let { name, value } if *name == "rs_base" => {
                    Some(format!("rs_base={value:?}"))
                }
                _ => None,
            })
            .collect(),
    )
}

/// Group builders by the rendering they produced, so a failure lists the
/// factions rather than a diff of two arbitrary programs.
fn factions(
    programs: &BTreeMap<&'static str, Program>,
    extract: fn(&Program) -> Option<String>,
) -> BTreeMap<String, Vec<&'static str>> {
    let mut grouped: BTreeMap<String, Vec<&'static str>> = BTreeMap::new();
    for (name, program) in programs {
        if let Some(rendering) = extract(program) {
            grouped.entry(rendering).or_default().push(*name);
        }
    }
    grouped
}

fn contributors(grouped: &BTreeMap<String, Vec<&'static str>>) -> usize {
    grouped.values().map(Vec::len).sum()
}

#[test]
fn every_packed_ac_builder_emits_the_same_transition_step() {
    let dfa = dfa();
    let programs = packed_walk_programs(&dfa);
    let grouped = factions(&programs, transition_step);
    assert_eq!(
        contributors(&grouped),
        programs.len(),
        "every packed AC builder must emit a transition step; contributors: {:#?}",
        grouped.values().collect::<Vec<_>>(),
    );
    assert_eq!(
        grouped.len(),
        1,
        "the crate must hold exactly ONE AC transition walk; found {} spellings across \
         these builder groups: {:#?}",
        grouped.len(),
        grouped.values().collect::<Vec<_>>(),
    );
}

#[test]
fn every_ac_builder_emits_the_same_output_link_span() {
    let dfa = dfa();
    let mut programs = packed_walk_programs(&dfa);
    // The unbounded classic walk loads unpacked bytes, so it sits out the
    // transition-step comparison, but its output-link span must be the same IR.
    programs.insert("classic_ac_program", unbounded_classic_program(&dfa));
    let grouped = factions(&programs, output_link_span);
    assert_eq!(
        contributors(&grouped),
        programs.len(),
        "every AC builder must read the flat output links; contributors: {:#?}",
        grouped.values().collect::<Vec<_>>(),
    );
    assert_eq!(
        grouped.len(),
        1,
        "the flat output-link span must have ONE spelling; found {} across these \
         builder groups: {:#?}",
        grouped.len(),
        grouped.values().collect::<Vec<_>>(),
    );
}

/// Peel `transitions[state * 256 + byte]` into the table name, the dense row
/// stride, and the byte operand.
fn transition_shape(value: &Expr) -> Option<(String, u32, String)> {
    let Expr::Load { buffer, index } = value else {
        return None;
    };
    let Expr::BinOp {
        op: BinOp::Add,
        left,
        right,
    } = &**index
    else {
        return None;
    };
    let Expr::BinOp {
        op: BinOp::Mul,
        left: row,
        right: stride,
    } = &**left
    else {
        return None;
    };
    let Expr::Var(state) = &**row else {
        return None;
    };
    let Expr::LitU32(stride) = &**stride else {
        return None;
    };
    if &**state != "state" {
        return None;
    }
    Some((buffer.to_string(), *stride, format!("{right:?}")))
}

#[test]
fn unpacked_walks_share_the_transition_shape() {
    let dfa = dfa();
    let packed = build_ac_bounded_ranges_program(&dfa, PATTERN_COUNT, MAX_MATCHES);
    let packed_step = all_nodes(&packed)
        .into_iter()
        .find_map(|node| match node {
            Node::Assign { name, value } if *name == "state" => transition_shape(value),
            _ => None,
        })
        .expect("the bounded walk emits a `state = transitions[...]` step");

    let unpacked = unbounded_programs(&dfa);
    let mut shapes = BTreeMap::new();
    for (name, program) in &unpacked {
        let step = all_nodes(program)
            .into_iter()
            .find_map(|node| match node {
                Node::Assign { name, value } if *name == "state" => transition_shape(value),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{name} must emit a `state = transitions[...]` step"));

        // An unpacked walk differs from the bounded family in exactly ONE way:
        // it indexes the haystack buffer directly instead of unpacking a u32
        // word. Everything else, the table it reads and the `state * 256 + byte`
        // row arithmetic, must match or it is a second transition walk in
        // disguise.
        assert_eq!(
            (step.0.as_str(), step.1),
            (packed_step.0.as_str(), packed_step.1),
            "{name} must read the same transition table through the same dense row \
             stride as the bounded family",
        );
        assert_ne!(
            step.2, packed_step.2,
            "{name} is an unpacked-haystack variant; if its byte operand now matches \
             the packed one, the two walks have merged and it belongs in the \
             transition-step equality test instead",
        );
        shapes.insert(step.2, *name);
    }

    assert_eq!(
        unpacked.len(),
        2,
        "both unpacked walks must be under test: {:?}",
        unpacked.keys().collect::<Vec<_>>(),
    );
    assert_eq!(
        shapes.len(),
        1,
        "the unpacked walks must spell their byte load the same way; found {shapes:#?}",
    );
}

#[test]
fn every_region_attributed_builder_emits_the_same_region_search() {
    let dfa = dfa();
    let programs = packed_walk_programs(&dfa);
    let grouped = factions(&programs, region_search);
    assert!(
        contributors(&grouped) >= 3,
        "expected the region-attributed builders to emit a region search, got {:#?}",
        grouped.values().collect::<Vec<_>>(),
    );
    assert_eq!(
        grouped.len(),
        1,
        "the `region_starts` binary search must have ONE spelling; found {} across \
         these builder groups: {:#?}",
        grouped.len(),
        grouped.values().collect::<Vec<_>>(),
    );
}

#[test]
fn region_row_stride_is_never_zero() {
    // `presence_bitmap_words`, `presence_by_region_words` and
    // `regex_admission_presence_words` all floor the per-region row width at one
    // word, so the row offset `region * presence_words` is floored too. A zero
    // stride aliases every region onto row 0 and reports one batch-wide bitmap
    // as if it were a per-region one.
    let dfa = dfa();
    let program = regex_admission_by_region_program(
        "haystack",
        "transitions",
        "output_offsets",
        "output_records",
        "region_starts",
        "region_base",
        "haystack_len",
        "presence",
        state_count(&dfa),
        output_records_len(&dfa),
        MAX_REGIONS,
        0,
        MAX_PATTERN_LEN,
        LOG2_MAX_REGIONS,
    );
    let rs_base = all_nodes(&program)
        .into_iter()
        .find_map(|node| match node {
            Node::Let { name, value } if *name == "rs_base" => Some(format!("{value:?}")),
            _ => None,
        })
        .expect("the region program binds a presence-row offset");
    assert!(
        !rs_base.contains("LitU32(0)"),
        "a degenerate presence_words must floor to a one-word row stride, not zero: {rs_base}"
    );
}

#[test]
fn walk_extraction_sees_the_constructs_it_claims_to_compare() {
    // A guard that silently extracts nothing passes forever. Pin the extractors
    // against a hand-built program holding one of each construct.
    let program = Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![Node::loop_for(
            "step",
            Expr::u32(0),
            Expr::u32(1),
            vec![
                Node::assign(
                    "state",
                    Expr::load(
                        "transitions",
                        Expr::add(
                            Expr::mul(Expr::var("state"), Expr::u32(256)),
                            Expr::var("byte"),
                        ),
                    ),
                ),
                Node::let_bind(
                    "out_begin",
                    Expr::load("output_offsets", Expr::var("state")),
                ),
                Node::let_bind("out_end", Expr::load("output_offsets", Expr::var("state"))),
                Node::loop_for(
                    "rs_step",
                    Expr::u32(0),
                    Expr::u32(1),
                    vec![Node::assign("rs_lo", Expr::u32(0))],
                ),
                Node::let_bind("rs_base", Expr::mul(Expr::var("rs_lo"), Expr::u32(2))),
            ],
        )],
    );
    assert!(transition_step(&program).is_some());
    assert_eq!(output_link_span(&program).expect("span").lines().count(), 2);
    assert_eq!(region_search(&program).expect("search").lines().count(), 2);

    let step = all_nodes(&program)
        .into_iter()
        .find_map(|node| match node {
            Node::Assign { name, value } if *name == "state" => transition_shape(value),
            _ => None,
        })
        .expect("shape");
    assert_eq!(step.0, "transitions");
    assert_eq!(step.1, 256);
}
