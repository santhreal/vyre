//! A sampler builder that takes a count traps on zero instead of emitting a
//! load at `count - 1`.
//!
//! WHY: `nucleus_select` is public, infallible and `#[must_use]`, and with
//! `candidates == 0` its prefix walk kept nothing, fell back to `kept =
//! candidates`, and indexed `kept - 1`, which underflows to `u32::MAX`. The
//! composed path never reached it because `TokenSampler::check` rejects an
//! empty candidate set first, so nothing in the suite saw the direct call. The
//! convention in this crate is a trap program for a degenerate shape, and this
//! pins that convention across the whole sampler path rather than the one
//! builder that was wrong: every count-taking stage a caller can reach directly
//! is checked here, at zero and at the one-element boundary either side of it.
//!
//! What it does not catch: a builder whose degenerate shape is expressed by
//! something other than a zero count, and a trap that is emitted but carries a
//! message naming the wrong parameter.
#![cfg(feature = "llm")]

use vyre_foundation::ir::{Node, Program};
use vyre_libs::llm::sampling::nucleus_select;
use vyre_libs::nn::moe::softmax_top_k;

/// The trap tags a program's entry carries, at any region depth.
fn trap_tags(program: &Program) -> Vec<String> {
    fn walk(node: &Node, out: &mut Vec<String>) {
        if let Node::Trap { tag, .. } = node {
            out.push(tag.as_str().to_string());
        }
        for body in vyre_foundation::visit::child_bodies(node) {
            for child in body {
                walk(child, out);
            }
        }
    }
    let mut out = Vec::new();
    program.entry().iter().for_each(|node| walk(node, &mut out));
    out
}

#[test]
fn an_empty_candidate_set_traps_instead_of_indexing_below_zero() {
    let tags = trap_tags(&nucleus_select("selected", "weights", "uniform", "token", 0, 0.9));
    assert_eq!(
        tags.len(),
        1,
        "nucleus_select with no candidates must build exactly one trap, got {tags:?}"
    );
    assert!(
        tags[0].contains("candidates > 0"),
        "the trap must name the parameter that is wrong, got {:?}",
        tags[0]
    );
}

#[test]
fn a_single_candidate_still_builds_a_real_draw() {
    assert!(
        trap_tags(&nucleus_select("selected", "weights", "uniform", "token", 1, 0.9)).is_empty(),
        "one candidate is the smallest drawable set and must not trap"
    );
}

#[test]
fn the_top_k_stage_holds_the_same_zero_count_contract() {
    assert_eq!(
        trap_tags(&softmax_top_k("logits", "selected", "weights", 8, 0)).len(),
        1,
        "softmax_top_k is the stage that feeds nucleus_select; a zero count must trap there too"
    );
    assert!(
        trap_tags(&softmax_top_k("logits", "selected", "weights", 8, 1)).is_empty(),
        "one candidate must not trap"
    );
}
