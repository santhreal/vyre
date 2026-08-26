//! The launch domain of a layout move comes from the move, not from its
//! buffers.
//!
//! WHY: every layout move guards on `index < count`, and for a gather or a
//! patch that count is the output, which is also the largest declared buffer.
//! A launch span inferred from the declared buffers therefore happened to be
//! right, and the paged append is the first move where it is not: a scatter
//! guards on the CHUNK and declares the whole cache as its write buffer, so an
//! inferred launch fires one lane per cache element to move one decoded token
//! and lets the guard discard the rest. That is a cache-sized dispatch for a
//! chunk-sized move on every decode step, which is the cost paging exists to
//! avoid.
//!
//! The guard is in the IR, so the compiler reads it. `guarded_logical_span` is
//! the span every move declares, and these contracts hold each move's guard to
//! the element count it was built from. No move publishes a grid for a caller
//! to pass back down.
//!
//! The index-map variants are read out of the base's own source at run time. A
//! fourth kind of move added to the base fails here until its guard is
//! decided, because a move whose domain nobody bounded inherits the inferred
//! one.

mod harness;

use std::collections::BTreeSet;

use vyre_foundation::guarded_logical_span;
use vyre_foundation::ir::{DataType, Program};
use vyre_libs::llm::paged_kv::{paged_kv_append, paged_kv_gather, PagedKvCache, PagedKvError};
use vyre_libs::nn::attention::{
    attention_head_to_token, attention_token_to_head, kv_cache_append, partial_rope,
    AttentionPermuteSpec, KvCacheAppendSpec, ATTENTION_LAYOUT_WORKGROUP_SIZE,
};

/// Elements one paged move touches, or the overflow it cannot address.
fn moved_elements(spec: &PagedKvCache<'_>, tokens: u32) -> Result<u32, PagedKvError> {
    spec.sequences
        .checked_mul(spec.heads)
        .and_then(|product| product.checked_mul(tokens))
        .and_then(|product| product.checked_mul(spec.head_dim))
        .ok_or(PagedKvError::ElementCountOverflow)
}

/// A cache far larger than the chunk a decode step appends: sixty-four blocks
/// of eight tokens against a one-token chunk.
fn cache() -> PagedKvCache<'static> {
    PagedKvCache {
        cache: "cache",
        block_table: "block_table",
        sequences: 2,
        heads: 4,
        blocks: 64,
        block_tokens: 8,
        blocks_per_sequence: 8,
        head_dim: 16,
        dtype: DataType::F32,
    }
}

/// Every `IndexMap` variant the layout base declares, read from its source.
///
/// The variant set is the axis this file judges, and reading it from the base
/// is what makes a new kind of move fail here instead of joining silently.
fn declared_index_map_variants() -> BTreeSet<String> {
    harness::declared_enum_variants(
        &harness::crate_file("src/nn/attention/layout.rs"),
        "enum IndexMap {",
    )
}

/// Every layout move a public entry point emits, by the index map it uses.
fn moves() -> Vec<(&'static str, Program, u32)> {
    let spec = cache();
    let permute = |input, output| AttentionPermuteSpec {
        input,
        output,
        batch: 2,
        heads: 4,
        sequence: 8,
        head_dim: 16,
        dtype: DataType::F32,
    };
    let permute_elements = 2 * 4 * 8 * 16;
    vec![
        (
            "Element",
            attention_head_to_token(permute("head_major", "token_major"))
                .expect("Fix: the head-major permute must build"),
            permute_elements,
        ),
        (
            "Element",
            attention_token_to_head(permute("token_major", "head_major"))
                .expect("Fix: the token-major permute must build"),
            permute_elements,
        ),
        (
            "Element",
            paged_kv_gather(&spec, "window", 4).expect("Fix: the paged gather must build"),
            2 * 4 * 4 * 16,
        ),
        (
            "Element",
            partial_rope("input", "cos", "sin", "output", 4, 8, 16, 8),
            4 * 8 * 16,
        ),
        (
            "Patch",
            kv_cache_append(KvCacheAppendSpec {
                prior: "prior",
                chunk: "chunk",
                next: "next",
                batch: 2,
                heads: 4,
                capacity: 64,
                chunk_len: 1,
                head_dim: 16,
                offset: 3,
                dtype: DataType::F32,
            })
            .expect("Fix: the contiguous cache append must build"),
            2 * 4 * 64 * 16,
        ),
        (
            "Scatter",
            paged_kv_append(&spec, "chunk", 1, 3).expect("Fix: the paged append must build"),
            2 * 4 * 1 * 16,
        ),
    ]
}

/// WHY: the guard bounds the domain and the workgroup size divides it. A move
/// that declares a workgroup size the base does not own, or a guard that does
/// not bound the element count the move was built from, covers a different
/// number of elements than the move touches, which is a silent partial write
/// for a scatter.
#[test]
fn every_layout_move_guards_the_element_count_it_was_built_from() {
    for (variant, program, elements) in moves() {
        assert_eq!(
            program.workgroup_size, ATTENTION_LAYOUT_WORKGROUP_SIZE,
            "Fix: the {variant} move declares a workgroup size the layout base does not own."
        );
        assert_eq!(
            guarded_logical_span(&program),
            Some(elements),
            "Fix: the {variant} move must guard exactly the {elements} element(s) it moves, so the compiler launches over the move and not over its buffers."
        );
    }
}

/// WHY: a kind of move added to the base inherits whatever launch geometry the
/// driver infers, which is what made the paged append cost a cache-sized
/// dispatch. A new variant has to be launched deliberately, so it fails here
/// until one of these cases exercises it.
#[test]
fn every_declared_index_map_variant_has_a_launched_move() {
    let declared = declared_index_map_variants();
    assert!(
        declared.len() >= 3,
        "Fix: only {} index-map variant(s) were derived from the layout base; the declaration no longer matches, so this file would pass by finding nothing.",
        declared.len()
    );
    let exercised = moves()
        .into_iter()
        .map(|(variant, _, _)| variant.to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        declared, exercised,
        "Fix: every index map the layout base declares needs a move here whose launch is judged."
    );
}

/// WHY: the coverage assertion above is worth exactly what its parser sees. A
/// parser keyed on the struct-like form reports a tuple or unit variant as
/// absent, and the set comparison then agrees with itself while the axis it
/// judges has grown. This states the parser against all three variant forms so
/// a blind spot fails here rather than passing there.
#[test]
fn the_variant_parser_reads_every_declaration_form() {
    let source = r"pub enum Shape {
    /// Struct-like.
    Named { field: u32 },
    /// Tuple.
    Positional(u32, String),
    /// Unit.
    Bare,
    #[doc(hidden)]
    Multiline {
        Field: u32,
    },
}
";
    let declared = harness::declared_enum_variants(source, "enum Shape {");
    assert_eq!(
        declared,
        ["Bare", "Multiline", "Named", "Positional"]
            .into_iter()
            .map(str::to_string)
            .collect::<BTreeSet<_>>(),
        "Fix: the variant parser reads one declaration form and calls the others absent."
    );
}

/// WHY: this is the defect. The paged append declares the whole cache as its
/// write buffer, so a launch span taken from the declared buffers covers one
/// lane per cache element. The guard admits only the chunk, and the span the
/// compiler reads must be exactly that: a wider span wastes a decode step's
/// worth of lanes, and a narrower one drops writes.
#[test]
fn a_scatter_guards_the_chunk_it_moves_and_not_the_cache() {
    let spec = cache();
    let cache_elements = spec.blocks * spec.heads * spec.block_tokens * spec.head_dim;

    for chunk_tokens in 1..=8u32 {
        let moved =
            moved_elements(&spec, chunk_tokens).expect("Fix: the paged move must be addressable");
        let program = paged_kv_append(&spec, "chunk", chunk_tokens, 3)
            .expect("Fix: the paged append must build");

        assert_eq!(
            guarded_logical_span(&program),
            Some(moved),
            "Fix: the paged append must guard {moved} element(s) so the compiler launches over the chunk."
        );
        assert!(
            moved < cache_elements,
            "Fix: this case no longer separates the chunk from the {cache_elements}-element cache, so it cannot detect the inferred span it exists to replace."
        );
    }
}

/// WHY: both ends of the domain are refusals. A move of zero tokens addresses
/// no element, so paged addressing rejects it instead of building a program
/// whose launch a driver would have to floor back up to one group. A move whose
/// element count does not fit `u32` indexing has no domain either, and says so
/// rather than wrapping into a small one. The launch floor itself belongs to
/// `admitted_logical_span`, and `vyre-foundation/tests/logical_span_contracts.rs`
/// holds it.
#[test]
fn a_zero_element_move_and_an_unaddressable_one_are_both_refused() {
    let spec = cache();
    assert_eq!(
        paged_kv_append(&spec, "chunk", 0, 3)
            .expect_err("Fix: a move of zero tokens addresses no element and has no domain."),
        PagedKvError::EmptyShape
    );
    assert_eq!(
        paged_kv_gather(&spec, "window", 0)
            .expect_err("Fix: both paged moves share one addressing boundary."),
        PagedKvError::EmptyShape
    );

    let overflow = PagedKvCache {
        heads: u32::MAX,
        ..cache()
    };
    assert_eq!(
        moved_elements(&overflow, 4),
        Err(PagedKvError::ElementCountOverflow),
        "Fix: a move whose element count overflows u32 indexing has no domain."
    );
    assert!(
        paged_kv_append(&overflow, "chunk", 4, 3).is_err(),
        "Fix: an unaddressable move must be refused rather than built with a wrapped count."
    );
}
