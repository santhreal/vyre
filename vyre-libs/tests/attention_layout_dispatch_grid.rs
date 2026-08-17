//! The launch geometry of a layout move comes from the move, not from its
//! buffers.
//!
//! WHY: every layout move guards on `index < count`, and for a gather or a
//! patch that count is the output, which is also the largest declared buffer.
//! A launch geometry inferred from the declared buffers therefore happened to
//! be right, and the paged append is the first move where it is not: a scatter
//! guards on the CHUNK and declares the whole cache as its write buffer, so an
//! inferred launch fires one lane per cache element to move one decoded token
//! and lets the guard discard the rest. That is a cache-sized dispatch for a
//! chunk-sized move on every decode step, which is the cost paging exists to
//! avoid. These contracts hold the grid to the guarded domain, and hold every
//! move the base emits to the workgroup size the grid is computed against, so
//! the two cannot drift into covering different numbers of elements.
//!
//! The index-map variants are read out of the base's own source at run time. A
//! fourth kind of move added to the base fails here until its launch is
//! decided, because a move whose grid nobody chose inherits the inferred one.

mod harness;

use std::collections::BTreeSet;

use vyre_foundation::ir::{DataType, Program};
use vyre_libs::llm::paged_kv::{paged_kv_append, paged_kv_gather, PagedKvCache, PagedKvError};

fn paged_kv_dispatch_grid(spec: &PagedKvCache<'_>, tokens: u32) -> Result<[u32; 3], PagedKvError> {
    let moved = spec
        .sequences
        .checked_mul(spec.heads)
        .and_then(|x| x.checked_mul(tokens))
        .and_then(|x| x.checked_mul(spec.head_dim))
        .ok_or(PagedKvError::ElementCountOverflow)?;
    Ok(attention_layout_dispatch_grid(moved))
}
use vyre_libs::nn::attention::{
    attention_head_to_token, attention_layout_dispatch_grid, attention_token_to_head,
    kv_cache_append, partial_rope, AttentionPermuteSpec, KvCacheAppendSpec,
    ATTENTION_LAYOUT_WORKGROUP_SIZE,
};

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

/// Lanes a grid launches.
fn lanes(grid: [u32; 3]) -> u32 {
    grid[0] * grid[1] * grid[2] * ATTENTION_LAYOUT_WORKGROUP_SIZE[0]
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

/// WHY: the grid is computed against one workgroup size and the program
/// declares another only if someone edits one of the two. The move and the
/// launch would then cover different numbers of elements, which is a silent
/// partial write for a scatter, so both come from the same constant.
#[test]
fn every_layout_move_declares_the_workgroup_size_the_grid_is_computed_against() {
    for (variant, program, _) in moves() {
        assert_eq!(
            program.workgroup_size, ATTENTION_LAYOUT_WORKGROUP_SIZE,
            "Fix: the {variant} move declares a workgroup size the layout dispatch grid does not compute against."
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
/// write buffer, so a launch sized from the declared buffers fires one lane per
/// cache element. The grid must come from the chunk the move actually touches,
/// and it must still cover every element the guard admits, because a grid that
/// under-fires drops writes instead of merely wasting lanes.
#[test]
fn a_scatter_launches_over_the_chunk_it_moves_and_not_over_the_cache() {
    let spec = cache();
    let cache_elements = spec.blocks * spec.heads * spec.block_tokens * spec.head_dim;

    for chunk_tokens in 1..=8u32 {
        let moved = spec.sequences * spec.heads * chunk_tokens * spec.head_dim;
        let grid = paged_kv_dispatch_grid(&spec, chunk_tokens)
            .expect("Fix: the paged move grid must be computable");

        assert!(
            lanes(grid) >= moved,
            "Fix: the grid launches {} lane(s) for {moved} guarded element(s), so the move drops writes.",
            lanes(grid)
        );
        assert!(
            lanes(grid) < moved + ATTENTION_LAYOUT_WORKGROUP_SIZE[0],
            "Fix: the grid launches {} lane(s) for {moved} guarded element(s), which is more than one partial workgroup of waste.",
            lanes(grid)
        );
        assert!(
            lanes(grid) < cache_elements,
            "Fix: the paged append launches {} lane(s) against a {cache_elements}-element cache, which is the inferred geometry the grid exists to replace.",
            lanes(grid)
        );
    }
}

/// WHY: `lane_grid` owns the zero case because a grid of zero groups is a
/// launch a CUDA driver rejects outright, and the layout grid must not
/// reintroduce a second answer to it. The overflow case is the other end: a
/// move whose element count does not fit `u32` indexing has no launchable grid
/// and says so rather than wrapping into a small one.
#[test]
fn the_layout_grid_is_launchable_at_zero_and_refuses_an_unaddressable_move() {
    assert_eq!(
        attention_layout_dispatch_grid(0),
        [1, 1, 1],
        "Fix: a zero-element move must still produce a launchable grid."
    );
    assert_eq!(attention_layout_dispatch_grid(1), [1, 1, 1]);
    assert_eq!(
        attention_layout_dispatch_grid(ATTENTION_LAYOUT_WORKGROUP_SIZE[0]),
        [1, 1, 1]
    );
    assert_eq!(
        attention_layout_dispatch_grid(ATTENTION_LAYOUT_WORKGROUP_SIZE[0] + 1),
        [2, 1, 1]
    );

    let overflow = PagedKvCache {
        heads: u32::MAX,
        ..cache()
    };
    assert_eq!(
        paged_kv_dispatch_grid(&overflow, 4),
        Err(PagedKvError::ElementCountOverflow),
        "Fix: a move whose element count overflows u32 indexing has no grid."
    );
}
