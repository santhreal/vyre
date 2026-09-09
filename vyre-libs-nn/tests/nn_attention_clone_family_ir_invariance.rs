//! IR invariance across the `nn/attention` clone families.
//!
//! Families of duplicated builder code were collapsed onto single owners: the
//! gated delta index/shape math, the online-softmax core, the reduce-then-
//! normalize skeleton, the three-pass score/sum/write owner, and the layout
//! index-map owner. Collapsing a clone family is only safe if the surviving
//! owner emits exactly what every former copy emitted, so this file pins the
//! structural IR of every entry point involved.
//!
//! # Three rules, and what each one alone cannot see
//!
//! `clone_family_entry_points_emit_the_pinned_ir` compares each entry point's
//! canonicalized buffer roster and node tree against a checked-in golden. It
//! sees a changed operand, a dropped node and a reordered data dependence, and
//! it reports them as a text diff naming the node that moved. It does not see a
//! shared owner that stopped being reached while emitting equivalent work.
//!
//! `clone_family_entry_points_carry_the_pinned_region_identities` answers that
//! by name: it pins which shared child regions each entry point embeds.
//!
//! `mla_and_flash_attention_2_share_the_online_softmax_skeleton` compares the
//! two tiled decoders node for node, so neither can reacquire a private copy of
//! the recurrence while both goldens move together.
//!
//! # Why the pin is structural IR and not `Program::fingerprint`
//!
//! This file pinned BLAKE3 over `canonical_wire_bytes` for 26 entry points. Wire
//! bytes open with `WIRE_FORMAT_VERSION`, so every serialization revision moved
//! all 26 digests at once while no program's meaning moved, and each time the
//! table was re-pinned by hand from the failure report. A guard that goes red on
//! a relabelling, reports the difference as 32 opaque bytes, and is answered by
//! copying the measured numbers back in certifies nothing about the IR. The
//! rendering is a function of the IR model now, and `harness::structural_ir`
//! states what that covers and what it deliberately drops.
//!
//! # Closure
//!
//! The roster is an enum whose `build` and `id` matches have no catch-all arm,
//! so a 27th member is a compile error until someone builds it and names it.
//! `the_roster_names_every_declared_entry_point` holds the enum's own
//! declaration equal to `ALL`, and
//! `every_public_attention_builder_has_a_recorded_decision` reads the module's
//! re-export list from source and refuses to pass until a new public builder is
//! either rostered or recorded as not a clone-family member.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use crate::harness;
use harness::structural_ir::{
    assert_matches_golden, golden_contains, render_golden, render_section, render_structural_ir,
    write_golden,
};
use vyre_foundation::ir::{DataType, Node, Program};
use vyre_libs_nn::nn::attention::{
    attention, attention_head_to_token, attention_reference, attention_token_to_head,
    chunked_gated_delta, flash_attention, flash_attention_2, gqa_attention, gqa_attention_causal,
    gqa_attention_causal_typed, kv_cache_append, mla_compress_kv, mla_decode, partial_rope,
    qk_gain, quest_paging, recurrent_gated_delta, softmax, turboquant_attention,
    AttentionPermuteSpec, GatedDeltaSpec, KvCacheAppendSpec,
};
use vyre_libs_nn::nn::norm::layer_norm;

/// Sequence length used by every tiled fixture. Deliberately not a multiple of
/// the 64-wide tile so the ragged final tile is part of the pinned IR.
const SEQ_LEN: u32 = 130;
const HEAD_DIM: u32 = 8;
const TILE_SIZE: u32 = 64;

/// Sequence length for the gated delta fixtures, likewise ragged against the
/// chunked schedule's fixed 64-token chunk.
const DELTA_SEQ: u32 = 70;

fn gated_delta_fixture(
    build: fn(
        &GatedDeltaSpec<'_>,
    ) -> Result<Program, vyre_libs_nn::nn::attention::RecurrentGatedDeltaError>,
    dtype: DataType,
) -> Program {
    build(&GatedDeltaSpec {
        query: "query",
        key: "key",
        value: "value",
        decay_log: "decay_log",
        beta_logits: "beta_logits",
        state_input: "state_in",
        output: "out",
        state_output: "state_out",
        batch: 2,
        sequence: DELTA_SEQ,
        key_heads: 2,
        value_heads: 4,
        key_dim: 3,
        value_dim: 5,
        eps: 1e-5,
        dtype,
    })
    .expect("gated delta fixture builds")
}

fn mla_fixture() -> Program {
    mla_decode(
        "q", "kv_cache", "kr_cache", "w_uk", "w_uv", "out", SEQ_LEN, 3, HEAD_DIM, 4, 4,
    )
    .expect("mla fixture builds")
}

fn flash_fixture() -> Program {
    flash_attention_2("q", "k", "v", "out", SEQ_LEN, HEAD_DIM, TILE_SIZE)
}

/// The layout-move fixture shape, ragged in every axis so a transposed index
/// derivation cannot produce the same IR.
fn permute_spec(dtype: DataType) -> AttentionPermuteSpec<'static> {
    AttentionPermuteSpec {
        input: "input",
        output: "output",
        batch: 2,
        heads: 3,
        sequence: 5,
        head_dim: 4,
        dtype,
    }
}

fn cache_spec(dtype: DataType) -> KvCacheAppendSpec<'static> {
    KvCacheAppendSpec {
        prior: "prior",
        chunk: "chunk",
        next: "next",
        batch: 2,
        heads: 2,
        capacity: 8,
        chunk_len: 3,
        head_dim: 4,
        offset: 2,
        dtype,
    }
}

/// One clone-family entry point.
///
/// The `build` and `id` matches below have no catch-all arm, so adding a
/// variant is a compile error in this file until the new member has a fixture
/// and a name. That is the point at which someone also has to bless a golden
/// section for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CloneFamilyEntry {
    RecurrentGatedDeltaF32,
    RecurrentGatedDeltaF16,
    ChunkedGatedDeltaF32,
    ChunkedGatedDeltaF16,
    MlaDecode,
    FlashAttention2,
    Softmax,
    LayerNorm,
    FlashAttention,
    FlashAttentionDirect,
    Attention,
    AttentionDirect,
    AttentionReference,
    GqaAttention,
    GqaAttentionCausal,
    GqaAttentionCausalF16,
    KvCacheAppend,
    KvCacheAppendF16,
    AttentionHeadToToken,
    AttentionHeadToTokenF16,
    AttentionTokenToHead,
    QuestPaging,
    PartialRope,
    QkGain,
    TurboquantAttention,
    MlaCompressKv,
}

impl CloneFamilyEntry {
    /// The roster, in golden order.
    ///
    /// A const array cannot be exhaustive on its own;
    /// `the_roster_names_every_declared_entry_point` holds it equal to the enum
    /// declaration this file carries.
    const ALL: [Self; 26] = [
        Self::RecurrentGatedDeltaF32,
        Self::RecurrentGatedDeltaF16,
        Self::ChunkedGatedDeltaF32,
        Self::ChunkedGatedDeltaF16,
        Self::MlaDecode,
        Self::FlashAttention2,
        Self::Softmax,
        Self::LayerNorm,
        Self::FlashAttention,
        Self::FlashAttentionDirect,
        Self::Attention,
        Self::AttentionDirect,
        Self::AttentionReference,
        Self::GqaAttention,
        Self::GqaAttentionCausal,
        Self::GqaAttentionCausalF16,
        Self::KvCacheAppend,
        Self::KvCacheAppendF16,
        Self::AttentionHeadToToken,
        Self::AttentionHeadToTokenF16,
        Self::AttentionTokenToHead,
        Self::QuestPaging,
        Self::PartialRope,
        Self::QkGain,
        Self::TurboquantAttention,
        Self::MlaCompressKv,
    ];

    /// Golden section name.
    fn id(self) -> &'static str {
        match self {
            Self::RecurrentGatedDeltaF32 => "recurrent_gated_delta/f32",
            Self::RecurrentGatedDeltaF16 => "recurrent_gated_delta/f16",
            Self::ChunkedGatedDeltaF32 => "chunked_gated_delta/f32",
            Self::ChunkedGatedDeltaF16 => "chunked_gated_delta/f16",
            Self::MlaDecode => "mla_decode",
            Self::FlashAttention2 => "flash_attention_2",
            Self::Softmax => "softmax",
            Self::LayerNorm => "layer_norm",
            Self::FlashAttention => "flash_attention",
            Self::FlashAttentionDirect => "flash_attention/direct",
            Self::Attention => "attention",
            Self::AttentionDirect => "attention/direct",
            Self::AttentionReference => "attention_reference",
            Self::GqaAttention => "gqa_attention",
            Self::GqaAttentionCausal => "gqa_attention_causal",
            Self::GqaAttentionCausalF16 => "gqa_attention_causal/f16",
            Self::KvCacheAppend => "kv_cache_append",
            Self::KvCacheAppendF16 => "kv_cache_append/f16",
            Self::AttentionHeadToToken => "attention_head_to_token",
            Self::AttentionHeadToTokenF16 => "attention_head_to_token/f16",
            Self::AttentionTokenToHead => "attention_token_to_head",
            Self::QuestPaging => "quest_paging",
            Self::PartialRope => "partial_rope",
            Self::QkGain => "qk_gain",
            Self::TurboquantAttention => "turboquant_attention",
            Self::MlaCompressKv => "mla_compress_kv",
        }
    }

    /// The public builder this entry point exercises.
    ///
    /// `every_public_attention_builder_has_a_recorded_decision` reads the
    /// module's re-export list and requires each exported builder to appear
    /// here or in [`NOT_A_CLONE_FAMILY_MEMBER`].
    fn builder(self) -> &'static str {
        match self {
            Self::RecurrentGatedDeltaF32 | Self::RecurrentGatedDeltaF16 => {
                "recurrent_gated_delta"
            }
            Self::ChunkedGatedDeltaF32 | Self::ChunkedGatedDeltaF16 => "chunked_gated_delta",
            Self::MlaDecode => "mla_decode",
            Self::FlashAttention2 => "flash_attention_2",
            Self::Softmax => "softmax",
            Self::LayerNorm => "layer_norm",
            Self::FlashAttention | Self::FlashAttentionDirect => "flash_attention",
            Self::Attention | Self::AttentionDirect => "attention",
            Self::AttentionReference => "attention_reference",
            Self::GqaAttention => "gqa_attention",
            Self::GqaAttentionCausal => "gqa_attention_causal",
            Self::GqaAttentionCausalF16 => "gqa_attention_causal_typed",
            Self::KvCacheAppend | Self::KvCacheAppendF16 => "kv_cache_append",
            Self::AttentionHeadToToken | Self::AttentionHeadToTokenF16 => {
                "attention_head_to_token"
            }
            Self::AttentionTokenToHead => "attention_token_to_head",
            Self::QuestPaging => "quest_paging",
            Self::PartialRope => "partial_rope",
            Self::QkGain => "qk_gain",
            Self::TurboquantAttention => "turboquant_attention",
            Self::MlaCompressKv => "mla_compress_kv",
        }
    }

    fn build(self) -> Program {
        match self {
            Self::RecurrentGatedDeltaF32 => {
                gated_delta_fixture(recurrent_gated_delta, DataType::F32)
            }
            Self::RecurrentGatedDeltaF16 => {
                gated_delta_fixture(recurrent_gated_delta, DataType::F16)
            }
            Self::ChunkedGatedDeltaF32 => gated_delta_fixture(chunked_gated_delta, DataType::F32),
            Self::ChunkedGatedDeltaF16 => gated_delta_fixture(chunked_gated_delta, DataType::F16),
            Self::MlaDecode => mla_fixture(),
            Self::FlashAttention2 => flash_fixture(),
            Self::Softmax => softmax("input", "output", 1000),
            Self::LayerNorm => layer_norm("input", "output", 1000, 1e-5),
            Self::FlashAttention => {
                flash_attention("q", "k", "v", "out", SEQ_LEN, HEAD_DIM).expect("flash builds")
            }
            Self::FlashAttentionDirect => {
                flash_attention("q", "k", "v", "out", 4, 4).expect("direct flash builds")
            }
            Self::Attention => attention("q", "k", "v", "out", SEQ_LEN, HEAD_DIM),
            Self::AttentionDirect => attention("q", "k", "v", "out", 4, 4),
            Self::AttentionReference => attention_reference("q", "k", "v", "out", 8, 4),
            Self::GqaAttention => {
                gqa_attention("q", "k", "v", "out", 4, 2, 8, 4).expect("gqa builds")
            }
            Self::GqaAttentionCausal => {
                gqa_attention_causal("q", "k", "v", "out", 2, 4, 2, 3, 8, 4, 2)
                    .expect("causal gqa builds")
            }
            Self::GqaAttentionCausalF16 => gqa_attention_causal_typed(
                "q",
                "k",
                "v",
                "out",
                2,
                4,
                2,
                3,
                8,
                4,
                2,
                DataType::F16,
            )
            .expect("typed causal gqa builds"),
            Self::KvCacheAppend => {
                kv_cache_append(cache_spec(DataType::F32)).expect("cache builds")
            }
            Self::KvCacheAppendF16 => {
                kv_cache_append(cache_spec(DataType::F16)).expect("typed cache builds")
            }
            Self::AttentionHeadToToken => {
                attention_head_to_token(permute_spec(DataType::F32)).expect("head to token builds")
            }
            Self::AttentionHeadToTokenF16 => attention_head_to_token(permute_spec(DataType::F16))
                .expect("typed head to token builds"),
            Self::AttentionTokenToHead => {
                attention_token_to_head(permute_spec(DataType::F32)).expect("token to head builds")
            }
            Self::QuestPaging => quest_paging("q", "meta", "scores", "io", 8, 3, 4),
            Self::PartialRope => partial_rope("input", "cos", "sin", "output", 2, 5, 8, 4),
            Self::QkGain => qk_gain("q_in", "q_out", "gain", 3, 5, 4),
            Self::TurboquantAttention => turboquant_attention("q", "k_packed", "v_packed", "out", 6, 4),
            Self::MlaCompressKv => {
                mla_compress_kv("h", "w_dk", "c_out", 6, 4).expect("mla compress builds")
            }
        }
    }
}

fn entry_points() -> Vec<(&'static str, Program)> {
    CloneFamilyEntry::ALL
        .iter()
        .map(|entry| (entry.id(), entry.build()))
        .collect()
}

/// Path of the structural IR golden.
fn golden_path() -> PathBuf {
    harness::crate_dir().join("tests/golden/nn_attention_clone_family_ir.txt")
}

/// The roster's structural IR, rendered in golden order.
fn render_corpus() -> String {
    render_golden(
        CloneFamilyEntry::ALL
            .iter()
            .map(|entry| (entry.id(), render_section(&entry.build()))),
    )
}

/// Public `nn::attention` builders that are deliberately not clone-family entry
/// points, each with the reason.
///
/// A public builder in this module is either a member of a collapsed family, in
/// which case its IR belongs in the golden, or it is not, in which case someone
/// has to say why. An export with neither turns
/// `every_public_attention_builder_has_a_recorded_decision` red.
const NOT_A_CLONE_FAMILY_MEMBER: [(&str, &str); 9] = [
    (
        "fused_tile_attention",
        "tile-dialect builder: emits tile nodes and shares no collapsed helper \
         with the scalar families",
    ),
    (
        "paged_attention",
        "the paged family owns its own three-pass builder rather than the \
         collapsed score/sum/write owner",
    ),
    (
        "paged_cache_append",
        "paged cache layout, not the kv_cache_append index-map owner",
    ),
    (
        "partial_rope_at_offset",
        "offset wrapper over the partial_rope owner the roster builds",
    ),
    (
        "partial_rope_at_offset_typed",
        "dtype wrapper over the partial_rope owner the roster builds",
    ),
    (
        "plan_flash_attention_scalar",
        "returns a work plan, not a Program",
    ),
    (
        "plan_flash_attention_tiled",
        "returns a work plan, not a Program",
    ),
    (
        "try_attention_reference",
        "fallible form of the rostered attention_reference, same owner",
    ),
    (
        "softmax_reference",
        "scalar reference path; the rostered softmax builds the tiled path that \
         carries the collapsed reduce owners",
    ),
];

/// The structural IR of every clone-family entry point, against the golden.
///
/// This is the rule a shared-helper edit turns red. It carries node kinds, field
/// names, operand expressions, literal values, identifier text, region
/// generators and nesting, after canonicalization, so a changed operand, a
/// dropped node or a reordered data dependence moves it and a buffer-table
/// reorder or a commutative operand swap does not.
#[test]
fn clone_family_entry_points_emit_the_pinned_ir() {
    assert_matches_golden(&golden_path(), &render_corpus());
}

/// A golden that no longer names an entry point silently stopped covering it.
#[test]
fn the_golden_names_every_roster_entry_point() {
    let golden = std::fs::read_to_string(golden_path()).expect("structural IR golden must exist");
    for entry in CloneFamilyEntry::ALL {
        assert!(
            golden_contains(&golden, entry.id()),
            "Fix: the structural IR golden is missing `{}`; re-bless it.",
            entry.id()
        );
    }
}

/// Every declared roster variant must be in `ALL`.
///
/// `ALL` is a const array, so it cannot be exhaustive by itself: a variant added
/// to the enum and given `build` and `id` arms would still be absent from the
/// golden with nothing red. The enum declaration is read from this file's own
/// source, which is the same closure the workspace uses for `Node` and `Expr`
/// variant coverage.
#[test]
fn the_roster_names_every_declared_entry_point() {
    let source = harness::crate_file("tests/nn_attention_clone_family_ir_invariance.rs");
    let declared = harness::declared_enum_variants(&source, "enum CloneFamilyEntry {");
    assert_eq!(
        declared.len(),
        CloneFamilyEntry::ALL.len(),
        "the roster declares {} entry points and ALL names {}; add the new \
         variant to ALL and bless its golden section",
        declared.len(),
        CloneFamilyEntry::ALL.len()
    );
    let named: std::collections::BTreeSet<String> = CloneFamilyEntry::ALL
        .iter()
        .map(|entry| format!("{entry:?}"))
        .collect();
    assert_eq!(
        declared, named,
        "the roster declaration and ALL name different entry points"
    );
}

/// Every public builder `nn::attention` re-exports is either rostered or
/// recorded as not a clone-family member.
///
/// The export list is read from `nn/attention/mod.rs` at run time, so a 27th
/// public builder turns this red until a decision exists for it. A hand-typed
/// list of covered builders would go stale in silence instead, which is the
/// same failure as having no coverage rule.
///
/// `layer_norm` is rostered and lives in `nn::norm`, so it has no export in this
/// module; the check runs one way, from exports to decisions.
#[test]
fn every_public_attention_builder_has_a_recorded_decision() {
    let source = harness::crate_file("src/nn/attention/mod.rs");
    let exported = reexported_builder_names(&source);
    assert!(
        exported.len() >= 20,
        "read only {} public builders out of nn/attention/mod.rs; the \
         re-export parser no longer matches the module",
        exported.len()
    );

    let rostered: std::collections::BTreeSet<&str> = CloneFamilyEntry::ALL
        .iter()
        .map(|entry| entry.builder())
        .collect();
    let excluded: std::collections::BTreeSet<&str> = NOT_A_CLONE_FAMILY_MEMBER
        .iter()
        .map(|(name, _)| *name)
        .collect();

    let undecided: Vec<&String> = exported
        .iter()
        .filter(|name| !rostered.contains(name.as_str()) && !excluded.contains(name.as_str()))
        .collect();
    assert!(
        undecided.is_empty(),
        "nn::attention exports {undecided:?} with no recorded decision. Fix: \
         add each to the CloneFamilyEntry roster and bless its golden section, \
         or record it in NOT_A_CLONE_FAMILY_MEMBER with the reason it is not a \
         clone-family entry point."
    );

    let stale: Vec<&&str> = excluded
        .iter()
        .filter(|name| !exported.contains(&(**name).to_string()))
        .collect();
    assert!(
        stale.is_empty(),
        "NOT_A_CLONE_FAMILY_MEMBER records {stale:?}, which nn::attention no \
         longer exports. Fix: drop the stale rows."
    );

    let both: Vec<&&str> = excluded
        .iter()
        .filter(|name| rostered.contains(**name))
        .collect();
    assert!(
        both.is_empty(),
        "{both:?} is both rostered and recorded as not a clone-family member"
    );
}

/// Snake-case item names a module re-exports, which are its public builders.
///
/// Types, errors and constants are named in the other two cases, so the initial
/// character decides: lowercase is a function, uppercase is a type or a
/// constant.
fn reexported_builder_names(source: &str) -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    let mut rest = source;
    while let Some((_, after)) = rest.split_once("pub use ") {
        let (statement, tail) = after.split_once(';').unwrap_or((after, ""));
        rest = tail;
        let items = statement
            .split_once('{')
            .map_or_else(|| statement.rsplit("::").next().unwrap_or(""), |(_, braced)| {
                braced.split_once('}').map_or(braced, |(inner, _)| inner)
            });
        for item in items.split(',') {
            let item = item.trim();
            let item = item.rsplit("::").next().unwrap_or(item).trim();
            if item.starts_with(|character: char| character.is_ascii_lowercase())
                && item
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || character == '_')
            {
                names.insert(item.to_string());
            }
        }
    }
    names
}

/// The rendering must be a pure function of the program.
///
/// A renderer that read an address, a cache or an iteration order would match
/// the golden once and diverge on the next run, which reads as an IR change.
#[test]
fn structural_ir_is_deterministic_across_builds() {
    assert_eq!(render_corpus(), render_corpus());
}

#[test]
#[ignore = "bless: rewrites the pinned structural IR golden; run deliberately and review the diff"]
fn bless_pinned_structural_ir_golden() {
    write_golden(&golden_path(), &render_corpus());
}

/// Write the full structural IR of every entry point under the test target
/// directory, for reading a digest move the histograms do not explain.
///
/// The golden pins a digest over this rendering rather than the rendering
/// itself, because two direct-path fixtures unroll to about 53000 lines each
/// and the corpus would be ten megabytes. This is how a maintainer gets the
/// text: run it on both sides of the change and diff the two trees.
#[test]
#[ignore = "diagnostic: writes the full structural IR rendering, for diffing a digest move"]
fn dump_full_structural_ir() {
    let out = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("nn_attention_structural_ir");
    for entry in CloneFamilyEntry::ALL {
        write_golden(
            &out.join(format!("{}.ir.txt", entry.id().replace('/', "_"))),
            &render_structural_ir(&entry.build()),
        );
    }
    println!(
        "wrote the full structural IR of 26 entry points to {}",
        out.display()
    );
}

/// Body of the innermost region an entry point wraps its kernel in.
///
/// An entry point that composes a registered core wraps twice: its own region
/// around a child region naming the core. Descending to the innermost one
/// compares the kernels rather than the attribution around them.
fn region_body(program: &Program) -> Vec<Node> {
    let mut nodes = program.entry().to_vec();
    loop {
        match nodes.as_slice() {
            [Node::Region { body, .. }] => nodes = body.as_ref().clone(),
            [] => panic!("expected a wrapping region, got an empty entry"),
            _ => return nodes,
        }
    }
}

/// The `if item < count { .. }` body that both tiled decoders end their entry
/// with.
fn guarded_body(nodes: &[Node]) -> Vec<Node> {
    match nodes.last() {
        Some(Node::If { then, .. }) => then.clone(),
        other => panic!("expected a trailing invocation guard, got {other:?}"),
    }
}

fn tile_loop_body(per_item: &[Node]) -> Vec<Node> {
    per_item
        .iter()
        .find_map(|node| match node {
            Node::Loop { var, body, .. } if var.as_str() == "tile_idx" => Some(body.clone()),
            _ => None,
        })
        .expect("per-item body drives a `tile_idx` loop")
}

/// `mla_decode` and `flash_attention_2` must run the identical online-softmax
/// recurrence; only the score pass and the accumulator update are theirs.
///
/// This goes red if either decoder acquires a private copy of the skeleton, or
/// if the two feed the shared skeleton parameters that make it emit different
/// nodes for the same tiling.
#[test]
fn mla_and_flash_attention_2_share_the_online_softmax_skeleton() {
    let mla = mla_fixture();
    let flash = flash_fixture();
    let mla_item = guarded_body(&region_body(&mla));
    let flash_item = guarded_body(&region_body(&flash));

    assert_eq!(
        mla_item.len(),
        flash_item.len(),
        "per-item skeleton gained or lost a stage in one decoder only"
    );
    // 0 is the query load (buffer name and item variable differ); 4 is the
    // tile loop, compared separately below.
    assert_eq!(
        mla_item[1..4],
        flash_item[1..4],
        "m / l / o_acc init drifted"
    );
    assert_eq!(mla_item[5], flash_item[5], "denominator guard drifted");

    let mla_tile = tile_loop_body(&mla_item);
    let flash_tile = tile_loop_body(&flash_item);
    // 3 is the op-specific score pass; 11.. is the op-specific accumulator
    // update, whose node count differs between the two decoders.
    assert_eq!(mla_tile[0..3], flash_tile[0..3], "tile bounds drifted");
    assert_eq!(
        mla_tile[4..11],
        flash_tile[4..11],
        "tile max / m_new / rescale / tile sum drifted"
    );
    assert_eq!(
        mla_tile.last(),
        flash_tile.last(),
        "running-max carry drifted"
    );
}

/// Every distinct region-generator identity reachable from a program's entry,
/// sorted and deduplicated.
fn region_identities(program: &Program) -> Vec<String> {
    fn walk(node: &Node, out: &mut Vec<String>) {
        match node {
            Node::Region {
                generator, body, ..
            } => {
                out.push(generator.as_str().to_string());
                body.iter().for_each(|child| walk(child, out));
            }
            Node::Block(body) => body.iter().for_each(|child| walk(child, out)),
            Node::Loop { body, .. } => body.iter().for_each(|child| walk(child, out)),
            Node::If {
                then, otherwise, ..
            } => {
                then.iter().for_each(|child| walk(child, out));
                otherwise.iter().for_each(|child| walk(child, out));
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    program.entry().iter().for_each(|node| walk(node, &mut out));
    out.sort();
    out.dedup();
    out
}

/// Region-generator identities each entry point is expected to carry.
///
/// Most entry points inline everything into their own region; the reduce-family
/// builders, the three-pass score owner and the online-softmax core embed
/// shared child regions, and those child identities are the collapse contract:
/// a shared owner that stops being reached, or a builder that reacquires a
/// private copy of a collapsed loop, changes this set.
const EXPECTED_IDENTITIES: [(&str, &[&str]); 26] = [
    (
        "recurrent_gated_delta/f32",
        &["vyre-libs::nn::recurrent_gated_delta"],
    ),
    (
        "recurrent_gated_delta/f16",
        &["vyre-libs::nn::recurrent_gated_delta"],
    ),
    (
        "chunked_gated_delta/f32",
        &["vyre-libs::nn::chunked_gated_delta"],
    ),
    (
        "chunked_gated_delta/f16",
        &["vyre-libs::nn::chunked_gated_delta"],
    ),
    ("mla_decode", &["vyre-libs::nn::mla_decode"]),
    (
        "flash_attention_2",
        &[
            "vyre-libs::nn::attention::absorb_values",
            "vyre-libs::nn::attention::online_softmax",
            "vyre-libs::nn::attention::tile_scores",
            "vyre-libs::nn::flash_attention_2",
        ],
    ),
    (
        "softmax",
        &[
            "anonymous::vyre-libs::builder::strided_writeback",
            "vyre-libs::builder::strided_accumulate",
            "vyre-libs::nn::softmax",
            "vyre-libs::reduce::workgroup_max_f32",
            "vyre-libs::reduce::workgroup_sum_f32",
        ],
    ),
    (
        "layer_norm",
        &[
            "anonymous::vyre-libs::builder::strided_writeback",
            "vyre-libs::builder::strided_accumulate",
            "vyre-libs::nn::layer_norm",
            "vyre-libs::reduce::workgroup_sum_f32",
        ],
    ),
    (
        "flash_attention",
        &[
            "vyre-libs::nn::attention::absorb_values",
            "vyre-libs::nn::attention::online_softmax",
            "vyre-libs::nn::attention::tile_scores",
            "vyre-libs::nn::flash_attention",
        ],
    ),
    (
        "flash_attention/direct",
        &["vyre-libs::nn::flash_attention"],
    ),
    ("attention", &["vyre-libs::nn::attention"]),
    ("attention/direct", &["vyre-libs::nn::attention"]),
    (
        "attention_reference",
        &[
            "vyre-libs::math::dot_partial",
            "vyre-libs::nn::attention_max_pass",
            "vyre-libs::nn::attention_reference",
            "vyre-libs::nn::attention_sum_pass",
            "vyre-libs::nn::attention_write_pass",
        ],
    ),
    (
        "gqa_attention",
        &[
            "vyre-libs::math::dot_partial",
            "vyre-libs::nn::attention_max_pass",
            "vyre-libs::nn::attention_sum_pass",
            "vyre-libs::nn::attention_write_pass",
            "vyre-libs::nn::gqa_attention",
        ],
    ),
    (
        "gqa_attention_causal",
        &[
            "vyre-libs::math::dot_partial",
            "vyre-libs::nn::attention_max_pass",
            "vyre-libs::nn::attention_sum_pass",
            "vyre-libs::nn::attention_write_pass",
            "vyre-libs::nn::gqa_attention_causal",
        ],
    ),
    (
        "gqa_attention_causal/f16",
        &[
            "vyre-libs::math::dot_partial",
            "vyre-libs::nn::attention_max_pass",
            "vyre-libs::nn::attention_sum_pass",
            "vyre-libs::nn::attention_write_pass",
            "vyre-libs::nn::gqa_attention_causal",
        ],
    ),
    ("kv_cache_append", &["vyre-libs::nn::kv_cache_append"]),
    ("kv_cache_append/f16", &["vyre-libs::nn::kv_cache_append"]),
    (
        "attention_head_to_token",
        &["vyre-libs::nn::attention_head_to_token"],
    ),
    (
        "attention_head_to_token/f16",
        &["vyre-libs::nn::attention_head_to_token"],
    ),
    (
        "attention_token_to_head",
        &["vyre-libs::nn::attention_token_to_head"],
    ),
    (
        "quest_paging",
        &[
            "vyre-libs::nn::attention::quest_paging",
            "vyre-libs::nn::quest_score_pages",
            "vyre-libs::nn::quest_select_top_k",
            "vyre-libs::nn::quest_zero_fill",
        ],
    ),
    ("partial_rope", &["vyre-libs::nn::partial_rope"]),
    ("qk_gain", &["vyre-libs::nn::qk_gain"]),
    (
        "turboquant_attention",
        &["vyre-libs::nn::attention::turboquant"],
    ),
    ("mla_compress_kv", &["vyre-libs::nn::mla_compress_kv"]),
];

/// Which shared child regions each entry point embeds, by name.
///
/// The golden sees an identity rename as a text difference but cannot say
/// whether the work behind it moved. This rule answers that half: it pins the
/// reuse graph, so an owner that stops being reached, or a builder that
/// reacquires a private copy of a collapsed loop, is named even when the
/// emitted work is equivalent.
///
/// What this does not catch: an IR change that keeps every identity, which is
/// what the golden is for. The two rules are complements.
#[test]
fn clone_family_entry_points_carry_the_pinned_region_identities() {
    let observed: Vec<(&'static str, Vec<String>)> = entry_points()
        .iter()
        .map(|(name, program)| (*name, region_identities(program)))
        .collect();

    assert_eq!(
        observed.len(),
        EXPECTED_IDENTITIES.len(),
        "fixture count drifted from the pinned identity table"
    );

    // Floor: a walker that stopped descending, or a builder that stopped
    // wrapping its entry in a region, would otherwise pass this vacuously.
    let mut union: Vec<&str> = Vec::new();
    for (name, identities) in &observed {
        assert!(
            !identities.is_empty(),
            "{name} emitted no region at all, so the identity walk proved nothing"
        );
        union.extend(identities.iter().map(String::as_str));
    }
    union.sort_unstable();
    union.dedup();
    assert!(
        union.len() >= 10,
        "the entry points reach only {} distinct region identities; the \
         walk is no longer descending into child regions",
        union.len()
    );

    let mut drifted = false;
    let mut report = String::new();
    for ((name, got), (pinned_name, pinned)) in observed.iter().zip(EXPECTED_IDENTITIES.iter()) {
        assert_eq!(name, pinned_name, "fixture order drifted from the table");
        let got: Vec<&str> = got.iter().map(String::as_str).collect();
        if got != *pinned {
            drifted = true;
        }
        report.push_str(&format!("    (\n        \"{name}\",\n        &[\n"));
        for identity in &got {
            report.push_str(&format!("            \"{identity}\",\n"));
        }
        report.push_str("        ],\n    ),\n");
    }
    assert!(
        !drifted,
        "region identities changed for at least one entry point. Observed:\n{report}"
    );

    // The collapse contract: the reduce-family owner is reused, not copied.
    let shared = "vyre-libs::builder::strided_accumulate";
    let consumers = observed
        .iter()
        .filter(|(_, ids)| ids.iter().any(|id| id == shared))
        .count();
    assert!(
        consumers >= 2,
        "{shared} is embedded by {consumers} entry point(s); a shared owner \
         reached by one caller has been cloned back apart"
    );
}
