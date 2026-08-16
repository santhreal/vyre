//! IR invariance across the three `nn/attention` clone families.
//!
//! Three families of duplicated builder code were collapsed onto single
//! owners: the gated delta index/shape math, the tiled online-softmax
//! skeleton, and the reduce-then-normalize skeleton. Collapsing a clone family
//! is only safe if the surviving owner emits exactly what every former copy
//! emitted, so this file pins the canonical wire fingerprint of every entry
//! point involved. Any change to a shared helper that is not a deliberate IR
//! change turns these red.
//!
//! What this does not catch: a change that alters the fingerprint on purpose.
//! That is the point at which a human has to decide whether the new IR is
//! correct and re-pin the constant.

#![forbid(unsafe_code)]

mod harness;

use harness::ir_fingerprint::assert_pinned_ir_fingerprints;
use vyre_foundation::ir::{DataType, Node, Program};
use vyre_libs::nn::attention::{
    chunked_gated_delta, flash_attention_2, mla_decode, recurrent_gated_delta, softmax,
    GatedDeltaSpec,
};
use vyre_libs::nn::norm::layer_norm;

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
    ) -> Result<Program, vyre_libs::nn::attention::RecurrentGatedDeltaError>,
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

fn entry_points() -> Vec<(&'static str, Program)> {
    vec![
        (
            "recurrent_gated_delta/f32",
            gated_delta_fixture(recurrent_gated_delta, DataType::F32),
        ),
        (
            "recurrent_gated_delta/f16",
            gated_delta_fixture(recurrent_gated_delta, DataType::F16),
        ),
        (
            "chunked_gated_delta/f32",
            gated_delta_fixture(chunked_gated_delta, DataType::F32),
        ),
        (
            "chunked_gated_delta/f16",
            gated_delta_fixture(chunked_gated_delta, DataType::F16),
        ),
        ("mla_decode", mla_fixture()),
        ("flash_attention_2", flash_fixture()),
        ("softmax", softmax("input", "output", 1000)),
        ("layer_norm", layer_norm("input", "output", 1000, 1e-5)),
    ]
}

/// Canonical wire fingerprints recorded on the pre-merge tree, before any
/// clone family was collapsed.
///
/// `softmax` and `layer_norm` were re-pinned when the shared reduce-family
/// child regions were renamed from `vyre-libs::substrate::*` to
/// `vyre-libs::builder::*`. A generator identity is part of the wire encoding,
/// so that rename moves the fingerprint of every program embedding it. It was
/// proved to be the whole difference by rewriting only those identity strings
/// back in the built programs, which reproduced the previous two digests
/// exactly (`08fde137..fc37373` and `54e9357e..3a643b7a9a6`); no node, buffer,
/// expression, or workgroup value moved. The other six entry points do not
/// embed a shared child region and were unaffected.
/// `clone_family_entry_points_carry_the_pinned_region_identities` now names
/// such a rename directly instead of leaving it as an opaque digest change.
const EXPECTED: [(&str, &str); 8] = [
    (
        "recurrent_gated_delta/f32",
        "8d6a5194770b4b70680793d431b9876ad49f855cc9d0c42b2af4142bb941b0ed",
    ),
    (
        "recurrent_gated_delta/f16",
        "d91721b7fb4ea4ba4a28f6606879b65cd260710ee9aa8116f6adad1987b99a5c",
    ),
    (
        "chunked_gated_delta/f32",
        "b8bd386c2cb2c43f0892e01be3c926674fc70ff72a97b8b772e6cc4da5553ca6",
    ),
    (
        "chunked_gated_delta/f16",
        "fbebff41ebd0ebdb0506e4e366b36db1b1371d9023557b709c1641fa81932896",
    ),
    (
        "mla_decode",
        "941fa633f1ede895109a44221f2413351a12ee36fb8cb931f47056233b1fae91",
    ),
    (
        "flash_attention_2",
        "21af2e2b23a3bfb853a9cc950cd70194fa2c958f26af159acd0ef25a91123ff2",
    ),
    (
        "softmax",
        "1ac237e7b2a89e3e2738346c6a41246e967e739a726ab0a44c09e927dbc19d8e",
    ),
    (
        "layer_norm",
        "5cc4d4c537072eb8f99ff971606ef940de385fc9b817b9700cb2d56105ec4b33",
    ),
];

#[test]
fn clone_family_entry_points_emit_the_pinned_ir() {
    assert_pinned_ir_fingerprints(&entry_points(), &EXPECTED);
}

/// Body of the single region every one of these builders wraps its entry in.
fn region_body(program: &Program) -> Vec<Node> {
    match program.entry().first() {
        Some(Node::Region { body, .. }) => body.as_ref().clone(),
        other => panic!("expected one wrapping region, got {other:?}"),
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
/// The first six entry points inline everything into their own region; only
/// the two reduce-family builders embed shared child regions, and those child
/// identities are the collapse contract: `strided_accumulate` and
/// `strided_writeback` exist so `softmax` and `layer_norm` stop carrying
/// private copies of the same loop.
const EXPECTED_IDENTITIES: [(&str, &[&str]); 8] = [
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
    ("flash_attention_2", &["vyre-libs::nn::flash_attention_2"]),
    (
        "softmax",
        &[
            "anonymous::vyre-libs::builder::strided_writeback",
            "vyre-libs::builder::strided_accumulate",
            "vyre-libs::nn::softmax",
            "vyre-primitives::reduce::workgroup_max_f32",
            "vyre-primitives::reduce::workgroup_sum_f32",
        ],
    ),
    (
        "layer_norm",
        &[
            "vyre-libs::builder::strided_accumulate",
            "vyre-libs::nn::layer_norm",
            "vyre-primitives::reduce::workgroup_sum_f32",
        ],
    ),
];

/// A generator identity is part of the wire encoding, so renaming one moves
/// every fingerprint that embeds it. `clone_family_entry_points_emit_the_pinned_ir`
/// sees that as an opaque 32-byte difference and cannot say whether the IR or
/// only a name moved. This rule answers that question by name.
///
/// It also pins which entry points share a child region: an owner that stops
/// being reused, or a builder that reacquires a private copy of a collapsed
/// loop, changes this set even when the emitted work is equivalent.
///
/// What this does not catch: an IR change that keeps every identity, which is
/// what the fingerprint pin is for. The two rules are complements.
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
        "the eight entry points reach only {} distinct region identities; the \
         walk is no longer descending into child regions",
        union.len()
    );

    for ((name, got), (pinned_name, pinned)) in observed.iter().zip(EXPECTED_IDENTITIES.iter()) {
        assert_eq!(name, pinned_name, "fixture order drifted from the table");
        let got: Vec<&str> = got.iter().map(String::as_str).collect();
        assert_eq!(
            got, *pinned,
            "{name} no longer carries the pinned region identities"
        );
    }

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
