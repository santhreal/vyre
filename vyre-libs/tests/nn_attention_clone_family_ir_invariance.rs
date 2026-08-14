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

use vyre_foundation::ir::{DataType, Node, Program};
use vyre_libs::nn::attention::{
    chunked_gated_delta, flash_attention_2, mla_decode, recurrent_gated_delta, softmax,
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
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
        &str,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        f32,
        DataType,
    ) -> Result<Program, vyre_libs::nn::attention::RecurrentGatedDeltaError>,
    dtype: DataType,
) -> Program {
    build(
        "query",
        "key",
        "value",
        "decay_log",
        "beta_logits",
        "state_in",
        "out",
        "state_out",
        2,
        DELTA_SEQ,
        2,
        4,
        3,
        5,
        1e-5,
        dtype,
    )
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
        "08fde137ddbc772f6c226786603cb5634079a1dbd3cf401f101eaf401fc37373",
    ),
    (
        "layer_norm",
        "54e9357ea14f91fd3ec159d84053e6bde84b6c2b7eb64078d980c3a643b7a9a6",
    ),
];

fn hex(program: &Program) -> String {
    program
        .fingerprint()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn clone_family_entry_points_emit_the_pinned_ir() {
    let actual: Vec<(&'static str, String)> = entry_points()
        .iter()
        .map(|(name, program)| (*name, hex(program)))
        .collect();
    assert_eq!(
        actual.len(),
        EXPECTED.len(),
        "fixture count drifted from the pinned table"
    );
    let mut report = String::new();
    let mut drifted = false;
    for ((name, got), (pinned_name, pinned)) in actual.iter().zip(EXPECTED.iter()) {
        assert_eq!(name, pinned_name, "fixture order drifted from the table");
        if got != pinned {
            drifted = true;
        }
        report.push_str(&format!(
            "    (\n        \"{name}\",\n        \"{got}\",\n    ),\n"
        ));
    }
    assert!(
        !drifted,
        "generated IR changed for at least one clone-family entry point. \
         Recorded fingerprints:\n{report}"
    );
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
