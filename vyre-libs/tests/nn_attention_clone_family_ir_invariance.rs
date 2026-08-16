//! IR invariance across the `nn/attention` clone families.
//!
//! Families of duplicated builder code were collapsed onto single owners: the
//! gated delta index/shape math, the online-softmax core, the reduce-then-
//! normalize skeleton, the three-pass score/sum/write owner, and the layout
//! index-map owner. Collapsing a clone family is only safe if the surviving
//! owner emits exactly what every former copy emitted, so this file pins the
//! canonical wire fingerprint of every entry point involved. Any change to a
//! shared helper that is not a deliberate IR change turns these red.
//!
//! What this does not catch: a change that alters the fingerprint on purpose.
//! That is the point at which a human has to decide whether the new IR is
//! correct and re-pin the constant.

#![forbid(unsafe_code)]

mod harness;

use harness::ir_fingerprint::assert_pinned_ir_fingerprints;
use vyre_foundation::ir::{DataType, Node, Program};
use vyre_libs::nn::attention::{
    attention, attention_head_to_token, attention_head_to_token_typed, attention_reference,
    attention_token_to_head, chunked_gated_delta, flash_attention, flash_attention_2,
    gqa_attention, gqa_attention_causal, gqa_attention_causal_typed, kv_cache_append,
    kv_cache_append_typed, mla_compress_kv, mla_decode, partial_rope, qk_gain, quest_paging,
    recurrent_gated_delta, softmax, turboquant_attention, GatedDeltaSpec,
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
        (
            "flash_attention",
            flash_attention("q", "k", "v", "out", SEQ_LEN, HEAD_DIM).expect("flash builds"),
        ),
        (
            "flash_attention/direct",
            flash_attention("q", "k", "v", "out", 4, 4).expect("direct flash builds"),
        ),
        ("attention", attention("q", "k", "v", "out", SEQ_LEN, HEAD_DIM)),
        ("attention/direct", attention("q", "k", "v", "out", 4, 4)),
        (
            "attention_reference",
            attention_reference("q", "k", "v", "out", 8, 4),
        ),
        (
            "gqa_attention",
            gqa_attention("q", "k", "v", "out", 4, 2, 8, 4).expect("gqa builds"),
        ),
        (
            "gqa_attention_causal",
            gqa_attention_causal("q", "k", "v", "out", 2, 4, 2, 3, 8, 4, 2)
                .expect("causal gqa builds"),
        ),
        (
            "gqa_attention_causal/f16",
            gqa_attention_causal_typed("q", "k", "v", "out", 2, 4, 2, 3, 8, 4, 2, DataType::F16)
                .expect("typed causal gqa builds"),
        ),
        (
            "kv_cache_append",
            kv_cache_append("prior", "chunk", "next", 2, 2, 8, 3, 4, 2).expect("cache builds"),
        ),
        (
            "kv_cache_append/f16",
            kv_cache_append_typed("prior", "chunk", "next", 2, 2, 8, 3, 4, 2, DataType::F16)
                .expect("typed cache builds"),
        ),
        (
            "attention_head_to_token",
            attention_head_to_token("input", "output", 2, 3, 5, 4).expect("head to token builds"),
        ),
        (
            "attention_head_to_token/f16",
            attention_head_to_token_typed("input", "output", 2, 3, 5, 4, DataType::F16)
                .expect("typed head to token builds"),
        ),
        (
            "attention_token_to_head",
            attention_token_to_head("input", "output", 2, 5, 3, 4).expect("token to head builds"),
        ),
        (
            "quest_paging",
            quest_paging("q", "meta", "scores", "io", 8, 3, 4),
        ),
        (
            "partial_rope",
            partial_rope("input", "cos", "sin", "output", 2, 5, 8, 4),
        ),
        ("qk_gain", qk_gain("q_in", "q_out", "gain", 3, 5, 4)),
        (
            "turboquant_attention",
            turboquant_attention("q", "k_packed", "v_packed", "out", 6, 4),
        ),
        (
            "mla_compress_kv",
            mla_compress_kv("h", "w_dk", "c_out", 6, 4).expect("mla compress builds"),
        ),
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
const EXPECTED: [(&str, &str); 26] = [
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
    (
        "flash_attention",
        "9f9805b63602d6b136c12756108f18c6df06074f51acc4807ec88c1b451e0d6a",
    ),
    (
        "flash_attention/direct",
        "dd3261d87484903b825a98979c36f3eb946d90fd4d02c9af823cf0bc97877871",
    ),
    (
        "attention",
        "fa758fcede885ecdcbf9a095d16ee1e20d4883d25844b475c9c4dce1ad73d885",
    ),
    (
        "attention/direct",
        "9fcb13a00c4c4b7ac0518574e0ebc1ebc949971c587bc09a889de5cb86d1f591",
    ),
    (
        "attention_reference",
        "058035e29c4514d5d9fb78ed82039006f94225fdbc3e2a10c42a72cfe05ed628",
    ),
    (
        "gqa_attention",
        "ba987863b55829511275b9b943d86662c7ead1172816981097456ed98737d80c",
    ),
    (
        "gqa_attention_causal",
        "96200c1d31d6a4023732377786557a517defcbaca504f3734f973054417b67eb",
    ),
    (
        "gqa_attention_causal/f16",
        "f9d16c17394ac3871c3dd8c1d715c0d22d0a694142c569832c415a8711ca405a",
    ),
    (
        "kv_cache_append",
        "13404f04d3caabb8e5b73e4a87b81f4e9a43f3769524bc9de222828223dba891",
    ),
    (
        "kv_cache_append/f16",
        "65b5891b893a906a2ac6f9c5d4364c2b989c93eda3c17b387b6c3e562cfbf306",
    ),
    (
        "attention_head_to_token",
        "3cdc5b0cae92da81221b5158a2d12968884db611a7340f106630bd685898b588",
    ),
    (
        "attention_head_to_token/f16",
        "bd08abf7a28bc81ef126b5a37dcb6bd8e2ac44b9496209967c04aab9527a2f8b",
    ),
    (
        "attention_token_to_head",
        "1620c80bcfe46cc52fc3a319a9bbbdd03b3dbc137cb135adc2f9e5bddc04ed6d",
    ),
    (
        "quest_paging",
        "0ef6ca5ca04dc26c4e2040894817d77e997212a06e9957f914cd5fcdc7998d2d",
    ),
    (
        "partial_rope",
        "80ab8860b8375cccde486cd3fcec0ab9529715fd4661551145ea9182ef4b9eb3",
    ),
    (
        "qk_gain",
        "9acc6d451eb5ff12028cbda2497a0b46135c160d4a1b3f34c7c65402b1e514d9",
    ),
    (
        "turboquant_attention",
        "a764e7aa79e16b2ecbbc9cde4bad5cc505fc10f1cf6369eceebf3198c45d2bb9",
    ),
    (
        "mla_compress_kv",
        "d2c7c68b9a9d6f8432d0589f6ca3875042db55d00f20961f2e27857b4ec49e1d",
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
    (
        "flash_attention",
        &[
            "vyre-libs::nn::flash_attention",
        ],
    ),
    (
        "flash_attention/direct",
        &[
            "vyre-libs::nn::flash_attention",
        ],
    ),
    (
        "attention",
        &[
            "vyre-libs::nn::attention",
        ],
    ),
    (
        "attention/direct",
        &[
            "vyre-libs::nn::attention",
        ],
    ),
    (
        "attention_reference",
        &[
            "vyre-libs::nn::attention_reference",
            "vyre-primitives::math::dot_partial",
            "vyre-primitives::nn::attention_max_pass",
            "vyre-primitives::nn::attention_sum_pass",
            "vyre-primitives::nn::attention_write_pass",
        ],
    ),
    (
        "gqa_attention",
        &[
            "vyre-libs::nn::gqa_attention",
            "vyre-primitives::math::dot_partial",
            "vyre-primitives::nn::attention_max_pass",
            "vyre-primitives::nn::attention_sum_pass",
            "vyre-primitives::nn::attention_write_pass",
        ],
    ),
    (
        "gqa_attention_causal",
        &[
            "vyre-libs::nn::gqa_attention_causal",
            "vyre-primitives::math::dot_partial",
            "vyre-primitives::nn::attention_max_pass",
            "vyre-primitives::nn::attention_sum_pass",
            "vyre-primitives::nn::attention_write_pass",
        ],
    ),
    (
        "gqa_attention_causal/f16",
        &[
            "vyre-libs::nn::gqa_attention_causal",
            "vyre-primitives::math::dot_partial",
            "vyre-primitives::nn::attention_max_pass",
            "vyre-primitives::nn::attention_sum_pass",
            "vyre-primitives::nn::attention_write_pass",
        ],
    ),
    (
        "kv_cache_append",
        &[
            "vyre-libs::nn::kv_cache_append",
        ],
    ),
    (
        "kv_cache_append/f16",
        &[
            "vyre-libs::nn::kv_cache_append",
        ],
    ),
    (
        "attention_head_to_token",
        &[
            "vyre-libs::nn::attention_head_to_token",
        ],
    ),
    (
        "attention_head_to_token/f16",
        &[
            "vyre-libs::nn::attention_head_to_token",
        ],
    ),
    (
        "attention_token_to_head",
        &[
            "vyre-libs::nn::attention_token_to_head",
        ],
    ),
    (
        "quest_paging",
        &[
            "vyre-libs::nn::attention::quest_paging",
            "vyre-primitives::nn::quest_score_pages",
            "vyre-primitives::nn::quest_select_top_k",
            "vyre-primitives::nn::quest_zero_fill",
        ],
    ),
    (
        "partial_rope",
        &[
            "vyre-libs::nn::partial_rope",
        ],
    ),
    (
        "qk_gain",
        &[
            "vyre-libs::nn::qk_gain",
        ],
    ),
    (
        "turboquant_attention",
        &[
            "vyre-libs::nn::attention::turboquant",
        ],
    ),
    (
        "mla_compress_kv",
        &[
            "vyre-libs::nn::mla_compress_kv",
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
