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
    attention, attention_head_to_token, attention_reference, attention_token_to_head,
    chunked_gated_delta, flash_attention, flash_attention_2, gqa_attention, gqa_attention_causal,
    gqa_attention_causal_typed, kv_cache_append, mla_compress_kv, mla_decode, partial_rope,
    qk_gain, quest_paging, recurrent_gated_delta, softmax, turboquant_attention,
    AttentionPermuteSpec, GatedDeltaSpec, KvCacheAppendSpec,
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

/// The layout-move fixture shape, ragged in every axis so a transposed index
/// derivation cannot produce the same fingerprint.
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
        (
            "attention",
            attention("q", "k", "v", "out", SEQ_LEN, HEAD_DIM),
        ),
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
            kv_cache_append(cache_spec(DataType::F32)).expect("cache builds"),
        ),
        (
            "kv_cache_append/f16",
            kv_cache_append(cache_spec(DataType::F16)).expect("typed cache builds"),
        ),
        (
            "attention_head_to_token",
            attention_head_to_token(permute_spec(DataType::F32)).expect("head to token builds"),
        ),
        (
            "attention_head_to_token/f16",
            attention_head_to_token(permute_spec(DataType::F16))
                .expect("typed head to token builds"),
        ),
        (
            "attention_token_to_head",
            attention_token_to_head(permute_spec(DataType::F32)).expect("token to head builds"),
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

/// Canonical wire fingerprints recorded for all clone-family entry points.
///
/// The fingerprints across all 26 entry points moved together during integration
/// due to canonical wire format and IR model unification:
/// 1. Region attribution `source_region` transitioned from `Option<GeneratorRef>`
///    to `Option<Ident>` in the AST and wire stream (`c7bdcef`).
/// 2. Canonical wire serialization unified output-buffer projections via
///    `OutputSet::encode_from_buffers_into` across the wire framing envelope.
/// 3. Operation registry and namespace consolidation moved shared builder
///    and reduction identifiers to canonical paths.
///
/// `softmax` and `layer_norm` embed the shared `strided_writeback_child` helper,
/// carrying `anonymous::vyre-libs::builder::strided_writeback`. By the composition
/// contract in `vyre_foundation::composition` (`ANONYMOUS_GENERATOR_PREFIXES`),
/// internal phase boundaries that are not standalone catalog operations use the
/// `anonymous::` prefix so validation and LEGO composability gates distinguish
/// phase attribution from catalog operation references without duplicating writeback.
///
/// `mla_decode` and `flash_attention_2` continue to share the exact online-softmax
/// core verified by `mla_and_flash_attention_2_share_the_online_softmax_skeleton`.
///
/// `partial_rope` alone moved when its pair base and rotation-table index were
/// folded through the `dim < rope_dims` predicate that already selected the
/// result: an `Expr::select` evaluates both arms, so the discarded arm was
/// issuing a load past the table. The values it computes are unchanged.
///
/// Value semantics, workgroup tiling, memory layout, and ABI contracts across all
/// 26 members are preserved unchanged.
const EXPECTED: [(&str, &str); 26] = [
    (
        "recurrent_gated_delta/f32",
        "b5243ad8be818a97a70d3964b892eb1ac495f8de5f6bed8d26a9252eba15e9cc",
    ),
    (
        "recurrent_gated_delta/f16",
        "c769cc83d2e7804bfa52c974f039a5fd88e9f6f1f18b7fe5b0cb02825925b8b0",
    ),
    (
        "chunked_gated_delta/f32",
        "b1037852078f03e033964cee03ebd2cf151de663b76302674b328e3d6768a080",
    ),
    (
        "chunked_gated_delta/f16",
        "4ed59de3ec45030af5ed245fecbad6036e17839401c14ba2aa9ebcd571e9e53b",
    ),
    (
        "mla_decode",
        "cb69d981fbb62eeffb8b84dafc6a04284281bd82aafee0b0502c3ccb606a04e3",
    ),
    (
        "flash_attention_2",
        "9aa03a353c3b87f76975cbde281077446d1b3c9f22ff5ec4d6b00bff65e17f92",
    ),
    (
        "softmax",
        "cf69d5908f9190581a5b8eeb23c4c60f4b2828f870a5ee2904d9604216e125b8",
    ),
    (
        "layer_norm",
        "7431d2f6951e9a8a081209cff20520140411b3aacfac1d09c6d1b5f0670e2d30",
    ),
    (
        "flash_attention",
        "5aa2a2cf9aa224dff5f228eba6fd9cc57473f570baf4445d4f3bcca19cc4fc5d",
    ),
    (
        "flash_attention/direct",
        "4a277768597960cf686844d7ac398e09991fbc616c438bf5fb4c4fa7326363f2",
    ),
    (
        "attention",
        "ac08103ff84292714c0f476ad65827006e61c61afa0dc5669e40aaf1a8037f98",
    ),
    (
        "attention/direct",
        "514240c2d5ec2fce566d2a3c3cc4359b037f466a653df9a73f357c6e5349cce6",
    ),
    (
        "attention_reference",
        "56499d8f1c4b50d2dc14a2b58bea386cf129bba8cbb8e03f9f448de0310e9e81",
    ),
    (
        "gqa_attention",
        "c28f389e6c6ef79dc5f16f71cbe1b771445dcf4e0db0627e8f05f3751d5f21cb",
    ),
    (
        "gqa_attention_causal",
        "1732d07ea3ee4a67a56206bff92a034916e2717b22e5ea8dd1ea2afaceb88106",
    ),
    (
        "gqa_attention_causal/f16",
        "414f71575b504f9d87994410e3c3886e55441df8b998eaeea1117250c54d1cda",
    ),
    (
        "kv_cache_append",
        "5fe0316e53a89d4089ab3053aec66b4dae56378bcbaac60ff6279b8be2490a0c",
    ),
    (
        "kv_cache_append/f16",
        "78b68a761f0b9ebd6992b575eab928f214386a5c8979bc3130f715beae355bb9",
    ),
    (
        "attention_head_to_token",
        "60289c54e895c5bfbf48377ba4fc8afb8e5c252990a6e922fdde0c2595de6f35",
    ),
    (
        "attention_head_to_token/f16",
        "6ad7f7c541c13f9649900b593be298c2aa946ec35f2211f074219557b8dc5b1d",
    ),
    (
        "attention_token_to_head",
        "3362251830e9e647e40b09a77ecdf7b9c421ffd765634efc5889df7ada55f97b",
    ),
    (
        "quest_paging",
        "75169e0495b1fa25ec835237ec6bd09c0158e0f6374bb266b46b148e1bb42a2d",
    ),
    (
        "partial_rope",
        "2d83f47ff07aa25d653d22c9c1b545afd0190201ca838868f8957e2dbf5017fc",
    ),
    (
        "qk_gain",
        "43ad8ebe8e28e11286d16661d777d87027a47bba2bc3d6c8ed975fe094762f62",
    ),
    (
        "turboquant_attention",
        "66b2497f36135d83c98c22a8e388469626cbfef5f47eb9c2f313488834ad5426",
    ),
    (
        "mla_compress_kv",
        "5e8d2fec52e53b4b7896113a343f9c508e34797a8de3cccf66725971f9234a36",
    ),
];

#[test]
fn clone_family_entry_points_emit_the_pinned_ir() {
    assert_pinned_ir_fingerprints(&entry_points(), &EXPECTED);
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
