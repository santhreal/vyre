//! Step 0 guard for the security flow-skeleton clone family.
//!
//! Three pairs of security entry points are built from one shared
//! reachability-plus-sanitizer-projection skeleton and differ only in the
//! source / sink / sanitizer predicates they supply:
//!
//! * `flows_to` / `flows_to_alias_only` / `taint_flow`  -  forward reach.
//! * `flows_to_to_sink` / `taint_pollution`  -  forward reach then sink hit.
//! * `bounded_by_comparison` / `dominance_predecessors`  -  backward reach.
//! * `flows_to_with_sanitizer`  -  the sanitizer projection of the hit form.
//!
//! Two independent halves:
//!
//! 1. `Program::fingerprint()` is pinned for every public entry point over a
//!    spread of shapes, including the boundary cases (one node / no edges,
//!    exactly one bitset word, one node past a word boundary, deep graph) and
//!    distinct buffer names. Rehoming a builder onto the shared skeleton must
//!    not move a single byte of emitted IR.
//! 2. Cross-entry-point behavior is checked through the reference interpreter,
//!    which pins the ONE thing each rehomed file is still allowed to own: its
//!    predicate. Paired entry points that declare the same predicate must agree
//!    bit for bit, and the pair that declares different edge masks must diverge
//!    exactly where those masks differ.

#![cfg(feature = "security")]
#![forbid(unsafe_code)]

use vyre::ir::Program;
use vyre_libs::graph::program_graph::ProgramGraphShape;
use vyre_libs::predicate::edge_kind;
use vyre_libs::security::bounded_by_comparison;
use vyre_libs::security::dominance_predecessors;
use vyre_libs::security::flows_to_to_sink;
use vyre_libs::security::flows_to_with_sanitizer;
use vyre_libs::security::taint_flow;
use vyre_libs::security::taint_pollution;
use vyre_libs::security::{flows_to, flows_to_alias_only};
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

fn hex(fingerprint: [u8; 32]) -> String {
    fingerprint.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every pinned entry point, keyed by a case label that names the builder and
/// the input it was built from. A pin only ever changes together with a
/// deliberate, recorded IR change.
fn pinned_cases() -> Vec<(&'static str, Program)> {
    vec![
        // ---- forward reach: flows_to ----
        (
            "flows_to/1n0e/fin-fout",
            flows_to(ProgramGraphShape::new(1, 0), "fin", "fout"),
        ),
        (
            "flows_to/4n3e/fin-fout",
            flows_to(ProgramGraphShape::new(4, 3), "fin", "fout"),
        ),
        (
            "flows_to/32n31e/fin-fout",
            flows_to(ProgramGraphShape::new(32, 31), "fin", "fout"),
        ),
        (
            "flows_to/33n32e/fin-fout",
            flows_to(ProgramGraphShape::new(33, 32), "fin", "fout"),
        ),
        (
            "flows_to/1024n4096e/renamed",
            flows_to(
                ProgramGraphShape::new(1024, 4096),
                "src_frontier",
                "dst_frontier",
            ),
        ),
        // ---- forward reach: flows_to_alias_only ----
        (
            "flows_to_alias_only/1n0e/fin-fout",
            flows_to_alias_only(ProgramGraphShape::new(1, 0), "fin", "fout"),
        ),
        (
            "flows_to_alias_only/4n3e/fin-fout",
            flows_to_alias_only(ProgramGraphShape::new(4, 3), "fin", "fout"),
        ),
        (
            "flows_to_alias_only/33n32e/renamed",
            flows_to_alias_only(
                ProgramGraphShape::new(33, 32),
                "src_frontier",
                "dst_frontier",
            ),
        ),
        // ---- forward reach: taint_flow ----
        (
            "taint_flow/1n0e/fin-fout",
            taint_flow(ProgramGraphShape::new(1, 0), "fin", "fout"),
        ),
        (
            "taint_flow/4n3e/fin-fout",
            taint_flow(ProgramGraphShape::new(4, 3), "fin", "fout"),
        ),
        (
            "taint_flow/32n31e/fin-fout",
            taint_flow(ProgramGraphShape::new(32, 31), "fin", "fout"),
        ),
        (
            "taint_flow/33n32e/fin-fout",
            taint_flow(ProgramGraphShape::new(33, 32), "fin", "fout"),
        ),
        (
            "taint_flow/1024n4096e/renamed",
            taint_flow(
                ProgramGraphShape::new(1024, 4096),
                "src_frontier",
                "dst_frontier",
            ),
        ),
        // ---- backward reach: bounded_by_comparison ----
        (
            "bounded_by_comparison/1n0e/fin-fout",
            bounded_by_comparison(ProgramGraphShape::new(1, 0), "fin", "fout"),
        ),
        (
            "bounded_by_comparison/4n4e/fin-fout",
            bounded_by_comparison(ProgramGraphShape::new(4, 4), "fin", "fout"),
        ),
        (
            "bounded_by_comparison/32n31e/fin-fout",
            bounded_by_comparison(ProgramGraphShape::new(32, 31), "fin", "fout"),
        ),
        (
            "bounded_by_comparison/33n32e/fin-fout",
            bounded_by_comparison(ProgramGraphShape::new(33, 32), "fin", "fout"),
        ),
        (
            "bounded_by_comparison/1024n4096e/renamed",
            bounded_by_comparison(
                ProgramGraphShape::new(1024, 4096),
                "src_frontier",
                "dst_frontier",
            ),
        ),
        // ---- backward reach: dominance_predecessors ----
        (
            "dominance_predecessors/1n0e/fin-fout",
            dominance_predecessors(ProgramGraphShape::new(1, 0), "fin", "fout"),
        ),
        (
            "dominance_predecessors/4n4e/fin-fout",
            dominance_predecessors(ProgramGraphShape::new(4, 4), "fin", "fout"),
        ),
        (
            "dominance_predecessors/32n31e/fin-fout",
            dominance_predecessors(ProgramGraphShape::new(32, 31), "fin", "fout"),
        ),
        (
            "dominance_predecessors/33n32e/fin-fout",
            dominance_predecessors(ProgramGraphShape::new(33, 32), "fin", "fout"),
        ),
        (
            "dominance_predecessors/1024n4096e/renamed",
            dominance_predecessors(
                ProgramGraphShape::new(1024, 4096),
                "src_frontier",
                "dst_frontier",
            ),
        ),
        // ---- reach then sink hit: flows_to_to_sink ----
        (
            "flows_to_to_sink/1n0e",
            flows_to_to_sink(
                ProgramGraphShape::new(1, 0),
                "source",
                "sink",
                "reach",
                "hits",
                "out_scalar",
            ),
        ),
        (
            "flows_to_to_sink/4n3e",
            flows_to_to_sink(
                ProgramGraphShape::new(4, 3),
                "source",
                "sink",
                "reach",
                "hits",
                "out_scalar",
            ),
        ),
        (
            "flows_to_to_sink/33n32e/renamed",
            flows_to_to_sink(
                ProgramGraphShape::new(33, 32),
                "src_set",
                "sink_set",
                "reach_scratch",
                "hit_scratch",
                "any_hit",
            ),
        ),
        // ---- reach then sink hit: taint_pollution ----
        (
            "taint_pollution/1n0e",
            taint_pollution(
                ProgramGraphShape::new(1, 0),
                "source",
                "sink",
                "reach",
                "hits",
                "out_scalar",
            ),
        ),
        (
            "taint_pollution/4n3e",
            taint_pollution(
                ProgramGraphShape::new(4, 3),
                "source",
                "sink",
                "reach",
                "hits",
                "out_scalar",
            ),
        ),
        (
            "taint_pollution/33n32e/renamed",
            taint_pollution(
                ProgramGraphShape::new(33, 32),
                "src_set",
                "label_set",
                "reach_scratch",
                "hit_scratch",
                "any_hit",
            ),
        ),
        // ---- sanitizer projection: flows_to_with_sanitizer ----
        (
            "flows_to_with_sanitizer/1n0e",
            flows_to_with_sanitizer(
                ProgramGraphShape::new(1, 0),
                "source",
                "sink",
                "sanitizer",
                "clean",
                "reach",
                "alive",
                "hits",
                "out_scalar",
            ),
        ),
        (
            "flows_to_with_sanitizer/4n3e",
            flows_to_with_sanitizer(
                ProgramGraphShape::new(4, 3),
                "source",
                "sink",
                "sanitizer",
                "clean",
                "reach",
                "alive",
                "hits",
                "out_scalar",
            ),
        ),
        (
            "flows_to_with_sanitizer/33n32e/renamed",
            flows_to_with_sanitizer(
                ProgramGraphShape::new(33, 32),
                "src_set",
                "sink_set",
                "san_set",
                "clean_scratch",
                "reach_scratch",
                "alive_scratch",
                "hit_scratch",
                "any_hit",
            ),
        ),
    ]
}

/// Pinned canonical IR fingerprints across the security flow-skeleton family.
///
/// Pinned post-merge against canonical wire hash updates (commits `811a42dabd`
/// and `16f1af5c07`), where `Program::canonical_wire_bytes` introduced
/// borrow-preserving canonicalization and transparent block-splicing across all
/// statement depths, normalizing nested block and traversal region structures.
const PINS: &[(&str, &str)] = &[
    (
        "flows_to/1n0e/fin-fout",
        "f2ccb970d416a0733fac5b2bb85d6eaa0a833e69e6a32127efa746225114f950",
    ),
    (
        "flows_to/4n3e/fin-fout",
        "0f81118e6062c8adf5602b1b90ba3b06a30ef723a6486eefadc2ed9aca9b7ab0",
    ),
    (
        "flows_to/32n31e/fin-fout",
        "69c54585ad6a7eea580d04e47fdb9708264a083be4791747c87decc05e17ae10",
    ),
    (
        "flows_to/33n32e/fin-fout",
        "e1ae43e164e6278a6b08de620a6733d32e2baca1f84919102e38ab3c4f71a346",
    ),
    (
        "flows_to/1024n4096e/renamed",
        "be5ff6ca49530b540a689067a8779628d313c1e9105e4117ae13509a4892478c",
    ),
    (
        "flows_to_alias_only/1n0e/fin-fout",
        "6d2a32672c4b12f82047503ccf2a29cc64e3d1a9e275bbf8ea016c121ba6e3a3",
    ),
    (
        "flows_to_alias_only/4n3e/fin-fout",
        "f0e9b8214b7ec82795643fba9f0edf5b6718cc7fdab083ec491000bb1344eed1",
    ),
    (
        "flows_to_alias_only/33n32e/renamed",
        "ae878d74e17e8204b3174328f7a9f6ecc35347a971c33016f2febfa31a69265f",
    ),
    (
        "taint_flow/1n0e/fin-fout",
        "70c43d105b6dbfc40c6f0a52b409719627df49ebe491d8a7888b3b003eee28a0",
    ),
    (
        "taint_flow/4n3e/fin-fout",
        "81306eb5024343361c4b99f729f398eaeb75fb24bf22c7835a0616a29552fe63",
    ),
    (
        "taint_flow/32n31e/fin-fout",
        "0cbb00305c0ff25bfe651d163e09d27464ba9b1a6c1d5fedb09064fbe98be3bd",
    ),
    (
        "taint_flow/33n32e/fin-fout",
        "b2461579c26ca079af48cb3fcf5cee8fd173c9a66163ee863869c6763bd7e8a6",
    ),
    (
        "taint_flow/1024n4096e/renamed",
        "4bdd9fa1f6d3da9e5c2080e81d421c9e26d35c0acacc8ea49cf358b67a818605",
    ),
    (
        "bounded_by_comparison/1n0e/fin-fout",
        "eedc3bbffff952a8191b1769e1457cdfb539eac2e47eec2135265b1ac6b93106",
    ),
    (
        "bounded_by_comparison/4n4e/fin-fout",
        "a3ab09b7563dfdab4acafda471a2eb1f042c2e5810214a2fb87a921b282492d8",
    ),
    (
        "bounded_by_comparison/32n31e/fin-fout",
        "0a65c05f39e28690633c52830d7ff4e9f6ae0aad68b165e72ee0af7186213df1",
    ),
    (
        "bounded_by_comparison/33n32e/fin-fout",
        "b2443ed84f319e559c720788085b21295815ed9f21f15859efdee618a6312185",
    ),
    (
        "bounded_by_comparison/1024n4096e/renamed",
        "4a43c69f0079cdebb4e037e9ebfd2b16a3f2c978d826ccedd00ddd189b2d284c",
    ),
    (
        "dominance_predecessors/1n0e/fin-fout",
        "b8dda9adde26f773c6e6f01a12798916eb16e4e71c9883bb2c76bb9969e7f295",
    ),
    (
        "dominance_predecessors/4n4e/fin-fout",
        "6d8395ae7fd85113355bf034ffe5f3b2111101b7c30450c10e539c8c92028a8a",
    ),
    (
        "dominance_predecessors/32n31e/fin-fout",
        "dac792ffa9f1736e48dc0ae8c523077deeb2661ec4930a79fdf849be685c88f3",
    ),
    (
        "dominance_predecessors/33n32e/fin-fout",
        "3c71f322c03592497780c1b5cf09459b8d01cdc666f2ca4d669a3f2f42341004",
    ),
    (
        "dominance_predecessors/1024n4096e/renamed",
        "40055c8ff73b3ba07db6d2a5050b83e950121d37be564a64edad582d7e59378c",
    ),
    (
        "flows_to_to_sink/1n0e",
        "5b3cef28e2d89c08414c5a84f3a1681734e5d216d5d7ad808d2e4a2daf0f8283",
    ),
    (
        "flows_to_to_sink/4n3e",
        "d95a648a54b985e3c4cd245c73e84f45ec278a5391493b787c344a24c9ae8ca9",
    ),
    (
        "flows_to_to_sink/33n32e/renamed",
        "eeccd28c95914074d9a5904ff15cb8601369e9f60dc7e07d884d7e411ebe278f",
    ),
    (
        "taint_pollution/1n0e",
        "da9abb7e4d90c379fbdd1d3ec927e15ccaae433a05d5e991b55dd5548a66182e",
    ),
    (
        "taint_pollution/4n3e",
        "312fd35b1b15d8fd6c1edb8d591907383ab464001fce660e3858aa45209b769d",
    ),
    (
        "taint_pollution/33n32e/renamed",
        "2fd977b49abd339183c2b55fe680703b3ae91eb0f79fe6b35bbc2107c7bf4347",
    ),
    (
        "flows_to_with_sanitizer/1n0e",
        "9d3bcb68a903c62ca15918604bb7477457a734ca002624995b2a11785dbbbecb",
    ),
    (
        "flows_to_with_sanitizer/4n3e",
        "348af72d975ef8a9ed31cea296660a746acd3588394a42928cd3bf6272bbcf64",
    ),
    (
        "flows_to_with_sanitizer/33n32e/renamed",
        "f4101950d89d6d67f8ebc79b8f8768c4ef45a046df08f8a300e95f640285bf02",
    ),
];

#[test]
fn security_flow_family_entry_point_fingerprints_are_pinned() {
    let cases = pinned_cases();
    assert_eq!(
        cases.len(),
        PINS.len(),
        "every built case needs exactly one pin"
    );
    let mut drift = Vec::new();
    for ((label, program), (pin_label, expected)) in cases.iter().zip(PINS) {
        assert_eq!(label, pin_label, "pin table order must match case order");
        let actual = hex(program.fingerprint());
        if actual != *expected {
            drift.push(format!("    (\"{label}\", \"{actual}\"),"));
        }
    }
    assert!(
        drift.is_empty(),
        "security flow-skeleton family IR moved. Observed fingerprints:\n{}",
        drift.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Cross-entry-point behavior. Each rehomed file may own its predicate and
// nothing else, so the predicate is what these assertions pin.
// ---------------------------------------------------------------------------

/// A four-node graph carrying one edge of each interesting kind out of node 0:
///
/// ```text
///   0 --kinds[0]--> 1
///   0 --kinds[1]--> 2
///   0 --kinds[2]--> 3
/// ```
fn graph_bytes(kinds: [u32; 3]) -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("pg_nodes", vec![0, 0, 0, 0]),
        ("pg_edge_offsets", vec![0, 3, 3, 3, 3]),
        ("pg_edge_targets", vec![1, 2, 3]),
        ("pg_edge_kind_mask", kinds.to_vec()),
        ("pg_node_tags", vec![0, 0, 0, 0]),
    ]
}

/// Feed a program in its own declared buffer order, passing values for
/// non-backend-allocated reference inputs matching `vyre_reference::is_reference_input`.
fn eval(program: &Program, named: &[(&str, Vec<u32>)]) -> Vec<Vec<u32>> {
    let values: Vec<Value> = program
        .buffers()
        .iter()
        .filter(|decl| vyre_reference::is_reference_input(decl))
        .map(|decl| {
            let words = named
                .iter()
                .find(|(name, _)| *name == decl.name())
                .map_or_else(
                    || vec![0u32; decl.count() as usize],
                    |(_, words)| words.clone(),
                );
            Value::from(pack_u32_slice(&words))
        })
        .collect();
    vyre_reference::reference_eval(program, &values)
        .expect("Fix: security family guard program must evaluate")
        .iter()
        .map(|value| decode_u32_le_bytes_all(&value.to_bytes()))
        .collect()
}

fn reach_inputs(kinds: [u32; 3], frontier: u32) -> Vec<(&'static str, Vec<u32>)> {
    let mut inputs = graph_bytes(kinds);
    inputs.push(("fin", vec![frontier]));
    inputs.push(("fout", vec![frontier]));
    inputs
}

#[test]
fn flows_to_and_taint_flow_declare_the_same_forward_predicate() {
    let kinds = [
        edge_kind::ASSIGNMENT,
        edge_kind::CALL_ARG,
        edge_kind::CONTROL,
    ];
    let shape = ProgramGraphShape::new(4, 3);
    let inputs = reach_inputs(kinds, 0b0001);
    let left = eval(&flows_to(shape, "fin", "fout"), &inputs);
    let right = eval(&taint_flow(shape, "fin", "fout"), &inputs);
    assert_eq!(
        left, right,
        "flows_to and taint_flow are one skeleton under two op ids; \
         their reached sets must be identical"
    );
    // Dataflow reaches the ASSIGNMENT and CALL_ARG neighbours, never CONTROL.
    assert_eq!(left[0][0], 0b0111, "FLOWS_TO_MASK must exclude CONTROL");
}

#[test]
fn flows_to_alias_only_narrows_the_forward_predicate_to_aliases() {
    let kinds = [
        edge_kind::ASSIGNMENT,
        edge_kind::CALL_ARG,
        edge_kind::CONTROL,
    ];
    let shape = ProgramGraphShape::new(4, 3);
    let inputs = reach_inputs(kinds, 0b0001);
    let dataflow = eval(&flows_to(shape, "fin", "fout"), &inputs);
    let alias = eval(&flows_to_alias_only(shape, "fin", "fout"), &inputs);
    assert_eq!(
        alias[0][0], 0b0011,
        "ALIAS_PROPAGATION_MASK must drop the CALL_ARG neighbour \
         (the `char *copy = strdup(msg)` false positive)"
    );
    assert_ne!(
        dataflow, alias,
        "the alias predicate is a strict subset; a shared skeleton must not \
         collapse the two masks"
    );
}

fn hit_inputs(source: u32, sink: u32) -> Vec<(&'static str, Vec<u32>)> {
    let mut inputs = graph_bytes([
        edge_kind::ASSIGNMENT,
        edge_kind::CALL_ARG,
        edge_kind::CONTROL,
    ]);
    inputs.push(("source", vec![source]));
    inputs.push(("reach", vec![source]));
    inputs.push(("sink", vec![sink]));
    inputs.push(("hits", vec![0]));
    inputs.push(("out_scalar", vec![0]));
    inputs
}

#[test]
fn flows_to_to_sink_and_taint_pollution_declare_the_same_hit_predicate() {
    let shape = ProgramGraphShape::new(4, 3);
    for (source, sink) in [
        (0b0001u32, 0b0010u32), // one dataflow hop lands on the sink
        (0b0001, 0b1000),       // only a CONTROL neighbour is tagged: no hit
        (0b0000, 0b1111),       // empty source
        (0b0001, 0b0000),       // empty sink
        (0b1111, 0b1111),       // saturated
    ] {
        let inputs = hit_inputs(source, sink);
        let left = eval(
            &flows_to_to_sink(shape, "source", "sink", "reach", "hits", "out_scalar"),
            &inputs,
        );
        let right = eval(
            &taint_pollution(shape, "source", "sink", "reach", "hits", "out_scalar"),
            &inputs,
        );
        assert_eq!(
            left, right,
            "flows_to_to_sink and taint_pollution supply the same predicate; \
             source={source:#06b} sink={sink:#06b} must agree"
        );
    }
}

#[test]
fn flows_to_to_sink_reports_a_reached_sink_and_stays_quiet_otherwise() {
    let shape = ProgramGraphShape::new(4, 3);
    let program = flows_to_to_sink(shape, "source", "sink", "reach", "hits", "out_scalar");
    let hit = eval(&program, &hit_inputs(0b0001, 0b0010));
    assert_eq!(
        hit.last().expect("out_scalar")[0],
        1,
        "a dataflow neighbour tagged as a sink is a hit"
    );
    let miss = eval(&program, &hit_inputs(0b0001, 0b1000));
    assert_eq!(
        miss.last().expect("out_scalar")[0],
        0,
        "a CONTROL-only neighbour tagged as a sink is not a hit"
    );
}

#[test]
fn sanitizer_projection_removes_sanitized_nodes_from_the_hit() {
    let shape = ProgramGraphShape::new(4, 3);
    let program = flows_to_with_sanitizer(
        shape,
        "source",
        "sink",
        "sanitizer",
        "clean",
        "reach",
        "alive",
        "hits",
        "out_scalar",
    );
    let base = |sanitizer: u32| {
        let mut inputs = graph_bytes([
            edge_kind::ASSIGNMENT,
            edge_kind::CALL_ARG,
            edge_kind::CONTROL,
        ]);
        inputs.push(("source", vec![0b0001]));
        inputs.push(("sanitizer", vec![sanitizer]));
        inputs.push(("clean", vec![0]));
        inputs.push(("reach", vec![0]));
        inputs.push(("alive", vec![0]));
        inputs.push(("sink", vec![0b0010]));
        inputs.push(("hits", vec![0]));
        inputs.push(("out_scalar", vec![0]));
        inputs
    };
    assert_eq!(
        eval(&program, &base(0)).last().expect("out_scalar")[0],
        1,
        "with no sanitizer the projection degenerates to the plain hit"
    );
    assert_eq!(
        eval(&program, &base(0b0010)).last().expect("out_scalar")[0],
        0,
        "sanitizing the sink node kills the hit"
    );
    assert_eq!(
        eval(&program, &base(0b0001)).last().expect("out_scalar")[0],
        0,
        "sanitizing the source node kills the hit before the walk starts"
    );
}

#[test]
fn backward_pair_agrees_on_a_dominance_only_graph() {
    let shape = ProgramGraphShape::new(4, 3);
    let inputs = reach_inputs([edge_kind::DOMINANCE; 3], 0b1000);
    let bounded = eval(&bounded_by_comparison(shape, "fin", "fout"), &inputs);
    let predecessors = eval(&dominance_predecessors(shape, "fin", "fout"), &inputs);
    assert_eq!(
        bounded, predecessors,
        "on a DOMINANCE-only graph the two backward predicates coincide"
    );
    assert_eq!(
        bounded[0][0], 0b1001,
        "a backward dominance step from {{3}} keeps the seed and reaches {{0}}"
    );
}

#[test]
fn backward_pair_diverges_exactly_on_block_member_edges() {
    let shape = ProgramGraphShape::new(4, 3);
    // Only the edge into node 3 is BLOCK_MEMBER, which is in the
    // dominance_predecessors mask and not in the bounded_by_comparison mask.
    let kinds = [
        edge_kind::DOMINANCE,
        edge_kind::DOMINANCE,
        edge_kind::BLOCK_MEMBER,
    ];
    let inputs = reach_inputs(kinds, 0b1000);
    let bounded = eval(&bounded_by_comparison(shape, "fin", "fout"), &inputs);
    let predecessors = eval(&dominance_predecessors(shape, "fin", "fout"), &inputs);
    assert_eq!(
        bounded[0][0], 0b1000,
        "bounded_by_comparison walks DOMINANCE only, so a BLOCK_MEMBER edge \
         into {{3}} reaches nothing"
    );
    assert_eq!(
        predecessors[0][0], 0b1001,
        "dominance_predecessors also walks BLOCK_MEMBER, so it reaches {{0}}"
    );
    assert_ne!(
        bounded, predecessors,
        "the backward pair must keep two distinct edge masks after the rehome"
    );
}
