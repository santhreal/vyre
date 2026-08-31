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
/// A fingerprint is a hash of canonical wire bytes, so every one of these moves
/// when the wire encoding of a construct these programs contain changes. Wire
/// revision 8 re-encoded the synchronization variants this family uses, which
/// moved all of them at once. `PINNED_WIRE_FORMAT_VERSION` is asserted before
/// any digest is compared, so the next revision reports itself instead of
/// arriving as an unexplained table of mismatched hashes.
const PINNED_WIRE_FORMAT_VERSION: u16 = 8;

const PINS: &[(&str, &str)] = &[
    (
        "flows_to/1n0e/fin-fout",
        "a9157bdd8b8bcd47d16d40000d24bc0447a9f302c2496a98a832481a67c341a5",
    ),
    (
        "flows_to/4n3e/fin-fout",
        "91ac3174011e31e031d49b1bbae58af0671e8c474a575bd69a37b865c2fda04d",
    ),
    (
        "flows_to/32n31e/fin-fout",
        "186196fda33f85acac2904ce51518a963adef9069c38f07393236a4b10818624",
    ),
    (
        "flows_to/33n32e/fin-fout",
        "28fdba298355590047cc6418b3e219a40c6c65bb17ded223b6b7f226fb08816f",
    ),
    (
        "flows_to/1024n4096e/renamed",
        "22859bffe1a46338f524e352c18c705eeb8e781c2a82fdf6d4227de79e7772f2",
    ),
    (
        "flows_to_alias_only/1n0e/fin-fout",
        "d01f683c755870dda5fdc7509046e83de7a02c8044228e539317ff7553dc26a4",
    ),
    (
        "flows_to_alias_only/4n3e/fin-fout",
        "6d7ccca0c4c6c36ae7368644be18596ee1b6a05ca72a4b55bb7a1358ac815563",
    ),
    (
        "flows_to_alias_only/33n32e/renamed",
        "e01a88e545b51a1a2989cfb143438d031da99c7f92759fd5b99b1db7c12d8380",
    ),
    (
        "taint_flow/1n0e/fin-fout",
        "4e4a2e6124045ee212e01d618282c8cab7a5d801311f36d2542b817cee46a10c",
    ),
    (
        "taint_flow/4n3e/fin-fout",
        "c3c92a83c191390fa8e80ba2f9a6882562fc052f6ede4a01d3111e756185931f",
    ),
    (
        "taint_flow/32n31e/fin-fout",
        "4b5660c07f063ee6ef222abe5051dacb0633cdb3351443c4f1058d5e217746ab",
    ),
    (
        "taint_flow/33n32e/fin-fout",
        "8103e30827e82fe9a09a77273069ce975222ff9a70a269b59676ae95ff514a8c",
    ),
    (
        "taint_flow/1024n4096e/renamed",
        "5e1d425ea358539c8592c80e1b164c071860cdc98027de10601e8a92c405c4ae",
    ),
    (
        "bounded_by_comparison/1n0e/fin-fout",
        "2d8a7b84c694b542c20a505687aa5f0a5415bd4b53e3b7387955443c8ab39c22",
    ),
    (
        "bounded_by_comparison/4n4e/fin-fout",
        "1f94ef30b861eb9571bc6132d8a7fadf486faad4c0085ea31658242c10864849",
    ),
    (
        "bounded_by_comparison/32n31e/fin-fout",
        "4b9773270d9a0b986a922c42ba3e2b2b146531ff65a08763edcb03462fb2ca2d",
    ),
    (
        "bounded_by_comparison/33n32e/fin-fout",
        "a85167f2b35dcde03f595852516c8c9cd792187c7cd1f622507d636bb7ea41b7",
    ),
    (
        "bounded_by_comparison/1024n4096e/renamed",
        "11b6037fb19eea0b20505ae4a87dc6b92960435b628d5e3b5d39c0f78fdf96fa",
    ),
    (
        "dominance_predecessors/1n0e/fin-fout",
        "f22c962aa40efb8e9e37f9bfff50925c6e1d9a6a151a3d6b40b5efb8752c567d",
    ),
    (
        "dominance_predecessors/4n4e/fin-fout",
        "a1b4761e86d1456692b32bd4d6a2115f539c8728760c768ffd6f551efd9e0a57",
    ),
    (
        "dominance_predecessors/32n31e/fin-fout",
        "a537ed7602c4df8433a195f121767bddcc8eb7d7ee76a10a423a3a34a01f671d",
    ),
    (
        "dominance_predecessors/33n32e/fin-fout",
        "92a2edb14f0bbc863860aacb3749089bd78426437d9b14514f9d76a7535153d9",
    ),
    (
        "dominance_predecessors/1024n4096e/renamed",
        "66a6511eaf79440c3c22e87d6d669c0e61644ff88778f65705ba8527d6158886",
    ),
    (
        "flows_to_to_sink/1n0e",
        "1fafcc20c2d1ab8820e5beee6890ab1ce357801c5ed520d243043b87317566a3",
    ),
    (
        "flows_to_to_sink/4n3e",
        "b35638d2159645b48930542c4114a6625d0a9df688e9a8e780809fcc2dc73e61",
    ),
    (
        "flows_to_to_sink/33n32e/renamed",
        "1519e349fdfd8824892a257cd88aed68c5039dcd041f9236360d302919f1a86d",
    ),
    (
        "taint_pollution/1n0e",
        "7be14dfb744fb304a5283a565677779889affe79172bfb11dc35a8afab0adaf2",
    ),
    (
        "taint_pollution/4n3e",
        "5f7cf8d1829ec7871ed633666032e6076d2e34a7411cffe7688abd053fe2a8e1",
    ),
    (
        "taint_pollution/33n32e/renamed",
        "5a569a46c164da214e1d53a80a921b06b647da7e3c2371b9f5c65e2ae714c60b",
    ),
    (
        "flows_to_with_sanitizer/1n0e",
        "91a0b80a9e1570916fdd463db9a97b1a955870df9a6e328ff7564d6aba7985a0",
    ),
    (
        "flows_to_with_sanitizer/4n3e",
        "de1d88aec0cd812124591ad3eb5dbe9a03183cc99017e73eb221ca365f510b22",
    ),
    (
        "flows_to_with_sanitizer/33n32e/renamed",
        "d5e78f1b8039317cc90ea31c24d0def6fda666e46449f134af62c03f1f9755c8",
    ),
];

#[test]
fn security_flow_family_entry_point_fingerprints_are_pinned() {
    assert_eq!(
        vyre_foundation::serial::wire::framing::WIRE_FORMAT_VERSION,
        PINNED_WIRE_FORMAT_VERSION,
        "Fix: the wire encoding moved, so every fingerprint below is stale. Re-pin the table against revision {} and record what the revision changed.",
        vyre_foundation::serial::wire::framing::WIRE_FORMAT_VERSION
    );
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
