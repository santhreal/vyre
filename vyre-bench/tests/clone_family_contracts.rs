//! Permanent guard for the benchmark clone families collapsed onto the
//! case-builder harness.
//!
//! WHY: five clone families under `vyre-bench/src/cases` each carried their own
//! copy of the same case scaffolding (id, metadata, suites, requirements,
//! contract, downcast plumbing, metric assembly). Collapsing them onto one
//! harness rewrites every one of those declarations. The defect class that
//! rewrite can introduce is a case that is dropped, renamed, reordered, or
//! whose description is silently retyped. This file pins the observable case
//! surface so any member of that class is RED.
//!
//! Coverage is derived from the registry at run time, never from a hardcoded
//! count: adding or removing a case turns this suite red until the pinned
//! enumeration records the decision.
//!
//! What it does NOT catch: anything only observable by dispatching a program on
//! a device. Program identity for the ctx-free builders is pinned in-crate by
//! `crate::cases::clone_family_guard`.

use std::collections::BTreeSet;

use vyre_bench::api::case::BenchCase;

/// Every benchmark id the registry publishes, in registry order.
///
/// Three `frontend.c.parser*` cases were removed when the C frontend crate left
/// the workspace. They measured that crate's whole-pipeline parse, and no
/// surviving crate publishes an equivalent. `release.ast_motif_traversal.1m` measures
/// the traversal predicate stage the release plan requires, over generated node
/// columns rather than a parse.
///
/// The three `frontend.rust.*` cases went the same way in a2e24fcdf1, which
/// moved the Rust front end to `software/frontend-rust` and took the case tree
/// that existed to exercise it. `parser.c_lexer.small_state_transition.4k`
/// followed when the C frontend itself left `vyre-libs`: it measured that
/// frontend's lexer state-transition pass and there is no builder left to
/// build it. No parsing workload remains; `release.ast_motif_traversal.1m` is a
/// synthetic traversal with no frontend dependency.
const EXPECTED_CASE_IDS: &[&str] = &[
    "adversarial.register_exhaustion.u32_1024",
    "bigint.modexp.4096",
    "callgraph.reachability.step.262k",
    "compound.pipeline.fused_filter.1m",
    "conditions.yara_like.batch.16x64k",
    "conditions.yara_like.eval.1m",
    "crypto.aes_ctr.encrypt.10mb",
    "cuda.ptx.patterns.release.corpus",
    "dataflow.ifds.skewed.closure.1m",
    "dataflow.ifds.skewed.queue_closure.1m",
    "dataflow.ifds.skewed.queue_materialize_step.1m",
    "dataflow.ifds.skewed.queue_step.1m",
    "dataflow.ifds.skewed.step.1m",
    "foundation.attention.64",
    "foundation.dfa_match.256k",
    "foundation.elementwise.add.1m",
    "foundation.gather.u32.1m",
    "foundation.histogram.u32_256.1m",
    "foundation.matmul.256",
    "foundation.optimizer.impact",
    "foundation.reduce.sum.crossover",
    "foundation.stencil3.u32.1m",
    "foundation.transpose.512",
    "hashtable.openaddr.probe.10m",
    "interpreter.bytecode.dispatch.10m",
    "metadata.condition.filesize_header.1m",
    "nn.linear_4bit_affine_grouped.1m",
    "primitives.graph.csr_skewed_frontier.1m",
    "primitives.graph.csr_skewed_queue_closure.1m",
    "primitives.graph.csr_skewed_queue_materialize.1m",
    "primitives.graph.frontier_step.1m",
    "regex.backtracking.adversarial",
    "release.alias_reaching_def.1m",
    "release.ast_motif_traversal.1m",
    "release.condition_eval.1m",
    "release.egraph_saturation.1m",
    "release.entropy_window.1m",
    "release.ifds_witness.1m",
    "release.megakernel_queue.1m",
    "release.offset_count_aggregation.1m",
    "release.quantified_condition_loops.1m",
    "release.string_bitmap_scatter.1m",
    "runtime.adaptive_routing.gpu_resident.1m",
    "runtime.megakernel.condition.64k",
    "runtime.megakernel.dispatch.256",
    "runtime.megakernel.truth.1024",
    "runtime.nvme_gpu_ingest.gpudirect_nvme.64g",
    "runtime.nvme_gpu_ingest.registered_mapped.4g",
    "scan.ac.irregular_count.4m",
    "scan.ac.irregular_literals.4m",
    "search.binary.u32.1m",
    "sparse.compaction.count.1m",
    "synthetic.flaky",
];

/// The cases owned by the clone families this campaign collapsed, plus the
/// neighbours that share the merged queue-stage owner.
///
/// The seven `foundation.*` micro cases joined when they were collapsed onto
/// `crate::cases::micro`, which took over their identity, metadata, dispatch,
/// reference timing and run assembly. Their surface must survive that move
/// unchanged, which is exactly what the digest below asserts.
const FAMILY_CASE_IDS: &[&str] = &[
    "compound.pipeline.fused_filter.1m",
    "conditions.yara_like.batch.16x64k",
    "conditions.yara_like.eval.1m",
    "dataflow.ifds.skewed.closure.1m",
    "dataflow.ifds.skewed.queue_closure.1m",
    "dataflow.ifds.skewed.queue_materialize_step.1m",
    "dataflow.ifds.skewed.queue_step.1m",
    "dataflow.ifds.skewed.step.1m",
    "foundation.attention.64",
    "foundation.dfa_match.256k",
    "foundation.gather.u32.1m",
    "foundation.histogram.u32_256.1m",
    "foundation.matmul.256",
    "foundation.stencil3.u32.1m",
    "foundation.transpose.512",
    "primitives.graph.csr_skewed_frontier.1m",
    "primitives.graph.csr_skewed_queue_closure.1m",
    "primitives.graph.csr_skewed_queue_materialize.1m",
    "primitives.graph.frontier_step.1m",
    "runtime.adaptive_routing.gpu_resident.1m",
    "runtime.megakernel.condition.64k",
    "runtime.megakernel.dispatch.256",
];

/// blake3 over the compact JSON surface of every `FAMILY_CASE_IDS` member.
/// Regenerate only with a recorded decision about what changed and why.
///
/// Last changed when the seven `foundation.*` micro cases were added to the
/// family list. Their own surfaces are byte-identical across that collapse,
/// which `registry_publishes_exactly_the_pinned_case_enumeration` and the
/// pre-collapse program fingerprints in `crate::cases::micro` both hold; the
/// digest moved only because the list it covers grew.
const FAMILY_SURFACE_DIGEST: &str =
    "62966bdad8dfdc5e5e738f67ad4bb25dbcd7e98ecc266aa5696080f1e74d9955";

fn registry_ids() -> Vec<String> {
    vyre_bench::registry::collect_all()
        .iter()
        .map(|case| case.id().0)
        .collect()
}

fn case_surface(case: &'static dyn BenchCase) -> serde_json::Value {
    serde_json::json!({
        "metadata": case.metadata(),
        "suites": case.suites(),
        "requirements": case.requirements(),
        "performance_contract": case.performance_contract(),
    })
}

/// The hard acceptance criterion of the deduplication: the published case set
/// and its order survive the harness migration byte for byte.
#[test]
fn registry_publishes_exactly_the_pinned_case_enumeration() {
    let actual = registry_ids();
    let pinned: Vec<String> = EXPECTED_CASE_IDS
        .iter()
        .map(|id| (*id).to_string())
        .collect();

    let published: BTreeSet<&String> = actual.iter().collect();
    let recorded: BTreeSet<&String> = pinned.iter().collect();
    let unrecorded: Vec<&&String> = published.difference(&recorded).collect();
    let vanished: Vec<&&String> = recorded.difference(&published).collect();

    assert!(
        unrecorded.is_empty() && vanished.is_empty(),
        "Fix: the benchmark case enumeration changed. Cases {unrecorded:?} are published and not \
         recorded in EXPECTED_CASE_IDS; cases {vanished:?} are recorded and no longer published. \
         Record each one with the decision that justifies it.",
    );
    assert_eq!(
        actual, pinned,
        "Fix: the published case set matches EXPECTED_CASE_IDS but the order differs. Registry \
         order is observable, so reorder the pin to the order the registry publishes."
    );
}

/// Ids must stay unique and sorted, so a harness that registers one description
/// twice cannot hide behind a stable count.
#[test]
fn registry_case_ids_are_unique_and_ordered() {
    let actual = registry_ids();
    let mut sorted = actual.clone();
    sorted.sort();
    sorted.dedup();

    assert_eq!(
        actual, sorted,
        "Fix: benchmark ids must be unique and registry order must stay sorted."
    );
}

/// Every clone-family member must still be registered. Derived from the family
/// list rather than a count so a dropped case cannot pass.
#[test]
fn every_clone_family_case_is_still_registered() {
    let registry = vyre_bench::registry::collect_all();
    let missing: Vec<&str> = FAMILY_CASE_IDS
        .iter()
        .copied()
        .filter(|id| {
            registry
                .get(&vyre_bench::api::case::BenchId((*id).to_string()))
                .is_none()
        })
        .collect();

    assert!(
        missing.is_empty(),
        "Fix: the harness migration dropped clone-family cases: {missing:?}"
    );
}

/// The full ctx-free surface of every clone-family case, pinned as one digest.
/// A mistyped tag, a lost suite, a dropped `min_vram_bytes`, or a contract that
/// silently disappeared all land here.
#[test]
fn clone_family_case_surface_is_pinned() {
    let registry = vyre_bench::registry::collect_all();
    let surfaces: Vec<serde_json::Value> = FAMILY_CASE_IDS
        .iter()
        .map(|id| {
            let case = registry
                .get(&vyre_bench::api::case::BenchId((*id).to_string()))
                .unwrap_or_else(|| panic!("Fix: clone-family case `{id}` is no longer registered"));
            case_surface(case)
        })
        .collect();
    let encoded = serde_json::to_string(&surfaces).expect("Fix: case surface must serialize");
    let digest = blake3::hash(encoded.as_bytes()).to_hex().to_string();

    assert_eq!(
        digest, FAMILY_SURFACE_DIGEST,
        "Fix: a clone-family case surface changed. Surface was:\n{encoded}"
    );
}
