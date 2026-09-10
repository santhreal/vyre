//! What an ahead-of-time compile carries from the caller's request.
//!
//! WHY: a product wrapper that states its own external facts, device facts or
//! search budget compiles something the caller never asked for, and the
//! artifact it produces is authenticated for a request that does not exist. A
//! zero facts digest gives every set of facts one artifact identity, an unknown
//! device fact vector gives every device one, and a one-candidate budget records
//! a search nobody bounded that way. Each case below compiles requests that
//! differ in exactly one of those fields and asserts the difference reaches the
//! artifact, so substituting a placeholder for any of them collapses a pair that
//! must stay apart.
//!
//! The source scan at the end closes the class rather than the three incidents:
//! it reads this crate's own source at run time and fails on any compile input
//! constructed inside it, so a fourth placeholder fails the day it is written.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use vyre_aot::{compile, ValidatedCompileRequest};
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::{
    CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric,
    SearchBudget,
};
use vyre_test_support::artifact_fixtures::params_and_out_graph;

use crate::fixture_target;

/// The facts digest a caller states for the checkpoint and configuration its
/// program was built against.
const STATED_FACTS: Digest = Digest([0x55; 32]);

/// A second caller's facts over the same graph.
const OTHER_FACTS: Digest = Digest([0xa1; 32]);

/// The digest a wrapper substituted when it stated no facts of its own.
const ZEROED_FACTS: Digest = Digest([0; 32]);

/// A budget wide enough for the search to rank more than one candidate.
const RANKED_BUDGET: SearchBudget = SearchBudget::new(8, 1_000, 2, 0, 10_000_000);

/// The budget a wrapper hardcoded: one candidate, so nothing is ranked.
const ONE_CANDIDATE_BUDGET: SearchBudget = SearchBudget::new(1, 1_000, 1, 0, 10_000_000);

/// The live facts a caller holding a backend passes.
fn live_device_facts() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 1_024)
        .with_compute_units(80)
        .with_calibration_version(3)
}

/// One validated request over [`fixture_graph`], varying only the three fields
/// a wrapper used to state for itself.
fn request(
    facts_digest: Digest,
    device: DeviceFacts,
    search_budget: SearchBudget,
) -> ValidatedCompileRequest {
    CompileRequest::new(
        params_and_out_graph(),
        ExternalFacts::new(facts_digest, BTreeMap::new()),
        device,
        search_budget,
        CompileObjective::minimize_latency()
            .with_bound(ObjectiveMetric::ArtifactBytes, 64 * 1024 * 1024),
    )
    .validate()
    .expect("the fixture request must validate. Fix: correct the graph, facts or bounds above")
}

/// The request identity the ahead-of-time path recorded for `request`.
fn aot_request_identity(request: &ValidatedCompileRequest) -> Digest {
    compile(request, fixture_target::fixture_target())
        .expect("the fixture target must compile the fixture request")
        .neutral()
        .provenance()
        .request
}

/// WHY: the external-facts digest authenticates what the program was built
/// against, and it is an input to artifact identity. A wrapper substituting a
/// zero digest gives two callers holding different checkpoints one artifact, so
/// a consumer verifying its facts against the artifact it received cannot tell
/// them apart.
#[test]
fn the_external_facts_digest_the_caller_states_reaches_the_artifact() {
    let stated = aot_request_identity(&request(STATED_FACTS, live_device_facts(), RANKED_BUDGET));
    let other = aot_request_identity(&request(OTHER_FACTS, live_device_facts(), RANKED_BUDGET));
    let zeroed = aot_request_identity(&request(ZEROED_FACTS, live_device_facts(), RANKED_BUDGET));

    assert_ne!(
        stated, other,
        "two callers stating different external facts must not receive one artifact identity"
    );
    assert_ne!(
        stated, zeroed,
        "an artifact compiled under stated facts must not carry the identity of one compiled under a zero digest"
    );

    let direct = vyre_megakernel::compile(&request(
        STATED_FACTS,
        live_device_facts(),
        RANKED_BUDGET,
    ))
    .expect("the same request must compile directly");
    assert_eq!(
        stated,
        direct.provenance().request,
        "the ahead-of-time path and a direct compile of one request must record one request identity"
    );
}

/// WHY: the plan is selected and priced against the device the caller named,
/// and the device projection is an input to artifact identity so a cache cannot
/// serve an artifact compiled for another device. A wrapper substituting
/// `DeviceFacts::unknown()` compiles every device's request as the device-neutral
/// one, and a recalibration then goes unnoticed because the facts it changed
/// were discarded before the compiler read them.
#[test]
fn the_device_facts_the_caller_names_reach_the_artifact() {
    let live = aot_request_identity(&request(STATED_FACTS, live_device_facts(), RANKED_BUDGET));
    let unknown =
        aot_request_identity(&request(STATED_FACTS, DeviceFacts::unknown(), RANKED_BUDGET));
    let recalibrated = aot_request_identity(&request(
        STATED_FACTS,
        live_device_facts().with_calibration_version(4),
        RANKED_BUDGET,
    ));

    assert_ne!(
        live, unknown,
        "an artifact compiled against a live capability snapshot must not carry the identity of a device-neutral compile"
    );
    assert_ne!(
        live, recalibrated,
        "a recalibrated device prices every candidate differently, so it must not share an artifact identity with the calibration it replaced"
    );

    let direct = vyre_megakernel::compile(&request(
        STATED_FACTS,
        live_device_facts(),
        RANKED_BUDGET,
    ))
    .expect("the same request must compile directly");
    assert_eq!(
        live,
        direct.provenance().request,
        "the ahead-of-time path must select against the caller's device facts, not its own"
    );
}

/// WHY: the artifact records the bounds its search ran under, and a consumer
/// reads that record to state how much of the space was examined. A wrapper
/// forcing one candidate records a bound the caller never stated, and the
/// unfused baseline is then the only plan that was ever ranked.
#[test]
fn the_search_budget_the_caller_states_is_the_budget_the_artifact_records() {
    let ranked = compile(
        &request(STATED_FACTS, live_device_facts(), RANKED_BUDGET),
        fixture_target::fixture_target(),
    )
    .expect("the ranked request must compile");
    let single = compile(
        &request(STATED_FACTS, live_device_facts(), ONE_CANDIDATE_BUDGET),
        fixture_target::fixture_target(),
    )
    .expect("the one-candidate request must compile");

    assert_eq!(
        ranked.neutral().selected_plan().search_budget,
        RANKED_BUDGET,
        "the artifact must record the bounds the caller stated"
    );
    assert_eq!(
        single.neutral().selected_plan().search_budget,
        ONE_CANDIDATE_BUDGET,
        "a caller that states one candidate must get an artifact recording one candidate"
    );
    assert_ne!(
        ranked.neutral().provenance().request,
        single.neutral().provenance().request,
        "two requests bounded differently must not share one artifact identity"
    );
    assert!(
        ranked.neutral().selected_plan().search_work.candidates_explored
            <= RANKED_BUDGET.max_candidates,
        "the recorded work must stay inside the recorded bound"
    );
}

/// Constructors that would state a compile input this crate has no business
/// stating. Each one is the exact shape of a fabricated default.
const FABRICATED_COMPILE_INPUTS: &[&str] = &[
    "CompileRequest::new",
    "ExternalFacts::new",
    "DeviceFacts::new",
    "DeviceFacts::unknown",
    "SearchBudget::new",
    "CompileObjective::",
    "MeshFacts::",
];

/// WHY: the three cases above prove three placeholders are gone. This one
/// closes the class they belong to: every compile input is the caller's, so
/// this crate constructs none of them. The file set and the signatures are read
/// out of the crate's own source at run time, so a fourth placeholder, or a new
/// entry point that builds its own request, fails the day it is written rather
/// than the day someone re-reads the file.
#[test]
fn the_crate_constructs_no_compile_input_of_its_own() {
    let sources = crate_source_files();
    assert!(
        sources.len() >= 2,
        "the source scan found {} files under vyre-aot/src; an almost empty set is a broken scan, not a clean crate",
        sources.len()
    );
    assert!(
        sources
            .iter()
            .any(|(_, text)| text.contains("ValidatedCompileRequest")),
        "the source scan read no file naming ValidatedCompileRequest, so it is not reading this crate"
    );

    for (path, text) in &sources {
        for input in FABRICATED_COMPILE_INPUTS {
            assert!(
                !text.contains(input),
                "{} constructs `{input}`. Fix: take the value from the caller's ValidatedCompileRequest instead of stating one here",
                path.display()
            );
        }
    }
}

/// WHY: a second entry point that takes a `Program` and builds its own request
/// is the defect returning under another name. Every public function that
/// returns an artifact envelope for a named target must consume a validated
/// request, and the set is derived from source so adding one is what turns this
/// red.
#[test]
fn every_public_target_compile_entry_point_consumes_a_validated_request() {
    let sources = crate_source_files();
    let mut entry_points = Vec::new();
    for (path, text) in &sources {
        for signature in public_signatures(text) {
            let Some((parameters, returns)) = signature.rsplit_once(") ->") else {
                continue;
            };
            if !returns.contains("ArtifactEnvelope") || !parameters.contains("TargetId") {
                continue;
            }
            entry_points.push((path.clone(), signature.clone()));
        }
    }

    assert!(
        !entry_points.is_empty(),
        "the scan found no public entry point returning an artifact envelope for a named target; an empty set is a broken scan, not a crate with no compile path"
    );
    for (path, signature) in &entry_points {
        assert!(
            signature.contains("ValidatedCompileRequest"),
            "{} declares `{signature}`, which produces an artifact envelope without a validated compile request. Fix: accept the caller's request",
            path.display()
        );
    }
}

/// Every Rust source file of this crate's library, with its text.
fn crate_source_files() -> Vec<(PathBuf, String)> {
    let root = vyre_test_support::monorepo::vyre_crate_directory("vyre-aot").join("src");
    let mut files = Vec::new();
    collect_rust_files(&root, &mut files);
    files.sort_by(|left, right| left.0.cmp(&right.0));
    files
}

/// Append every `.rs` file under `directory`, recursively.
fn collect_rust_files(directory: &Path, into: &mut Vec<(PathBuf, String)>) {
    let entries = fs::read_dir(directory).unwrap_or_else(|error| {
        panic!(
            "the crate source directory {} must be readable: {error}",
            directory.display()
        )
    });
    for entry in entries {
        let path = entry
            .unwrap_or_else(|error| panic!("directory entry must be readable: {error}"))
            .path();
        if path.is_dir() {
            collect_rust_files(&path, into);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let text = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{} must be readable: {error}", path.display()));
            into.push((path, text));
        }
    }
}

/// Every `pub fn` signature in `text`, joined onto one line.
///
/// A signature runs from `pub fn` to the `{` that opens the body, which is how
/// a multi-line parameter list is read as one declaration.
fn public_signatures(text: &str) -> Vec<String> {
    let mut signatures = Vec::new();
    let mut collecting: Option<String> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if collecting.is_none() && !trimmed.starts_with("pub fn ") {
            continue;
        }
        let signature = collecting.get_or_insert_with(String::new);
        if !signature.is_empty() {
            signature.push(' ');
        }
        signature.push_str(trimmed);
        if trimmed.ends_with('{') || trimmed.ends_with(';') {
            let mut done = collecting.take().unwrap_or_default();
            while done.ends_with('{') || done.ends_with(';') || done.ends_with(' ') {
                done.pop();
            }
            signatures.push(done);
        }
    }
    signatures
}
