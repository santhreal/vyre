//! Byte-stability golden for the PTX this driver emits.
//!
//! A refactor of the CUDA driver is only safe if the bytes it hands the device
//! do not move. This pins the whole host-side emission path with no GPU: the
//! registered target compiler chooses the emit options from its own
//! `TargetProfile`, `compile_selected_modules` fuses and lowers each selected
//! group, the emitter renders PTX, and driver admission decodes the module
//! bundle back out. Every one of those steps sits between a `Program` and the
//! `.visible .entry main` text, and none of them needs a device.
//!
//! The program corpus is `vyre_lower::program_stability_corpus`, shared with the
//! reference-oracle golden. The section format and the comparison live in
//! `vyre_lower::artifact_golden`. This file supplies only the PTX rendering.
//!
//! Materialization and dispatch are covered by the live tests in
//! `tests/target_compiler.rs`; this file deliberately stops at the bytes.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use vyre_driver::materialize::{self, MaterializerTarget};
use vyre_foundation::ir::ProgramGraph;
use vyre_lower::artifact_golden::{
    assert_matches_golden, contains_case, render_sections, write_golden,
};
use vyre_lower::program_stability_corpus::{self, StabilityCase};
use vyre_megakernel::{
    Artifact, CompileRequest, DeviceFacts, Digest, ExternalFacts, SearchBudget, TargetCompiler, TargetPayload,
};

fn golden_path() -> PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root()
        .join("vyre-driver-cuda/tests/golden/emitted_ptx_corpus.ptx")
}

/// Wrap one corpus program in the single-node graph the artifact route expects.
///
/// `ProgramGraph::from_program` owns lifting host-visible buffers into typed
/// external values, so the corpus does not restate that contract per case.
fn artifact_for(case: &StabilityCase) -> Artifact {
    let graph = ProgramGraph::from_program("main", case.program.clone()).unwrap_or_else(|error| {
        panic!(
            "Fix: corpus case `{}` must form a graph node: {error}",
            case.id
        )
    });
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 0, 0, 1),
        1_000_000,
    )
    .validate()
    .unwrap_or_else(|error| panic!("Fix: corpus case `{}` must validate: {error}", case.id));
    vyre_megakernel::compile(&request)
        .unwrap_or_else(|error| panic!("Fix: corpus case `{}` must compile: {error}", case.id))
}

/// The registered CUDA target compiler, acquired without a device.
fn cuda_target_compiler() -> Box<dyn TargetCompiler> {
    vyre_driver::backend_registration(vyre_driver_cuda::CUDA_BACKEND_ID)
        .expect("Fix: the CUDA backend registration must be linked into this test binary.")
        .target_compiler()
        .expect("Fix: CUDA target-payload production must not require a device.")
}

/// Render every admitted module of one payload as PTX text plus its entry metadata.
fn render_payload(
    compiler: &dyn TargetCompiler,
    artifact: &Artifact,
    payload: &TargetPayload,
    case_id: &str,
) -> String {
    let admitted = materialize::admit(
        artifact,
        payload,
        MaterializerTarget {
            backend_id: vyre_driver_cuda::CUDA_BACKEND_ID,
            format: compiler.format(),
            profile: compiler.profile(),
        },
    )
    .unwrap_or_else(|error| panic!("Fix: corpus case `{case_id}` must admit: {error}"));
    let mut text = String::new();
    for entry in payload.entries() {
        writeln!(
            text,
            "entry {} grid {:?} workgroup {:?} dynamic_shared {}",
            entry.name, entry.grid_size, entry.workgroup_size, entry.dynamic_shared_bytes
        )
        .expect("string write");
        for binding in &entry.resource_bindings {
            writeln!(text, "  binding {binding:?}").expect("string write");
        }
    }
    for module in &admitted {
        let ptx = std::str::from_utf8(&module.image.bytes).unwrap_or_else(|error| {
            panic!("Fix: corpus case `{case_id}` must emit UTF-8 PTX: {error}")
        });
        writeln!(text, "module group {:?}", module.image.group).expect("string write");
        text.push_str(ptx);
        if !ptx.ends_with('\n') {
            text.push('\n');
        }
    }
    text
}

/// Render the shared corpus through the registered CUDA payload route.
fn render_corpus() -> String {
    let compiler = cuda_target_compiler();
    render_sections(program_stability_corpus::cases().into_iter().map(|case| {
        let artifact = artifact_for(&case);
        let payload = compiler.compile(&artifact).unwrap_or_else(|error| {
            panic!(
                "Fix: corpus case `{}` must produce a payload: {error:?}",
                case.id
            )
        });
        (
            case.id,
            render_payload(compiler.as_ref(), &artifact, &payload, case.id),
        )
    }))
}

/// WHY: the PTX this driver hands the device is the product. A dedup refactor
/// must not move one byte of it, and only a pinned corpus can prove that.
#[test]
fn emitted_ptx_matches_the_pinned_corpus() {
    assert_matches_golden(&golden_path(), &render_corpus());
}

/// WHY: emission must be a pure function of the artifact. A renderer that
/// depended on iteration order or an address would pass the golden once and
/// fail the next run.
#[test]
fn emitted_ptx_is_deterministic_across_runs() {
    assert_eq!(render_corpus(), render_corpus());
}

/// WHY: a pinned corpus that no longer names every shared case would silently
/// stop covering whichever case was added last.
#[test]
fn pinned_corpus_covers_every_shared_case() {
    let golden = std::fs::read_to_string(golden_path()).expect("pinned PTX corpus must exist");
    for case in program_stability_corpus::cases() {
        assert!(
            contains_case(&golden, case.id),
            "Fix: pinned PTX corpus is missing case `{}`; re-bless it.",
            case.id
        );
    }
}

#[test]
#[ignore = "bless: rewrites the pinned emitted-PTX corpus"]
fn bless_pinned_ptx_corpus() {
    write_golden(&golden_path(), &render_corpus());
}
