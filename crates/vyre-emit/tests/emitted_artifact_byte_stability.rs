//! Emitted artifacts stay byte-identical across refactors of every target.
//!
//! The shared success corpus renders to one pinned file per target: WGSL text
//! for the Naga module (which also pins the SPIR-V and Metal inputs), PTX text,
//! and SPIR-V hex words. Any change to register allocation, instruction
//! selection, handle order, writer options, or whitespace moves those bytes and
//! fails here: shared code may move, emitted output may not.
//!
//! The corpus, the section format, and the comparison are defined in
//! `vyre_compiler::lower::artifact_golden`. This file supplies the per-target rendering.

use std::path::PathBuf;

use naga::back::wgsl::{write_string, WriterFlags};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use vyre_compiler::lower::{artifact_golden, KernelDescriptor};

struct Target {
    name: &'static str,
    golden: &'static str,
    render: fn(&KernelDescriptor) -> String,
}

const TARGETS: [Target; 3] = [
    Target {
        name: "naga",
        golden: "success_corpus.wgsl",
        render: render_wgsl,
    },
    Target {
        name: "ptx",
        golden: "success_corpus.ptx",
        render: render_ptx,
    },
    Target {
        name: "spirv",
        golden: "success_corpus.spv.hex",
        render: render_spirv,
    },
];

fn render_wgsl(descriptor: &KernelDescriptor) -> String {
    let module = vyre_emit::naga::emit(descriptor).unwrap_or_else(|error| {
        panic!("Fix: success-corpus descriptor must emit a module: {error:?}")
    });
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|error| {
            panic!("Fix: emitted module must validate before serialization: {error:?}")
        });
    write_string(&module, &info, WriterFlags::empty()).unwrap_or_else(|error| {
        panic!("Fix: validated module must serialize to shader text: {error:?}")
    })
}

fn render_ptx(descriptor: &KernelDescriptor) -> String {
    vyre_emit::ptx::emit(descriptor)
        .unwrap_or_else(|error| panic!("Fix: success-corpus descriptor must emit PTX: {error:?}"))
}

fn render_spirv(descriptor: &KernelDescriptor) -> String {
    let bytes = vyre_emit::spirv::emit_bytes(descriptor).unwrap_or_else(|error| {
        panic!("Fix: success-corpus descriptor must emit SPIR-V: {error:?}")
    });
    artifact_golden::hex_words(&bytes)
}

fn golden_path(target: &Target) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(target.golden)
}

fn render_corpus(target: &Target) -> String {
    artifact_golden::render_success_corpus(target.render)
}

#[test]
fn every_target_matches_its_pinned_corpus() {
    for target in &TARGETS {
        artifact_golden::assert_matches_golden(&golden_path(target), &render_corpus(target));
    }
}

/// A pinned corpus that no longer names every case would stop checking the
/// cases it dropped.
#[test]
fn every_pinned_corpus_covers_every_shared_success_case() {
    for target in &TARGETS {
        let golden = std::fs::read_to_string(golden_path(target)).unwrap_or_else(|error| {
            panic!("Fix: pinned {} corpus must exist: {error}", target.name)
        });
        for case in vyre_compiler::lower::emit_adversarial_corpus::success_cases() {
            assert!(
                golden.contains(&format!("===== {}\n", case.id)),
                "Fix: pinned {} corpus is missing case `{}`; re-bless it.",
                target.name,
                case.id
            );
        }
    }
}

/// Emission is a pure function of the descriptor. A renderer that depended on a
/// hash seed, iteration order, or an address would pass the golden once and
/// fail the next run.
#[test]
fn every_target_is_deterministic_across_runs() {
    for target in &TARGETS {
        assert_eq!(
            render_corpus(target),
            render_corpus(target),
            "{} emission is not deterministic",
            target.name
        );
    }
}

#[test]
#[ignore = "writes the pinned corpora; run deliberately and review the diff"]
fn bless_pinned_corpora() {
    for target in &TARGETS {
        artifact_golden::write_golden(&golden_path(target), &render_corpus(target));
    }
}
