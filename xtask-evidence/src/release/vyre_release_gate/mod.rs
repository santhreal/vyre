//! Evidence gate for the Vyre release objective.
//!
//! This gate intentionally checks artifacts, not intent. The release is
//! blocked until every requirement in `release/vyre-release-evidence.toml` is
//! closed and backed by concrete files.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

mod checks;
mod gate_inputs;
mod paths;
mod semantic;

use checks::check_markdown_evidence_path_ready;
use gate_inputs::{EvidenceManifest, GateMode};
use paths::{escapes_repository, options_from_args, read_text_bounded, resolve_manifest_path};
use semantic::run_semantic_requirement_checks;
use xtask::gate::{Gate, GateCtx, GateError, Report};

/// Holds the release to the evidence manifest: every requirement closed and
/// every cited artifact present, fresh, and internally consistent.
pub struct VyreReleaseGate;

impl Gate for VyreReleaseGate {
    fn name(&self) -> &'static str {
        "vyre-release-gate"
    }

    fn help(&self) -> &'static str {
        "Judge the release evidence manifest. The default prepublication mode accepts the three \
         approval-gated external actions as pending and rejects every internal blocker. \
         --launch-complete additionally requires completed publication, repository verification \
         and pushes, so it is only meaningful after the release has shipped. --manifest PATH \
         judges another manifest."
    }

    fn usage(&self) -> &'static [&'static str] {
        &[
            "--launch-complete additionally requires completed publication, repository verification and pushes",
            "--manifest PATH judges another release evidence manifest",
        ]
    }

    fn generates(&self) -> bool {
        false
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        inspect(&ctx.root, &ctx.args)
    }
}

fn inspect(root: &Path, args: &[String]) -> Result<Report, GateError> {
    let options = options_from_args(root, args).map_err(|message| {
        GateError::new(
            message,
            "Pass only --launch-complete and --manifest PATH.".to_string(),
        )
    })?;
    let manifest_path = options.manifest_path;

    let manifest_text = read_text_bounded(&manifest_path).map_err(|error| {
        GateError::new(
            format!(
                "failed to read release evidence manifest `{}`: {error}",
                manifest_path.display()
            ),
            "Restore the release evidence manifest or name another with --manifest.",
        )
    })?;

    let manifest: EvidenceManifest = toml::from_str(&manifest_text).map_err(|error| {
        GateError::new(
            format!(
                "release evidence manifest `{}` is invalid TOML: {error}",
                manifest_path.display()
            ),
            "Repair the manifest syntax so the gate can judge it.",
        )
    })?;

    let base_dir = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut failures = Vec::new();

    if manifest.schema_version != 2 {
        failures.push(format!(
            "schema_version must be 2, found {}",
            manifest.schema_version
        ));
    }
    if manifest.release.vyre.trim().is_empty() {
        failures.push("release.vyre is empty".to_string());
    }

    // The release contract is a tracked document in this repository.
    let release_contract_path = resolve_manifest_path(&base_dir, &manifest.release_contract_path);
    if !release_contract_path.is_file() {
        failures.push(format!(
            "release_contract_path `{}` does not resolve to a file",
            release_contract_path.display()
        ));
    } else {
        match read_text_bounded(&release_contract_path) {
            Ok(_) => {}
            Err(error) => failures.push(format!(
                "release contract `{}` could not be read: {error}",
                release_contract_path.display()
            )),
        }
    }

    let mut ids = BTreeSet::new();
    for requirement in &manifest.requirements {
        if !ids.insert(requirement.id.as_str()) {
            failures.push(format!("duplicate requirement id `{}`", requirement.id));
        }
        if requirement.title.trim().is_empty() {
            failures.push(format!(
                "requirement `{}` has an empty title",
                requirement.id
            ));
        }
        if requirement.status != "closed" {
            failures.push(format!(
                "requirement `{}` is `{}`; release requires `closed`",
                requirement.id, requirement.status
            ));
        }
        if requirement.status == "closed" {
            for (command, artifact) in unlisted_produced_artifacts(requirement) {
                failures.push(format!(
                    "requirement `{}` runs `{command}`, which writes `{artifact}`, and does not list it as evidence",
                    requirement.id
                ));
            }
            for evidence in &requirement.evidence {
                if is_manifest_command_evidence(evidence) {
                    continue;
                }
                if escapes_repository(evidence) {
                    failures.push(format!(
                        "requirement `{}` evidence path `{evidence}` resolves outside the repository",
                        requirement.id
                    ));
                    continue;
                }
                let evidence_path = resolve_manifest_path(&base_dir, evidence);
                match fs::metadata(&evidence_path) {
                    Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {
                        if evidence.ends_with(".md") {
                            check_markdown_evidence_path_ready(
                                requirement,
                                &evidence_path,
                                evidence,
                                &mut failures,
                            );
                        }
                    }
                    Ok(metadata) if metadata.is_file() => failures.push(format!(
                        "requirement `{}` evidence path `{}` is empty",
                        requirement.id,
                        evidence_path.display()
                    )),
                    Ok(_) => failures.push(format!(
                        "requirement `{}` evidence path `{}` exists but is not a file",
                        requirement.id,
                        evidence_path.display()
                    )),
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {
                        failures.push(format!(
                            "requirement `{}` evidence path `{}` does not exist",
                            requirement.id,
                            evidence_path.display()
                        ));
                    }
                    Err(error) => failures.push(format!(
                        "requirement `{}` evidence path `{}` is unreadable: {error}",
                        requirement.id,
                        evidence_path.display()
                    )),
                }
            }
            run_semantic_requirement_checks(requirement, &base_dir, options.mode, &mut failures);
        }
    }

    const REQUIRED_IDS: &[&str] = &[
        "version-story",
        "cuda-first-path",
        "wgpu-fallback",
        "megakernel-default",
        "optimization-corpus-4096",
        "optimization-benchmark-proof",
        "semantic-optimizer-registration",
        "proof-workloads-12",
        "cpu-only-100x-proof",
        "conformance-hard-gate",
        "docs-evidence-linked",
        "crate-metadata",
        "release-hygiene",
        "public-launch",
    ];

    for required in REQUIRED_IDS {
        if !ids.contains(required) {
            failures.push(format!("manifest is missing required id `{required}`"));
        }
    }

    let scope = match options.mode {
        GateMode::LaunchComplete => "launch complete",
        GateMode::Prepublish => "prepublication",
    };
    let stale = collapse_stale_source_verdicts(&mut failures);
    let mut report = Report::from_messages(
        failures,
        "Attach real evidence artifacts and close every manifest requirement.",
    );
    report.note(format!(
        "{} requirement(s) judged for Vyre {} ({scope})",
        manifest.requirements.len(),
        manifest.release.vyre
    ));
    for note in stale {
        report.note(note);
    }
    Ok(report)
}

/// The command that re-measures the release benchmark suite for one backend.
const RE_MEASURE: &str = "cargo_full run -q -p xtask --bin xtask -- release-benchmarks --write --backend";

/// Replace every stale-source verdict with the one command that answers them all.
///
/// A recorded benchmark names the tree it ran on, so one commit invalidates every
/// artifact at once and each invalidated artifact then fails several checks: the
/// freshness of its own fingerprint, the aggregate that cites it, the backend
/// suite that inventories it. That was 164 of this gate's findings, and reading
/// them cost the same as reading one, because a finding per artifact per check
/// describes one action: re-measure on a release host. The verdicts are not
/// discarded, they become notes, so the artifact list stays readable while the
/// finding count says how many decisions are open rather than how many artifacts
/// one decision touches.
///
/// The action is a command and not a hand edit. A fingerprint typed into a JSON
/// file records a measurement that never happened, and there is no way to tell
/// the two apart afterwards, so the only honest way to refresh one is to run the
/// generator that stamps the tree it measured.
fn collapse_stale_source_verdicts(failures: &mut Vec<String>) -> Vec<String> {
    let stale: Vec<String> = failures
        .iter()
        .filter(|failure| xtask::source_provenance::is_stale_source_verdict(failure))
        .cloned()
        .collect();
    if stale.is_empty() {
        return Vec::new();
    }
    failures.retain(|failure| !xtask::source_provenance::is_stale_source_verdict(failure));
    let artifacts: BTreeSet<String> = stale.iter().filter_map(|verdict| cited_artifact(verdict)).collect();
    failures.push(format!(
        "{} recorded verdict(s) over {} benchmark evidence artifact(s) name a source tree that is no longer this one; re-measure on a release host with `{RE_MEASURE} cuda` and `{RE_MEASURE} wgpu`",
        stale.len(),
        artifacts.len()
    ));
    let mut notes = vec![format!(
        "stale benchmark evidence: {}",
        artifacts.iter().cloned().collect::<Vec<_>>().join(", ")
    )];
    notes.extend(stale);
    notes
}

/// The artifact a verdict is about: the first backtick-quoted `.json` path in it.
fn cited_artifact(verdict: &str) -> Option<String> {
    verdict
        .split('`')
        .skip(1)
        .step_by(2)
        .find(|quoted| quoted.ends_with(".json"))
        .map(str::to_string)
}

pub(super) fn is_manifest_command_evidence(evidence: &str) -> bool {
    evidence.starts_with("cargo_full ")
}

/// The xtask subcommand a command-form evidence line runs.
///
/// The manifest spells a producer as the wrapper invocation that runs it, so the
/// subcommand is the token after the `--` separator.
fn manifest_command_subcommand(evidence: &str) -> Option<&str> {
    let mut words = evidence.split_whitespace().skip_while(|word| *word != "--");
    words.next()?;
    words.next()
}

/// Every artifact a requirement's own producers write and the requirement does
/// not list, as `(subcommand, artifact)`.
///
/// This replaces a hand-set `minimum_evidence` integer. A count is correct only
/// while somebody keeps it level with the list beside it: `release-hygiene`
/// required 16 and carried 13 because an earlier commit removed three evidence
/// paths no producer had ever written and left the floor alone, so the gate
/// reported a shortfall that named nothing and the list stayed wrong. The
/// relation that actually matters is that the requirement lists what its own
/// generators produce, and `expected_artifacts_for_command` is where that
/// producer set is already declared, so it is derived here rather than tallied.
/// A generator that grows an artifact makes every requirement running it red
/// until the artifact is listed.
fn unlisted_produced_artifacts(
    requirement: &gate_inputs::Requirement,
) -> Vec<(String, &'static str)> {
    let listed: BTreeSet<&str> = requirement
        .evidence
        .iter()
        .map(|entry| entry.trim_start_matches("../").trim_start_matches("release/"))
        .collect();
    let mut missing = Vec::new();
    for evidence in &requirement.evidence {
        if !is_manifest_command_evidence(evidence) {
            continue;
        }
        let Some(subcommand) = manifest_command_subcommand(evidence) else {
            continue;
        };
        for artifact in crate::release::release_evidence::expected_artifacts::
            expected_artifacts_for_command(subcommand)
        {
            let relative = artifact.trim_start_matches("release/");
            if !listed.contains(relative) {
                missing.push((subcommand.to_string(), *artifact));
            }
        }
    }
    missing
}

#[cfg(test)]
mod tests {
    use super::*;
    use gate_inputs::Requirement;

    fn closed(id: &str, evidence: &[&str]) -> Requirement {
        Requirement {
            id: id.to_string(),
            title: id.to_string(),
            status: "closed".to_string(),
            evidence: evidence.iter().map(|entry| (*entry).to_string()).collect(),
        }
    }

    /// The subcommand is the word after the wrapper's `--` separator, and a
    /// command with no separator names no subcommand.
    #[test]
    fn subcommand_is_read_from_the_wrapper_invocation() {
        assert_eq!(
            manifest_command_subcommand("cargo_full run -p xtask --bin xtask -- version-matrix"),
            Some("version-matrix")
        );
        assert_eq!(
            manifest_command_subcommand("cargo_full run -p xtask --bin xtask -- backend-matrix --backend cuda"),
            Some("backend-matrix")
        );
        assert_eq!(manifest_command_subcommand("cargo_full test --workspace"), None);
        assert_eq!(manifest_command_subcommand("cargo_full run -p xtask --"), None);
    }

    /// A requirement that lists every artifact its own producer writes has
    /// nothing unlisted, whether the path is spelled with the `release/` prefix
    /// or relative to the manifest.
    #[test]
    fn a_complete_listing_reports_nothing() {
        let produced = crate::release::release_evidence::expected_artifacts::expected_artifacts_for_command("version-matrix");
        assert!(!produced.is_empty(), "Fix: fixture subcommand must produce artifacts");
        let mut evidence = vec!["cargo_full run -p xtask --bin xtask -- version-matrix".to_string()];
        evidence.extend(produced.iter().map(|artifact| (*artifact).to_string()));
        let borrowed: Vec<&str> = evidence.iter().map(String::as_str).collect();
        assert!(unlisted_produced_artifacts(&closed("complete", &borrowed)).is_empty());

        let relative: Vec<String> = std::iter::once(evidence[0].clone())
            .chain(
                produced
                    .iter()
                    .map(|artifact| format!("../{}", artifact.trim_start_matches("release/"))),
            )
            .collect();
        let borrowed: Vec<&str> = relative.iter().map(String::as_str).collect();
        assert!(unlisted_produced_artifacts(&closed("relative", &borrowed)).is_empty());
    }

    /// A requirement that runs a producer and omits one of its artifacts is
    /// reported with the command and the artifact, which is what a hand-set
    /// evidence count could not say.
    #[test]
    fn an_omitted_artifact_is_named_with_its_producer() {
        let produced = crate::release::release_evidence::expected_artifacts::expected_artifacts_for_command("version-matrix");
        let omitted = produced
            .first()
            .copied()
            .expect("Fix: fixture subcommand must produce artifacts");
        let mut evidence = vec!["cargo_full run -p xtask --bin xtask -- version-matrix".to_string()];
        evidence.extend(
            produced
                .iter()
                .filter(|artifact| **artifact != omitted)
                .map(|artifact| (*artifact).to_string()),
        );
        let borrowed: Vec<&str> = evidence.iter().map(String::as_str).collect();
        let unlisted = unlisted_produced_artifacts(&closed("incomplete", &borrowed));
        assert_eq!(
            unlisted,
            vec![("version-matrix".to_string(), omitted)],
            "Fix: an unlisted produced artifact must name its producing subcommand"
        );
    }

    /// Evidence that is a path rather than a command produces no expectation of
    /// its own, so a requirement backed only by files is never reported.
    #[test]
    fn path_evidence_produces_no_expectation() {
        assert!(
            unlisted_produced_artifacts(&closed(
                "paths-only",
                &["evidence/version/version-matrix.json", "../evidence/other.json"]
            ))
            .is_empty()
        );
    }

    /// Every closed requirement in the shipped manifest lists every artifact its
    /// own producers write.
    ///
    /// The relation is derived from the producer map at run time, so a generator
    /// that grows an artifact turns this red until the requirements running it
    /// list the new file. The manifest is found by walking up from the working
    /// directory, because a shared target directory can hand this test a binary
    /// another checkout compiled.
    #[test]
    fn the_shipped_manifest_lists_what_its_producers_write() {
        let mut directory = std::env::current_dir().expect("Fix: working directory must be readable");
        let manifest_path = loop {
            let candidate = directory.join("release/vyre-release-evidence.toml");
            if candidate.is_file() {
                break candidate;
            }
            assert!(
                directory.pop(),
                "Fix: `release/vyre-release-evidence.toml` must exist in an ancestor of the working directory"
            );
        };
        let text = fs::read_to_string(&manifest_path).expect("Fix: manifest must be readable");
        let manifest: EvidenceManifest =
            toml::from_str(&text).expect("Fix: manifest must be valid TOML");
        let mut unlisted = Vec::new();
        for requirement in &manifest.requirements {
            if requirement.status != "closed" {
                continue;
            }
            for (command, artifact) in unlisted_produced_artifacts(requirement) {
                unlisted.push(format!("{}: {command} writes {artifact}", requirement.id));
            }
        }
        assert!(
            unlisted.is_empty(),
            "Fix: list every produced artifact as evidence: {unlisted:?}"
        );
    }

    /// Stale-source verdicts collapse to one finding that names the re-measure
    /// command, and every verdict survives as a note.
    ///
    /// The predicate comes from `xtask::source_provenance`, the same owner the
    /// four producers format their message from, so a producer that reworded its
    /// sentence without moving the owner cannot slip past this. The count is the
    /// contract: many verdicts over many artifacts are one decision, and a
    /// finding per artifact per check reported the same decision once per
    /// artifact.
    #[test]
    fn stale_source_verdicts_collapse_to_one_finding() {
        let predicate = xtask::source_provenance::STALE_SOURCE_PREDICATE;
        let mut failures: Vec<String> = (0..7)
            .map(|index| {
                format!(
                    "`release/evidence/benchmarks/suite-{index}.json` {predicate}, so its numbers are not this tree's"
                )
            })
            .collect();
        failures.push("`release/evidence/benchmarks/suite-0.json` fingerprint is missing".to_string());

        let notes = collapse_stale_source_verdicts(&mut failures);

        assert_eq!(
            failures.len(),
            2,
            "Fix: seven stale verdicts and one unrelated failure must leave one collapsed finding beside the unrelated one: {failures:?}"
        );
        assert_eq!(
            failures[0], "`release/evidence/benchmarks/suite-0.json` fingerprint is missing",
            "Fix: a failure that is not a freshness verdict must survive unchanged"
        );
        let collapsed = &failures[1];
        assert!(
            collapsed.starts_with("7 recorded verdict(s) over 7 benchmark evidence artifact(s)"),
            "Fix: the collapsed finding must state how many verdicts and artifacts it stands for: {collapsed}"
        );
        assert!(
            collapsed.contains(&format!("{RE_MEASURE} cuda"))
                && collapsed.contains(&format!("{RE_MEASURE} wgpu")),
            "Fix: the collapsed finding must name the command that re-measures each backend: {collapsed}"
        );
        assert_eq!(
            notes.len(),
            8,
            "Fix: the artifact list plus every collapsed verdict must survive as notes"
        );
        for index in 0..7 {
            let artifact = format!("release/evidence/benchmarks/suite-{index}.json");
            assert!(
                notes[0].contains(&artifact),
                "Fix: the artifact note must list every stale artifact, missing {artifact}"
            );
        }
    }

    /// Verdicts over one artifact collapse to one finding that counts the
    /// artifact once.
    ///
    /// Three checks read the same recorded fingerprint, so the artifact count and
    /// the verdict count are different numbers and a reader needs both to know
    /// whether one re-measure closes the finding.
    #[test]
    fn repeated_verdicts_over_one_artifact_count_the_artifact_once() {
        let predicate = xtask::source_provenance::STALE_SOURCE_PREDICATE;
        let mut failures: Vec<String> = ["freshness", "aggregate", "suite inventory"]
            .iter()
            .map(|check| format!("{check}: `release/evidence/benchmarks/only.json` {predicate}"))
            .collect();

        collapse_stale_source_verdicts(&mut failures);

        assert_eq!(failures.len(), 1);
        assert!(
            failures[0].starts_with("3 recorded verdict(s) over 1 benchmark evidence artifact(s)"),
            "Fix: three verdicts over one artifact must count the artifact once: {}",
            failures[0]
        );
    }

    /// A run with no stale verdict changes nothing and adds no note.
    #[test]
    fn a_tree_with_fresh_evidence_is_left_alone() {
        let mut failures = vec!["`release/evidence/benchmarks/a.json` is not listed by any requirement".to_string()];
        let notes = collapse_stale_source_verdicts(&mut failures);
        assert_eq!(failures.len(), 1);
        assert!(notes.is_empty());
    }
}
