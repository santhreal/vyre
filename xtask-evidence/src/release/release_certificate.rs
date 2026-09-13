//! Own the merged all-backend conformance certificate.
//!
//! The release script proved 64 conformance shards, merged them, and installed
//! the result as release evidence with `cp`. A copy carries no provenance head,
//! so the committed certificate stated nothing about the tree, host or device
//! that proved it, no registered gate declared the path, and the body drifted
//! to a field name no current conformance run writes without anyone reading it.
//!
//! This gate is the single writer of that path. `--write` proves the shards,
//! merges them and records the merge under the provenance of this run; without
//! it the committed certificate is held to the plan it states and the outcome
//! of every pair it carries.
//!
//! The certificate's wire type and the signature over it belong to the
//! conformance layer, which no gate may link: the tooling that grades a parity
//! harness cannot depend on the harness. So the body is read as JSON and the
//! fields the release claim rests on are named here. The signature is verified
//! by the crate that issues it, against the type that defines the signed field
//! order, which is the only place a verifier and a signer cannot disagree.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use xtask::evidence_record::MeasurementRecord;
use xtask::gate::{Finding, GateCtx, GateError, Report};

/// The artifact this gate owns, relative to the workspace root.
const ARTIFACT: &str = "release/evidence/conformance/release-all-backends-certificate.json";

/// The script that proves the shards and merges them.
const PROOF_SCRIPT: &str = "scripts/prove-release-shards.sh";

/// Where the shard pool leaves its merge when nothing overrides the directory.
const DEFAULT_MERGE: &str = ".internals/certs/release-shards/merged.json";

/// The `backend_id` a merge of every shard records.
const MERGED_BACKEND_ID: &str = "merged";

/// Flags this gate answers `--help` with.
const USAGE: &[&str] = &[
    "usage: release-certificate [--write] [--from PATH]",
    "  --write      prove and merge the release conformance shards, then record the",
    "               merged certificate under the provenance of this run",
    "  --from PATH  record this already merged certificate instead of proving one",
];

pub(crate) struct ReleaseCertificateGate;

impl xtask::gate::GateBehavior for ReleaseCertificateGate {
    fn usage(&self) -> &'static [&'static str] {
        USAGE
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        Ok(xtask::artifact_gate::settle_measured(
            ctx,
            &[ARTIFACT],
            "merged all-backend conformance certificates",
            USAGE,
            parse_args,
            record,
            audit,
        ))
    }
}

/// Where the merge this gate records comes from.
struct Config {
    /// An already merged certificate, or `None` to prove the shards here.
    from: Option<PathBuf>,
}

/// The configuration to record with, `None` for the option list.
fn parse_args(args: &[String]) -> Result<Option<Config>, String> {
    let mut from = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--from" => {
                let value = rest
                    .next()
                    .ok_or_else(|| "`--from` names no certificate path".to_string())?;
                from = Some(PathBuf::from(value));
            }
            other => return Err(format!("`{other}` is not a flag this gate reads")),
        }
    }
    Ok(Some(Config { from }))
}

/// Prove the shards when asked to, then record the merged certificate.
fn record(root: &Path, config: &Config, report: &mut Report) {
    let merged = match config.from.as_deref() {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => root.join(path),
        None => match prove_and_merge(root) {
            Ok(path) => path,
            Err(finding) => {
                report.find(finding);
                return;
            }
        },
    };
    let text = match std::fs::read_to_string(&merged) {
        Ok(text) => text,
        Err(error) => {
            report.find(Finding::in_file(
                PathBuf::from(ARTIFACT),
                format!(
                    "merged certificate `{}` cannot be read: {error}",
                    merged.display()
                ),
                format!(
                    "Run `{PROOF_SCRIPT}` and pass its merge with `--from PATH`, or rerun this \
                     gate's `--write` with no `--from` so it proves the shards itself."
                ),
            ));
            return;
        }
    };
    let certificate = match parse(&text) {
        Ok(certificate) => certificate,
        Err(message) => {
            report.find(Finding::in_file(
                PathBuf::from(ARTIFACT),
                format!("merged certificate `{}` {message}", merged.display()),
                "Reprove the shards with the current conformance binary so the merge carries the \
                 wire shape this reader parses.",
            ));
            return;
        }
    };
    let findings = judge(&certificate);
    if !findings.is_empty() {
        for finding in findings {
            report.find(finding);
        }
        return;
    }
    if let Err(message) = xtask::artifact_gate::write_recorded(
        root,
        Path::new(ARTIFACT),
        MeasurementRecord::device(),
        &certificate,
    ) {
        report.find(Finding::in_file(
            PathBuf::from(ARTIFACT),
            message,
            "Record the certificate from a checkout git can identify, on the host whose devices \
             proved it.",
        ));
        return;
    }
    report.note(format!("recorded {ARTIFACT} from {}", merged.display()));
}

/// Run the shard pool and return the merge it printed.
///
/// The pool, its worker accounting and its shard count belong to the script,
/// which a contract test already pins. Reimplementing them here would give the
/// release two pools that can disagree about what a failed shard is.
fn prove_and_merge(root: &Path) -> Result<PathBuf, Finding> {
    let output = Command::new("bash")
        .arg(root.join(PROOF_SCRIPT))
        .current_dir(root)
        .output()
        .map_err(|error| {
            Finding::in_file(
                PathBuf::from(PROOF_SCRIPT),
                format!("`{PROOF_SCRIPT}` could not be run: {error}"),
                "Install a bash the release lane can run, or pass an existing merge with \
                 `--from PATH`.",
            )
        })?;
    if !output.status.success() {
        let diagnostic = String::from_utf8_lossy(&output.stderr);
        return Err(Finding::in_file(
            PathBuf::from(PROOF_SCRIPT),
            format!(
                "`{PROOF_SCRIPT}` did not prove the release shards: {}",
                diagnostic.trim_end()
            ),
            "Resolve what the shard pool reported and rerun this gate's `--write` on a host whose \
             devices acquire.",
        ));
    }
    let printed = String::from_utf8_lossy(&output.stdout);
    let Some(path) = printed
        .lines()
        .next_back()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    else {
        return Err(Finding::in_file(
            PathBuf::from(PROOF_SCRIPT),
            format!("`{PROOF_SCRIPT}` proved the shards and printed no merge path"),
            format!("The script prints the merge on stdout; expected it at `{DEFAULT_MERGE}`."),
        ));
    };
    let path = PathBuf::from(path);
    Ok(if path.is_absolute() {
        path
    } else {
        root.join(path)
    })
}

/// Hold the committed certificate to the plan it states.
///
/// This runs no proof and needs no device. The certificate was installed by a
/// copy, so nothing read it after the merge wrote it: the committed body states
/// its pairs under a field name the conformance schema retired, and no reader
/// noticed because no reader existed. `--from` names where a merge to record
/// comes from, so an audit of the recorded certificate reads it not at all.
fn audit(root: &Path, _config: &Config, report: &mut Report) {
    let committed = match std::fs::read_to_string(root.join(ARTIFACT)) {
        Ok(committed) => committed,
        Err(error) => {
            report.find(Finding::in_file(
                PathBuf::from(ARTIFACT),
                format!("`{ARTIFACT}` cannot be read: {error}"),
                "Prove the release conformance shards and record them with this gate's `--write`.",
            ));
            return;
        }
    };
    let (_, body) = xtask::artifact_gate::split_provenance(&committed);
    let certificate = match parse(&body) {
        Ok(certificate) => certificate,
        Err(message) => {
            report.find(Finding::in_file(
                PathBuf::from(ARTIFACT),
                format!("`{ARTIFACT}` {message}"),
                "Reprove the release conformance shards with the current conformance binary and \
                 record them with this gate's `--write`.",
            ));
            return;
        }
    };
    for finding in judge(&certificate) {
        report.find(finding);
    }
}

/// Read one certificate body as the object this gate reads fields out of.
fn parse(text: &str) -> Result<Value, String> {
    let value: Value = serde_json::from_str(text)
        .map_err(|error| format!("is not a JSON document: {error}"))?;
    if value.is_object() {
        Ok(value)
    } else {
        Err("is not a JSON object, so it carries no certificate fields".to_string())
    }
}

/// Every judgement this gate makes about one merged certificate.
///
/// The same function judges the merge before it is recorded and the committed
/// body afterwards, so a certificate cannot be installed in a state the audit
/// would report.
fn judge(certificate: &Value) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut require = |message: String, fix: &str| {
        findings.push(Finding::in_file(PathBuf::from(ARTIFACT), message, fix));
    };

    match certificate.pointer("/backend_id").and_then(Value::as_str) {
        Some(MERGED_BACKEND_ID) => {}
        Some(other) => require(
            format!(
                "the certificate covers backend filter `{other}`, and release evidence is the \
                 merge of every shard, which records `{MERGED_BACKEND_ID}`"
            ),
            "Record the merge the shard pool produced, not one shard's certificate.",
        ),
        None => require(
            "the certificate states no `backend_id`".to_string(),
            "Reprove the shards with the current conformance binary; a certificate with no \
             backend filter is not one this reader can place.",
        ),
    }

    if let Some(index) = certificate.pointer("/plan/selection/shard_index") {
        if !index.is_null() {
            require(
                format!(
                    "the certificate records shard index {index}, so it is one shard rather than \
                     the merge of every shard"
                ),
                "Record the merge the shard pool produced, not one shard's certificate.",
            );
        }
    }

    let pairs = certificate.pointer("/pairs").and_then(Value::as_array);
    match (
        certificate
            .pointer("/plan/pair_count")
            .and_then(Value::as_u64),
        pairs,
    ) {
        (Some(planned), Some(carried)) if planned as usize != carried.len() => require(
            format!(
                "the certificate plans {planned} pair(s) and carries {}",
                carried.len()
            ),
            "Reprove the shards; a merge that dropped a pair states a plan it did not execute.",
        ),
        (None, _) => require(
            "the certificate plans no pair count".to_string(),
            "Reprove the shards with the current conformance binary so the plan states what it \
             executed.",
        ),
        _ => {}
    }

    let planned_backends = certificate
        .pointer("/plan/backend_count")
        .and_then(Value::as_u64);
    match (
        certificate
            .pointer("/plan/selection/selected_backend_count")
            .and_then(Value::as_u64),
        planned_backends,
    ) {
        (Some(selected), Some(planned)) if selected != planned => require(
            format!("the certificate selected {selected} backend(s) and plans {planned}"),
            "Reprove the shards on a host that acquires every backend the selection names.",
        ),
        _ => {}
    }

    let unavailable = certificate
        .pointer("/plan/selection/unavailable_backends")
        .and_then(Value::as_array)
        .map_or(0, Vec::len) as u64;
    match (
        certificate
            .pointer("/plan/selection/universe_backend_count")
            .and_then(Value::as_u64),
        planned_backends,
    ) {
        (Some(universe), Some(planned)) if universe != planned + unavailable => require(
            format!(
                "the certificate registers {universe} backend(s), proved {planned} and names \
                 {unavailable} it could not acquire, so {} are unaccounted for",
                universe.saturating_sub(planned + unavailable)
            ),
            "Reprove the shards; a backend that was neither proved nor recorded as unacquirable \
             is one the certificate is silent about.",
        ),
        (None, _) => require(
            "the certificate states no registered backend count".to_string(),
            "Reprove the shards with the current conformance binary so the selection states the \
             universe it chose from.",
        ),
        _ => {}
    }

    let Some(pairs) = pairs else {
        require(
            "the certificate carries no `pairs` array".to_string(),
            "Prove the release conformance shards on a host whose devices acquire.",
        );
        return findings;
    };
    if pairs.is_empty() {
        require(
            "the certificate carries no proved pair".to_string(),
            "Prove the release conformance shards on a host whose devices acquire.",
        );
    }
    for pair in pairs {
        let op_id = pair
            .pointer("/op_id")
            .and_then(Value::as_str)
            .unwrap_or("<unnamed>");
        let Some(executor) = pair.pointer("/executor_id").and_then(Value::as_str) else {
            require(
                format!(
                    "pair `{op_id}` names no `executor_id`, so the certificate does not state \
                     what executed it"
                ),
                "Reprove the shards with the current conformance binary; `backend_id` on a pair \
                 is the retired field name, and a record label is not a device.",
            );
            continue;
        };
        match pair.pointer("/passed").and_then(Value::as_bool) {
            Some(true) => {}
            Some(false) => require(
                format!(
                    "`{op_id}` did not conform on executor `{executor}`: {}",
                    pair.pointer("/message")
                        .and_then(Value::as_str)
                        .unwrap_or("<no diagnostic>")
                ),
                "Fix the operation the sentence names, or the backend lowering it took, and \
                 reprove the shards.",
            ),
            None => require(
                format!("pair `{op_id}` on executor `{executor}` states no outcome"),
                "Reprove the shards with the current conformance binary so every pair records \
                 whether it conformed.",
            ),
        }
    }

    let mut executors: Vec<&str> = pairs
        .iter()
        .filter_map(|pair| pair.pointer("/executor_id").and_then(Value::as_str))
        .collect();
    executors.sort_unstable();
    executors.dedup();
    match planned_backends {
        Some(planned) if executors.len() as u64 != planned => require(
            format!(
                "the certificate plans {planned} backend(s) and its pairs name {}: {}",
                executors.len(),
                executors.join(", ")
            ),
            "Reprove the shards; a merge that dropped one backend's pairs states a plan it did \
             not carry.",
        ),
        _ => {}
    }

    if let Some(unavailable) = certificate
        .pointer("/plan/selection/unavailable_backends")
        .and_then(Value::as_array)
    {
        for backend in unavailable {
            let id = backend.pointer("/id").and_then(Value::as_str);
            let reason = backend
                .pointer("/reason")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|reason| !reason.is_empty());
            if id.is_none() || reason.is_none() {
                require(
                    format!(
                        "an unacquired backend is recorded as id `{}` with refusal `{}`",
                        id.unwrap_or("<unnamed>"),
                        reason.unwrap_or("<none>")
                    ),
                    "Reprove the shards with the current conformance binary so each backend the \
                     host could not acquire names itself and what refused it.",
                );
            }
        }
    }
    findings
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Executor ids one release merge carries, as the conformance runner
    /// writes them.
    ///
    /// The gate counts executors instead of matching these strings, because
    /// the op matrix names the oracle column `reference` and the runner writes
    /// `cpu-ref`. The fixture uses the runner's spelling so a real merge and
    /// this body differ in nothing the gate reads.
    const EXECUTORS: [&str; 3] = ["cpu-ref", "cuda", "wgpu"];

    /// One certificate body that passes every judgement.
    fn merged(pairs: Value) -> Value {
        let rows = pairs.as_array().map_or(0, Vec::len);
        let mut executors: Vec<&str> = pairs
            .as_array()
            .map(|pairs| {
                pairs
                    .iter()
                    .filter_map(|pair| pair.pointer("/executor_id").and_then(Value::as_str))
                    .collect()
            })
            .unwrap_or_default();
        executors.sort_unstable();
        executors.dedup();
        let backends = executors.len().max(1);
        serde_json::json!({
            "wire_format_version": 1,
            "program_hash": "00",
            "backend_id": MERGED_BACKEND_ID,
            "plan": {
                "backend_count": backends,
                "op_count": rows,
                "pair_count": rows,
                "witness_case_count": rows,
                "catalog_hash": "00",
                "execution_hash": "00",
                "selection": {
                    "backend_filter": MERGED_BACKEND_ID,
                    "ops_filter": MERGED_BACKEND_ID,
                    "shard_index": Value::Null,
                    "shard_count": 64,
                    "universe_backend_count": backends,
                    "universe_op_count": rows,
                    "selected_backend_count": backends,
                    "selected_op_count": rows
                }
            },
            "signature": "00",
            "public_key": "00",
            "pairs": pairs
        })
    }

    /// The pairs a release merge carries: every operation on every executor.
    fn release_pairs(ops: &[(&str, bool)]) -> Value {
        let mut pairs = Vec::with_capacity(ops.len() * EXECUTORS.len());
        for executor in EXECUTORS {
            for (op_id, passed) in ops {
                pairs.push(pair_on(executor, op_id, *passed));
            }
        }
        Value::Array(pairs)
    }

    /// One proved pair with the stated outcome on the stated executor.
    fn pair_on(executor: &str, op_id: &str, passed: bool) -> Value {
        serde_json::json!({
            "op_id": op_id,
            "executor_id": executor,
            "passed": passed,
            "message": "1 witness case(s) matched"
        })
    }

    /// The sentences `judge` reported, joined for a substring assertion.
    fn reported(certificate: &Value) -> String {
        judge(certificate)
            .iter()
            .map(|finding| format!("{finding:?}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// WHY: a merge of 64 shards is the release artifact and one shard is not.
    /// The script writes both into the same directory, so the wrong file is one
    /// path away.
    #[test]
    fn one_shard_recorded_as_the_merge_is_a_finding() {
        let mut certificate = merged(release_pairs(&[("vyre::add", true)]));
        certificate["plan"]["selection"]["shard_index"] = serde_json::json!(7);
        certificate["backend_id"] = serde_json::json!("cuda");
        let rendered = reported(&certificate);
        assert!(
            rendered.contains("shard index 7"),
            "Fix: a shard certificate must be refused, got {rendered}"
        );
        assert!(
            rendered.contains("backend filter `cuda`"),
            "Fix: the refusal must name the filter the body records, got {rendered}"
        );
    }

    /// WHY: the release gate read the certificate's pass counts and never the
    /// pairs, so a failed pair inside a non-empty certificate was release
    /// evidence that the operation conformed.
    #[test]
    fn a_failed_pair_is_a_finding_that_names_the_operation() {
        let certificate =
            merged(release_pairs(&[("vyre::add", true), ("vyre::mul", false)]));
        let rendered = reported(&certificate);
        assert!(
            rendered.contains("vyre::mul") && rendered.contains("did not conform"),
            "Fix: a failed pair must be a finding naming the operation, got {rendered}"
        );
        assert!(
            !rendered.contains("vyre::add"),
            "Fix: a passing pair must not be a finding, got {rendered}"
        );
    }

    /// WHY: `pair_count` is what the release gate counted. A merge that dropped
    /// a shard's rows keeps the planned count and carries fewer pairs, which
    /// reads as a complete run.
    #[test]
    fn a_plan_that_counts_more_pairs_than_the_body_carries_is_a_finding() {
        let mut certificate = merged(release_pairs(&[("vyre::add", true)]));
        certificate["plan"]["pair_count"] = serde_json::json!(1077);
        let rendered = reported(&certificate);
        assert!(
            rendered.contains("plans 1077 pair(s) and carries 3"),
            "Fix: the finding must name both counts, got {rendered}"
        );
    }

    /// WHY: a backend the host could not acquire is a property of the host, not
    /// a defect in the record: the `gpu` feature links the Metal and SPIR-V
    /// registrations on every platform so that naming one gives a refusal
    /// instead of `unknown backend`. The certificate states which backends it
    /// covers and why it covers no more, so an unacquirable backend that names
    /// its refusal is complete evidence, not a finding.
    #[test]
    fn a_backend_the_run_could_not_acquire_is_recorded_without_a_finding() {
        let mut certificate = merged(release_pairs(&[("vyre::add", true)]));
        certificate["plan"]["selection"]["universe_backend_count"] = serde_json::json!(4);
        certificate["plan"]["selection"]["unavailable_backends"] = serde_json::json!([
            {"id": "metal", "reason": "unsupported feature `Apple Metal.framework native runtime`"}
        ]);
        assert_eq!(reported(&certificate), "");
    }

    /// WHY: an unacquired backend with no refusal states that something was
    /// missing and not what refused it, which is the same as not recording it.
    #[test]
    fn an_unacquired_backend_with_no_stated_refusal_is_a_finding() {
        let mut certificate = merged(release_pairs(&[("vyre::add", true)]));
        certificate["plan"]["selection"]["universe_backend_count"] = serde_json::json!(4);
        certificate["plan"]["selection"]["unavailable_backends"] =
            serde_json::json!([{"id": "metal", "reason": "   "}]);
        let rendered = reported(&certificate);
        assert!(
            rendered.contains("metal") && rendered.contains("refusal `<none>`"),
            "Fix: an unstated refusal must be a finding naming the backend, got {rendered}"
        );
    }

    /// WHY: a registered backend that is neither proved nor recorded as
    /// unacquirable is one the certificate is silent about, and silence reads
    /// as coverage. Every count in such a body agrees with every other: the
    /// plan, the selection and the pairs all say three, and the registered
    /// universe says four.
    #[test]
    fn a_registered_backend_that_is_neither_proved_nor_refused_is_a_finding() {
        let mut certificate = merged(release_pairs(&[("vyre::add", true)]));
        certificate["plan"]["selection"]["universe_backend_count"] = serde_json::json!(4);
        let rendered = reported(&certificate);
        assert!(
            rendered.contains("registers 4 backend(s), proved 3 and names 0")
                && rendered.contains("1 are unaccounted for"),
            "Fix: an unaccounted backend must be a finding naming the counts, got {rendered}"
        );
    }

    /// WHY: a merge that dropped one backend's pairs keeps the plan the shards
    /// stated, so the counts agree and the body carries two backends' rows
    /// under a three-backend plan.
    #[test]
    fn a_plan_naming_more_backends_than_the_pairs_carry_is_a_finding() {
        let mut certificate = merged(release_pairs(&[("vyre::add", true)]));
        certificate["pairs"] = Value::Array(vec![
            pair_on("cuda", "vyre::add", true),
            pair_on("wgpu", "vyre::add", true),
        ]);
        certificate["plan"]["pair_count"] = serde_json::json!(2);
        let rendered = reported(&certificate);
        assert!(
            rendered.contains("plans 3 backend(s) and its pairs name 2: cuda, wgpu"),
            "Fix: the finding must name the plan, the count and the executors, got {rendered}"
        );
    }

    /// WHY: an empty certificate has a plan that agrees with itself at zero, so
    /// every count check passes and the release has no conformance evidence.
    #[test]
    fn a_certificate_with_no_proved_pair_is_a_finding() {
        let rendered = reported(&merged(serde_json::json!([])));
        assert!(
            rendered.contains("carries no proved pair"),
            "Fix: an empty certificate must be a finding, got {rendered}"
        );
    }

    /// WHY: the committed body states its pairs under `backend_id`, which the
    /// conformance schema renamed to `executor_id` because a record label is
    /// not a device. A reader that tolerated the old name would accept a
    /// certificate no current run produces.
    #[test]
    fn a_pair_under_the_retired_field_name_is_a_finding() {
        let certificate = merged(serde_json::json!([
            {"backend_id": "cpu-ref", "op_id": "vyre::add", "passed": true, "message": "ok"}
        ]));
        let rendered = reported(&certificate);
        assert!(
            rendered.contains("vyre::add") && rendered.contains("names no `executor_id`"),
            "Fix: the retired field name must be a finding naming the operation, got {rendered}"
        );
    }

    /// WHY: `passed` absent is not `passed: false`. A pair with no outcome
    /// passed every loop that only looked for an explicit false.
    #[test]
    fn a_pair_with_no_outcome_is_a_finding() {
        let certificate = merged(serde_json::json!([
            {"op_id": "vyre::add", "executor_id": "cuda", "message": "ok"}
        ]));
        let rendered = reported(&certificate);
        assert!(
            rendered.contains("states no outcome"),
            "Fix: a pair with no outcome must be a finding, got {rendered}"
        );
    }

    /// WHY: every judgement reads one pointer, and a body with none of them is
    /// the shape a truncated or foreign document takes. Silence on it would
    /// record whatever it is as a release certificate.
    #[test]
    fn a_body_that_states_none_of_the_certificate_fields_is_refused() {
        let rendered = reported(&serde_json::json!({}));
        for expected in [
            "states no `backend_id`",
            "plans no pair count",
            "carries no `pairs` array",
        ] {
            assert!(
                rendered.contains(expected),
                "Fix: an empty body must report `{expected}`, got {rendered}"
            );
        }
    }

    /// WHY: a body that is valid JSON and not an object carries no field this
    /// gate reads, and indexing it would report every field as absent rather
    /// than the document as wrong.
    #[test]
    fn a_json_document_that_is_not_an_object_is_refused() {
        let message = match parse("[]") {
            Err(message) => message,
            Ok(_) => panic!("Fix: a JSON array must not parse as a certificate"),
        };
        assert!(
            message.contains("is not a JSON object"),
            "Fix: the refusal must name what the document is, got {message}"
        );
    }

    /// WHY: a clean certificate has to be silent, or every finding above is
    /// indistinguishable from noise the gate always emits.
    #[test]
    fn a_merged_certificate_with_every_pair_passing_is_silent() {
        let certificate = merged(release_pairs(&[("vyre::add", true), ("vyre::mul", true)]));
        assert_eq!(reported(&certificate), "");
    }

    /// WHY: the descriptor's artifact and the path this gate writes have to be
    /// one string, or the write is unowned and the committed file unattributed.
    #[test]
    fn the_descriptor_declares_exactly_the_artifact_this_gate_records() {
        let descriptor = xtask::gate_metadata::descriptor("release-certificate")
            .expect("Fix: `release-certificate` must be a registered gate descriptor");
        assert_eq!(descriptor.artifacts, [ARTIFACT]);
    }

    /// WHY: an unknown flag used to be ignored, so `--merged PATH` recorded
    /// whatever the default directory happened to hold.
    #[test]
    fn an_unknown_flag_is_refused_rather_than_ignored() {
        let args = vec!["--merged".to_string(), "merged.json".to_string()];
        assert!(parse_args(&args).is_err());
    }

    /// WHY: `--from` with no value used to leave the default in place, which
    /// silently recorded a stale merge.
    #[test]
    fn from_without_a_path_is_refused() {
        assert!(parse_args(&["--from".to_string()]).is_err());
    }
}
