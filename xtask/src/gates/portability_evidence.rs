//! Execute the host-identity corpus on every runnable host cell.
//!
//! `platform-support-matrix` publishes what each cell has been shown to do,
//! and reads that from `release/evidence/portability/host-identity.json`. This
//! gate is the only thing that writes that file, and it writes it by running
//! the corpus rather than by asserting a result.
//!
//! The cell set is not restated here. A cell is runnable when the workspace
//! cargo configuration declares how to reach it: `[target.<triple>]` supplies
//! the cross linker and, where the host cannot execute the binary directly,
//! the user-mode emulator. The triple this cargo builds for natively is the
//! remaining cell. A cell whose toolchain is absent is reported, never
//! skipped: a silent skip is how a matrix comes to claim a host nothing ran
//! on.
//!
//! A run under an emulator proves host arithmetic, byte order and decoding.
//! It is never evidence of device support or performance, and the evidence
//! class it records says so.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use crate::artifact_gate::{settle_inspection, Inspection};
use crate::gate::{Finding, GateBehavior, GateCtx, GateError, Report};
use crate::gates::platform_support_matrix::{
    PortabilityLedger, PortabilityRun, EVIDENCE_EMULATED, EVIDENCE_NATIVE, LEDGER_PATH,
    PORTABILITY_LEDGER_SCHEMA_VERSION,
};

/// Cargo configuration that declares how each cross cell is reached.
const CARGO_CONFIG_PATH: &str = ".cargo/config.toml";

/// Test target carrying the corpus, and the filter that selects it.
const IDENTITY_PACKAGE: &str = "vyre-foundation";
/// Integration test binary the corpus lives in.
const IDENTITY_TEST: &str = "all_tests";
/// Test-name filter that runs only the corpus.
const IDENTITY_FILTER: &str = "host_identity_invariance";

/// Source file carrying the one-cell pins the corpus asserts against.
const PIN_SOURCE_PATH: &str = "vyre-foundation/tests/host_identity_invariance.rs";
/// Constant holding the canonical corpus digest.
const CANONICAL_PIN: &str = "CANONICAL_CORPUS_IDENTITY";
/// Constant holding the structural-fallback corpus digest.
const FALLBACK_PIN: &str = "FALLBACK_CORPUS_IDENTITY";

/// Line prefix the corpus prints one digest on.
const DIGEST_MARKER: &str = "VYRE-HOST-IDENTITY ";

/// Portability evidence gate.
pub struct PortabilityEvidenceGate;

impl GateBehavior for PortabilityEvidenceGate {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let cells = runnable_cells(&ctx.root)?;
        let mut inspection = Inspection::new();

        let ledger = if ctx.write {
            let (ledger, findings) = execute_cells(&ctx.root, &cells);
            for finding in findings {
                inspection.find(finding);
            }
            ledger
        } else {
            read_ledger(&ctx.root)?
        };

        for finding in judge_ledger(&ledger, &cells) {
            inspection.find(finding);
        }

        let agreed = agreed_identities(&ledger);
        inspection.generates_document_text(PIN_SOURCE_PATH, pinned_source(&ctx.root, &agreed)?);
        inspection.generates_host_evidence(LEDGER_PATH, &ledger);

        let mut report = settle_inspection(ctx, ctx.gate_name()?, inspection);
        report.note(format!(
            "{} runnable cell(s), {} recorded run(s), identity {}",
            cells.len(),
            ledger.runs.len(),
            agreed.map_or_else(
                || "not agreed".to_string(),
                |(canonical, _)| canonical[..16].to_string()
            )
        ));
        Ok(report)
    }
}

/// One cell this host can reach, and how it reaches it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunnableCell {
    /// Target triple cargo builds for.
    pub target_triple: String,
    /// Host operating system identifier the triple names.
    pub os: String,
    /// Host CPU architecture identifier the triple names.
    pub arch: String,
    /// User-mode emulator the run needs, empty when the host executes directly.
    pub emulator: String,
}

impl RunnableCell {
    /// Evidence class a completed run of this cell records.
    #[must_use]
    pub fn evidence(&self) -> &'static str {
        if self.emulator.is_empty() {
            EVIDENCE_NATIVE
        } else {
            EVIDENCE_EMULATED
        }
    }
}

/// Every cell this workspace declares a way to run, native cell first.
///
/// # Errors
///
/// Returns `GateError` when the cargo configuration cannot be read or parsed,
/// or when the native target triple cannot be established.
pub fn runnable_cells(root: &Path) -> Result<Vec<RunnableCell>, GateError> {
    let mut cells = vec![native_cell(root)?];
    let text = std::fs::read_to_string(root.join(CARGO_CONFIG_PATH)).map_err(|error| {
        GateError::new(
            format!("cannot read {CARGO_CONFIG_PATH}: {error}"),
            "restore the workspace cargo configuration",
        )
    })?;
    let config = toml::from_str::<toml::Table>(&text).map_err(|error| {
        GateError::new(
            format!("cannot parse {CARGO_CONFIG_PATH}: {error}"),
            "repair the workspace cargo configuration",
        )
    })?;
    let Some(targets) = config.get("target").and_then(toml::Value::as_table) else {
        return Ok(cells);
    };
    for (triple, entry) in targets {
        if entry.get("linker").is_none() {
            continue;
        }
        if cells.iter().any(|cell| &cell.target_triple == triple) {
            continue;
        }
        let (os, arch) = classify_triple(triple).ok_or_else(|| {
            GateError::new(
                format!("cargo configuration declares target `{triple}`, which names no host cell"),
                "add the operating system and architecture to vyre_foundation::platform, or remove the target entry",
            )
        })?;
        let emulator = entry
            .get("runner")
            .and_then(runner_binary)
            .unwrap_or_default();
        cells.push(RunnableCell {
            target_triple: triple.clone(),
            os,
            arch,
            emulator,
        });
    }
    Ok(cells)
}

/// The emulator binary a `runner` entry names, whichever form it takes.
fn runner_binary(value: &toml::Value) -> Option<String> {
    match value {
        toml::Value::String(command) => Some(command.clone()),
        toml::Value::Array(parts) => parts
            .first()
            .and_then(toml::Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

/// The cell this cargo builds for with no `--target`.
fn native_cell(root: &Path) -> Result<RunnableCell, GateError> {
    let output = Command::new(crate::cargo_runner::binary(root))
        .arg("--version")
        .arg("--verbose")
        .current_dir(root)
        .output()
        .map_err(|error| {
            GateError::new(
                format!("cannot start cargo to read the native target triple: {error}"),
                "make the workspace cargo wrapper executable",
            )
        })?;
    let text = String::from_utf8_lossy(&output.stdout);
    let triple = text
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::trim)
        .ok_or_else(|| {
            GateError::new(
                "cargo --version --verbose printed no host triple",
                "run the gate through a cargo that reports its host",
            )
        })?
        .to_string();
    let (os, arch) = classify_triple(&triple).ok_or_else(|| {
        GateError::new(
            format!("native target `{triple}` names no host cell"),
            "add the operating system and architecture to vyre_foundation::platform",
        )
    })?;
    Ok(RunnableCell {
        target_triple: triple,
        os,
        arch,
        emulator: String::new(),
    })
}

/// The `(os, arch)` identifiers a target triple names.
///
/// The identifiers are the ones `HostOs::id` and `HostArch::id` return, which
/// is what the support matrix keys its cells on. A triple whose components
/// neither enum names yields nothing, and the caller reports it: a cell the
/// platform source cannot express must not reach the ledger under a guess.
#[must_use]
pub fn classify_triple(triple: &str) -> Option<(String, String)> {
    let arch = match triple.split('-').next()? {
        "x86_64" => "x86_64",
        "i586" | "i686" => "x86",
        "aarch64" => "aarch64",
        "armv7" => "armv7",
        "riscv64gc" | "riscv64" => "riscv64",
        "powerpc" => "powerpc",
        "s390x" => "s390x",
        "wasm32" => "wasm32",
        _ => return None,
    };
    let os = if triple.contains("-linux-") || triple.ends_with("-linux") {
        if triple.contains("-android") {
            "android"
        } else {
            "linux"
        }
    } else if triple.contains("-android") {
        "android"
    } else if triple.contains("-darwin") {
        "macos"
    } else if triple.contains("-ios") {
        "ios"
    } else if triple.contains("-windows-") {
        "windows"
    } else if triple.contains("-freebsd") {
        "freebsd"
    } else {
        return None;
    };
    Some((os.to_string(), arch.to_string()))
}

/// Run the corpus on every cell and record what each one printed.
fn execute_cells(root: &Path, cells: &[RunnableCell]) -> (PortabilityLedger, Vec<Finding>) {
    let mut runs = Vec::with_capacity(cells.len());
    let mut findings = Vec::new();
    for cell in cells {
        match execute_cell(root, cell) {
            Ok(run) => runs.push(run),
            Err(finding) => findings.push(finding),
        }
    }
    (
        PortabilityLedger {
            schema_version: PORTABILITY_LEDGER_SCHEMA_VERSION,
            runs,
        },
        findings,
    )
}

/// Run the corpus on one cell.
///
/// The corpus prints each digest before it asserts against its pin, so a run
/// whose pins are stale still reports what this cell computed. A run that
/// printed no digest is a failure of the run itself and is reported as one.
fn execute_cell(root: &Path, cell: &RunnableCell) -> Result<PortabilityRun, Finding> {
    let mut command = crate::cargo_runner::command(root);
    command
        .arg("test")
        .arg("-p")
        .arg(IDENTITY_PACKAGE)
        .arg("--offline")
        .arg("--test")
        .arg(IDENTITY_TEST);
    if cell.emulator.is_empty() && cell.target_triple == native_triple_of(root) {
        // No --target: the native build reuses the workspace's own units.
    } else {
        command.arg("--target").arg(&cell.target_triple);
    }
    command
        .arg("--")
        .arg(IDENTITY_FILTER)
        .arg("--nocapture")
        .arg("--test-threads=1");

    // The corpus prints its digests on standard output, so both streams are
    // captured here rather than through the streaming runner, which inherits
    // standard output and would hand this gate only the compiler's stderr.
    let captured = command.output().map_err(|error| {
        Finding::in_file(
            LEDGER_PATH,
            format!(
                "cell {}/{} on `{}` could not be started: {error}",
                cell.os, cell.arch, cell.target_triple
            ),
            format!(
                "install the toolchain the cell needs: rustup target add {}{}",
                cell.target_triple,
                if cell.emulator.is_empty() {
                    String::new()
                } else {
                    format!(", and the {} emulator", cell.emulator)
                }
            ),
        )
    })?;
    let mut output = String::from_utf8_lossy(&captured.stdout).into_owned();
    let errors = String::from_utf8_lossy(&captured.stderr);
    eprint!("{errors}");
    output.push_str(&errors);

    let digests = printed_digests(&output);
    let canonical = digests
        .get("canonical")
        .cloned()
        .ok_or_else(|| missing_digest(cell, "canonical", &output))?;
    let fallback = digests
        .get("fallback")
        .cloned()
        .ok_or_else(|| missing_digest(cell, "fallback", &output))?;

    Ok(PortabilityRun {
        os: cell.os.clone(),
        arch: cell.arch.clone(),
        target_triple: cell.target_triple.clone(),
        evidence: cell.evidence().to_string(),
        emulator: cell.emulator.clone(),
        canonical_identity: canonical,
        fallback_identity: fallback,
    })
}

/// The native triple, cached per call site rather than re-derived on failure.
fn native_triple_of(root: &Path) -> String {
    native_cell(root).map_or_else(|_| String::new(), |cell| cell.target_triple)
}

/// The finding a cell that printed no digest reports.
fn missing_digest(cell: &RunnableCell, which: &str, output: &str) -> Finding {
    let tail: String = output.lines().rev().take(4).collect::<Vec<_>>().join(" | ");
    Finding::in_file(
        LEDGER_PATH,
        format!(
            "cell {}/{} on `{}` printed no {which} identity: {tail}",
            cell.os, cell.arch, cell.target_triple
        ),
        format!("make `{IDENTITY_FILTER}` reach its digest print on this cell before recording it"),
    )
}

/// Every `VYRE-HOST-IDENTITY <label> <digest>` line the run printed.
#[must_use]
pub fn printed_digests(output: &str) -> BTreeMap<String, String> {
    let mut digests = BTreeMap::new();
    for line in output.lines() {
        // The marker is found anywhere in the line, not at its start. With one
        // test thread and no capture, libtest writes its own `test <name> ... `
        // progress prefix and the test's first println onto the same line, so a
        // prefix match read every digest as absent and reported three agreeing
        // cells as three cells that printed nothing.
        let Some(offset) = line.find(DIGEST_MARKER) else {
            continue;
        };
        let rest = &line[offset + DIGEST_MARKER.len()..];
        let mut parts = rest.split_whitespace();
        let (Some(label), Some(digest)) = (parts.next(), parts.next()) else {
            continue;
        };
        digests.insert(label.to_string(), digest.to_string());
    }
    digests
}

/// The identity every recorded run agrees on, when they all agree.
#[must_use]
pub fn agreed_identities(ledger: &PortabilityLedger) -> Option<(String, String)> {
    let first = ledger.runs.first()?;
    let agreed = ledger.runs.iter().all(|run| {
        run.canonical_identity == first.canonical_identity
            && run.fallback_identity == first.fallback_identity
    });
    agreed.then(|| {
        (
            first.canonical_identity.clone(),
            first.fallback_identity.clone(),
        )
    })
}

/// Name every way the ledger fails to cover the runnable cells.
#[must_use]
pub fn judge_ledger(ledger: &PortabilityLedger, cells: &[RunnableCell]) -> Vec<Finding> {
    let mut findings = Vec::new();
    if ledger.schema_version != PORTABILITY_LEDGER_SCHEMA_VERSION {
        findings.push(Finding::in_file(
            LEDGER_PATH,
            format!(
                "portability ledger schema version {} is not {PORTABILITY_LEDGER_SCHEMA_VERSION}",
                ledger.schema_version
            ),
            "re-record the runs with `cargo xtask portability-evidence --write`",
        ));
    }
    for cell in cells {
        if !ledger
            .runs
            .iter()
            .any(|run| run.target_triple == cell.target_triple)
        {
            findings.push(Finding::in_file(
                LEDGER_PATH,
                format!(
                    "cell {}/{} on `{}` is runnable here and has no recorded run",
                    cell.os, cell.arch, cell.target_triple
                ),
                "record it with `cargo xtask portability-evidence --write`",
            ));
        }
    }
    let Some(first) = ledger.runs.first() else {
        return findings;
    };
    for run in ledger.runs.iter().skip(1) {
        if run.canonical_identity != first.canonical_identity {
            findings.push(Finding::in_file(
                LEDGER_PATH,
                format!(
                    "`{}` computed canonical identity {} and `{}` computed {}",
                    run.target_triple,
                    run.canonical_identity,
                    first.target_triple,
                    first.canonical_identity
                ),
                "a persisted identity must not depend on the host; find the value reaching the digest through usize or native byte order",
            ));
        }
        if run.fallback_identity != first.fallback_identity {
            findings.push(Finding::in_file(
                LEDGER_PATH,
                format!(
                    "`{}` computed fallback identity {} and `{}` computed {}",
                    run.target_triple,
                    run.fallback_identity,
                    first.target_triple,
                    first.fallback_identity
                ),
                "a persisted identity must not depend on the host; find the value reaching the digest through usize or native byte order",
            ));
        }
    }
    findings
}

/// Read the committed ledger, or report that there is none.
fn read_ledger(root: &Path) -> Result<PortabilityLedger, GateError> {
    let path = root.join(LEDGER_PATH);
    let text = std::fs::read_to_string(&path).map_err(|error| {
        GateError::new(
            format!("cannot read {LEDGER_PATH}: {error}"),
            "record the runs with `cargo xtask portability-evidence --write`",
        )
    })?;
    let (_, body) = crate::artifact_gate::split_provenance(&text);
    serde_json::from_str(&body).map_err(|error| {
        GateError::new(
            format!("cannot parse {LEDGER_PATH}: {error}"),
            "re-record the runs with `cargo xtask portability-evidence --write`",
        )
    })
}

/// The corpus source carrying the pins the agreed identity settles them to.
///
/// The pins live beside the corpus that computes them, so a developer who
/// moves the encoder sees the failure in the test rather than in a gate. They
/// are still this gate's to write: the value they carry is only trustworthy
/// once every cell has agreed on it, so a run with no agreement renders the
/// source unchanged and leaves the disagreement to be reported on its own.
///
/// Writing goes through the declared artifact set rather than through this
/// function. A gate that writes a file itself writes it whether or not
/// `--write` was passed, and it writes a path the mutation guard never
/// charged it for.
///
/// # Errors
///
/// Returns `GateError` when the corpus cannot be read or no longer carries the
/// pinned constants.
fn pinned_source(root: &Path, agreed: &Option<(String, String)>) -> Result<String, GateError> {
    let source = std::fs::read_to_string(root.join(PIN_SOURCE_PATH)).map_err(|error| {
        GateError::new(
            format!("cannot read {PIN_SOURCE_PATH}: {error}"),
            "restore the host identity corpus",
        )
    })?;
    let Some((canonical, fallback)) = agreed else {
        return Ok(source);
    };
    let updated = replace_pin(&source, CANONICAL_PIN, canonical)?;
    replace_pin(&updated, FALLBACK_PIN, fallback)
}

/// Return `source` with `name`'s string literal set to `digest`.
///
/// # Errors
///
/// Returns `GateError` when the constant is absent or does not hold a string
/// literal, which means the corpus no longer carries the pin this gate writes.
pub fn replace_pin(source: &str, name: &str, digest: &str) -> Result<String, GateError> {
    let anchor = format!("const {name}: &str =");
    let start = source.find(&anchor).ok_or_else(|| {
        GateError::new(
            format!("{PIN_SOURCE_PATH} declares no `{name}`"),
            "restore the pinned constant the host identity corpus asserts against",
        )
    })?;
    let tail = &source[start + anchor.len()..];
    let open = tail.find('"').ok_or_else(|| {
        GateError::new(
            format!("`{name}` does not hold a string literal"),
            "declare the pin as a plain string literal this gate can write",
        )
    })?;
    let close = tail[open + 1..].find('"').ok_or_else(|| {
        GateError::new(
            format!("`{name}` has an unterminated string literal"),
            "declare the pin as a plain string literal this gate can write",
        )
    })?;
    let literal_start = start + anchor.len() + open + 1;
    let literal_end = literal_start + close;
    let mut updated = String::with_capacity(source.len());
    updated.push_str(&source[..literal_start]);
    updated.push_str(digest);
    updated.push_str(&source[literal_end..]);
    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A triple this workspace configures must resolve to a platform cell.
    ///
    /// WHY: the ledger keys every run on the identifiers the support matrix
    /// publishes. A triple that classified to a guess would record a run
    /// against a cell nothing executed on.
    #[test]
    fn every_configured_target_names_a_platform_cell() {
        for triple in [
            "x86_64-unknown-linux-gnu",
            "i686-unknown-linux-gnu",
            "powerpc-unknown-linux-gnu",
            "s390x-unknown-linux-gnu",
            "aarch64-apple-darwin",
            "x86_64-pc-windows-msvc",
        ] {
            assert!(
                classify_triple(triple).is_some(),
                "`{triple}` must resolve to a host cell"
            );
        }
    }

    /// A triple no platform variant names resolves to nothing.
    #[test]
    fn an_unnamed_target_resolves_to_no_cell() {
        assert_eq!(classify_triple("sparc64-unknown-netbsd"), None);
        assert_eq!(classify_triple("x86_64-unknown-none"), None);
    }

    /// The android arm is reached before the linux arm.
    ///
    /// WHY: an android triple contains `-linux-`, so an order that tested
    /// linux first would record every android run as a linux run.
    #[test]
    fn an_android_target_is_not_recorded_as_linux() {
        assert_eq!(
            classify_triple("aarch64-linux-android"),
            Some(("android".to_string(), "aarch64".to_string()))
        );
    }

    /// Digest lines are read out of surrounding test output.
    #[test]
    fn a_digest_is_read_from_the_run_output() {
        let output = "running 4 tests\nVYRE-HOST-IDENTITY canonical abc123\nVYRE-HOST-CELL pointer_width=64 endian=little\nVYRE-HOST-IDENTITY fallback def456\ntest result: ok.\n";
        let digests = printed_digests(output);
        assert_eq!(digests.get("canonical").map(String::as_str), Some("abc123"));
        assert_eq!(digests.get("fallback").map(String::as_str), Some("def456"));
    }

    /// A digest libtest wrote after its own progress prefix is still read.
    ///
    /// WHY: with one test thread and no capture, libtest emits
    /// `test <name> ... ` and the test's first println on one line. A reader
    /// that matched only at the start of a line found no digest on any cell
    /// and reported three agreeing hosts as three hosts that printed nothing.
    #[test]
    fn a_digest_behind_a_libtest_progress_prefix_is_read() {
        let output = "test host_identity_invariance::canonical_identity_is_byte_identical_on_every_host_cell ... VYRE-HOST-IDENTITY canonical 74d5\ntest host_identity_invariance::fallback_identity_is_byte_identical_on_every_host_cell ... VYRE-HOST-IDENTITY fallback 9a73\n";
        let digests = printed_digests(output);
        assert_eq!(digests.get("canonical").map(String::as_str), Some("74d5"));
        assert_eq!(digests.get("fallback").map(String::as_str), Some("9a73"));
    }

    /// Two cells that disagree yield no agreed identity.
    ///
    /// WHY: the pin this gate writes certifies that every cell computed the
    /// same bytes. Writing the first cell's value when a second disagreed
    /// would pin a host-dependent identity as canonical.
    #[test]
    fn disagreeing_cells_agree_on_nothing() {
        let ledger = PortabilityLedger {
            schema_version: PORTABILITY_LEDGER_SCHEMA_VERSION,
            runs: vec![
                run("x86_64-unknown-linux-gnu", "aa", "bb"),
                run("s390x-unknown-linux-gnu", "cc", "bb"),
            ],
        };
        assert_eq!(agreed_identities(&ledger), None);
        assert!(judge_ledger(&ledger, &[])
            .iter()
            .any(|finding| finding.message.contains("computed canonical identity")));
    }

    /// Cells that agree yield the identity, and no disagreement finding.
    #[test]
    fn agreeing_cells_yield_one_identity() {
        let ledger = PortabilityLedger {
            schema_version: PORTABILITY_LEDGER_SCHEMA_VERSION,
            runs: vec![
                run("x86_64-unknown-linux-gnu", "aa", "bb"),
                run("s390x-unknown-linux-gnu", "aa", "bb"),
            ],
        };
        assert_eq!(
            agreed_identities(&ledger),
            Some(("aa".to_string(), "bb".to_string()))
        );
        assert!(judge_ledger(&ledger, &[]).is_empty());
    }

    /// A runnable cell with no recorded run is reported.
    #[test]
    fn an_unrecorded_runnable_cell_is_reported() {
        let ledger = PortabilityLedger {
            schema_version: PORTABILITY_LEDGER_SCHEMA_VERSION,
            runs: vec![run("x86_64-unknown-linux-gnu", "aa", "bb")],
        };
        let cells = vec![RunnableCell {
            target_triple: "s390x-unknown-linux-gnu".to_string(),
            os: "linux".to_string(),
            arch: "s390x".to_string(),
            emulator: "qemu-s390x-static".to_string(),
        }];
        assert!(judge_ledger(&ledger, &cells)
            .iter()
            .any(|finding| finding.message.contains("has no recorded run")));
    }

    /// An emulated cell records a weaker evidence class than a native one.
    #[test]
    fn an_emulated_cell_is_not_recorded_as_native() {
        let emulated = RunnableCell {
            target_triple: "s390x-unknown-linux-gnu".to_string(),
            os: "linux".to_string(),
            arch: "s390x".to_string(),
            emulator: "qemu-s390x-static".to_string(),
        };
        let direct = RunnableCell {
            emulator: String::new(),
            ..emulated.clone()
        };
        assert_eq!(emulated.evidence(), EVIDENCE_EMULATED);
        assert_eq!(direct.evidence(), EVIDENCE_NATIVE);
    }

    /// The pin writer replaces the literal and nothing around it.
    #[test]
    fn a_pin_is_replaced_in_place() {
        let source = "const CANONICAL_CORPUS_IDENTITY: &str =\n    \"0000\";\nconst OTHER: &str = \"0000\";\n";
        let updated = replace_pin(source, "CANONICAL_CORPUS_IDENTITY", "abcd").expect("pin exists");
        assert_eq!(
            updated,
            "const CANONICAL_CORPUS_IDENTITY: &str =\n    \"abcd\";\nconst OTHER: &str = \"0000\";\n"
        );
    }

    /// A missing pin is an error, not a silent no-op.
    #[test]
    fn a_missing_pin_is_reported() {
        assert!(replace_pin(
            "const OTHER: &str = \"0\";",
            "CANONICAL_CORPUS_IDENTITY",
            "a"
        )
        .is_err());
    }

    fn run(triple: &str, canonical: &str, fallback: &str) -> PortabilityRun {
        PortabilityRun {
            os: "linux".to_string(),
            arch: triple.split('-').next().unwrap_or_default().to_string(),
            target_triple: triple.to_string(),
            evidence: EVIDENCE_NATIVE.to_string(),
            emulator: String::new(),
            canonical_identity: canonical.to_string(),
            fallback_identity: fallback.to_string(),
        }
    }
}
