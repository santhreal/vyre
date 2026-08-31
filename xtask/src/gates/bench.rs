//! The gates that hold the benchmark surface to what it claims.
//!
//! Three shell scripts owned this before: one read `benches/RESULTS.md`, one
//! timed the smoke suite against `contracts/perf_targets.toml`, and one asked
//! the `vyre-bench` registry whether every measured dimension still has a case.
//! None had a baseline row, none was swept, and each exited on its first
//! failure, so a tree with four broken rows reported one.
//!
//! The judgment is separated from the spawn on purpose. Deciding which crates
//! owe a measured section, which dimensions are covered and which `--case`
//! references resolve is a pure function of the tree and a case-id set, so it is
//! tested without building `vyre-bench`. Only the registry listing and the smoke
//! run need cargo.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{numbered, Tree};

/// The published baseline every bench-bearing crate is measured into.
const RESULTS: &str = "benches/RESULTS.md";

/// Registry-derived semantic families whose optimizer baselines must be published.
const OPTIMIZATION_FAMILY_MANIFEST: &str =
    "release/evidence/optimization/optimization-family-manifest.json";

/// Header fields a baseline must carry to be reproducible by a reader.
///
/// A median with no machine, no toolchain and no commit is a number, not a
/// measurement: nobody can tell whether a later run disagrees because the code
/// changed or because the host did.
const REQUIRED_FIELDS: &[&str] = &["machine:", "gpu:", "cpu:", "rustc:", "commit:"];

/// Every crate with a bench target has published numbers under `benches/RESULTS.md`.
pub struct BenchBaselines;

impl crate::gate::GateBehavior for BenchBaselines {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::default();
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }
        let packages = bench_bearing_packages(&tree)?;
        report.cover_complete("bench-bearing packages", packages.len());
        if !tree.has(RESULTS) {
            report.find(Finding::in_file(
                RESULTS,
                format!("`{RESULTS}` is not published, so no benchmark claim in this repository is reproducible"),
                "run the bench targets, record the medians with the host and toolchain that produced them, and commit the file",
            ));
            return Ok(report);
        }
        let text = tree.read(RESULTS)?;
        for field in REQUIRED_FIELDS {
            if !text.contains(field) {
                report.find(Finding::in_file(
                    RESULTS,
                    format!("the baseline header carries no `{field}` field"),
                    format!("state `{field}` in the header so a reader can reproduce the run"),
                ));
            }
        }
        for package in packages {
            if !has_section(&text, &package) {
                report.find(Finding::in_file(
                    RESULTS,
                    format!("`{package}` declares a bench target but has no `### {package}` section"),
                    format!("run `./cargo_full bench -p {package}` and record the medians under a `### {package}` heading"),
                ));
            }
        }
        require_optimizer_family_baselines(&tree, &text, &mut report)?;
        Ok(report)
    }
}

/// Whether the baseline carries a level-three section for one package.
fn has_section(text: &str, package: &str) -> bool {
    let heading = format!("### {package}");
    text.lines()
        .any(|line| line.trim_end() == heading || line.starts_with(&format!("{heading} ")))
}

/// Require one independently measured optimizer baseline for every registered family.
fn require_optimizer_family_baselines(
    tree: &Tree,
    results: &str,
    report: &mut Report,
) -> Result<(), GateError> {
    if !tree.has(OPTIMIZATION_FAMILY_MANIFEST) {
        report.find(Finding::in_file(
            OPTIMIZATION_FAMILY_MANIFEST,
            "the optimizer family manifest is absent, so benchmark-family coverage cannot be judged",
            "regenerate optimization evidence with `./cargo_full run -p xtask -- optimization-corpus --write`",
        ));
        return Ok(());
    }
    let manifest: serde_json::Value =
        serde_json::from_str(&tree.read(OPTIMIZATION_FAMILY_MANIFEST)?).map_err(|error| {
            GateError::new(
                format!("cannot parse JSON `{OPTIMIZATION_FAMILY_MANIFEST}`: {error}"),
                "regenerate optimization evidence with `./cargo_full run -p xtask -- optimization-corpus --write`",
            )
        })?;
    let families = manifest
        .get("required_families")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            GateError::new(
                format!(
                    "`{OPTIMIZATION_FAMILY_MANIFEST}` has no `required_families` array"
                ),
                "regenerate optimization evidence with `./cargo_full run -p xtask -- optimization-corpus --write`",
            )
        })?;
    if families.is_empty() {
        return Err(GateError::new(
            format!(
                "`{OPTIMIZATION_FAMILY_MANIFEST}` declares no required optimizer families"
            ),
            "regenerate optimization evidence with `./cargo_full run -p xtask -- optimization-corpus --write`",
        ));
    }
    report.cover_complete("optimizer benchmark families", families.len());
    for family in families {
        let family = family.as_str().ok_or_else(|| {
            GateError::new(
                format!(
                    "`{OPTIMIZATION_FAMILY_MANIFEST}` contains a non-string optimizer family"
                ),
                "regenerate optimization evidence with `./cargo_full run -p xtask -- optimization-corpus --write`",
            )
        })?;
        let row = format!("| optimizer/pipeline/corpus/{family} |");
        if !results.lines().any(|line| line.starts_with(&row)) {
            report.find(Finding::in_file(
                RESULTS,
                format!(
                    "registered optimizer family `{family}` has no independent Criterion baseline"
                ),
                format!(
                    "run `./cargo_full bench -p vyre-foundation --bench optimizer_pipeline` and record `optimizer/pipeline/corpus/{family}`"
                ),
            ));
        }
    }
    Ok(())
}

/// Every workspace package `cargo bench` can run a target for.
///
/// A crate qualifies by owning a bench target, not by owning a directory called
/// `benches`: a `benches/` directory holding documentation and no target made a
/// directory-name search demand a measured section for a crate `cargo bench`
/// cannot run, which could only be satisfied with an invented number.
///
/// Both target shapes count, because both publish a benchmark. An explicit
/// `[[bench]]` table declares one wherever its source lives, and autodiscovery
/// declares one for `benches/<name>.rs` and `benches/<name>/main.rs` unless the
/// manifest sets `autobenches = false`. The package name comes from the
/// manifest rather than the directory name, so a crate nested under another
/// directory is named as `cargo bench -p` accepts it.
fn bench_bearing_packages(tree: &Tree) -> Result<Vec<String>, GateError> {
    let mut packages = BTreeSet::new();
    for member in tree.member_manifests()? {
        let declares_explicitly = member
            .manifest
            .get("bench")
            .and_then(toml::Value::as_array)
            .is_some_and(|entries| !entries.is_empty());
        let autobenches = member
            .manifest
            .get("package")
            .and_then(|package| package.get("autobenches"))
            .and_then(toml::Value::as_bool)
            .unwrap_or(true);
        let discovered = autobenches && has_discoverable_bench(tree, &member.path);
        if declares_explicitly || discovered {
            packages.insert(member.name);
        }
    }
    Ok(packages.into_iter().collect())
}

/// Whether autodiscovery would find a bench target in a member directory.
fn has_discoverable_bench(tree: &Tree, member: &str) -> bool {
    let prefix = format!("{member}/benches/");
    tree.paths().iter().any(|path| {
        let Some(path) = path.to_str() else {
            return false;
        };
        let Some(rest) = path.strip_prefix(&prefix) else {
            return false;
        };
        match rest.split_once('/') {
            None => rest.ends_with(".rs"),
            Some((_, tail)) => tail == "main.rs",
        }
    })
}

/// The perf-target manifest the smoke budget is declared in.
const PERF_TARGETS: &str = "contracts/perf_targets.toml";

/// The perf target the smoke gate enforces.
const SMOKE_TARGET: (&str, &str) = ("vyre-bench", "smoke_runtime");

/// The canonical smoke case the budget is measured over.
const SMOKE_CASE: &str = "foundation.elementwise.add.1m";

/// One correctness-preserving warmup run and the measured warm-state run share this shape.
const SMOKE_ARGS: &[&str] = &[
    "run",
    "--suite",
    "smoke",
    "--format",
    "json",
    "--case",
    SMOKE_CASE,
    "--warmup-samples",
    "0",
    "--measured-samples",
    "30",
    "--sample-timeout-secs",
    "30",
    "--determinism-runs",
    "1",
];

/// The canonical smoke samples execute inside their declared budget.
///
/// The harness owns each measured sample interval. Environment probes, process
/// startup, backend acquisition, and one-time artifact preparation are setup,
/// and release evidence records their cold costs separately.
pub struct BenchSmokeRuntime;

impl crate::gate::GateBehavior for BenchSmokeRuntime {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let budget_ms = smoke_budget_ms(&tree.read_toml(PERF_TARGETS)?)?;
        let binary = build_bench_binary(&ctx.root)?;
        // Listing first: a registry that cannot be listed measures nothing.
        run_bench(&ctx.root, &binary, &["list", "--format", "json"])?;
        let output = run_bench(&ctx.root, &binary, SMOKE_ARGS)?;
        let measured_ms = smoke_measured_ms(&output)?;
        let mut report = Report::default();
        report.cover_complete("runtime smoke benchmark", 1);
        if measured_ms > budget_ms {
            report.find(Finding::in_file(
                PERF_TARGETS,
                format!(
                    "the `{}.{}` smoke suite took {measured_ms}ms against a {budget_ms}ms budget",
                    SMOKE_TARGET.0, SMOKE_TARGET.1
                ),
                "reduce the canonical smoke runtime, or move the heavy cases to the release or deep suites",
            ));
        } else {
            report.note(format!("smoke suite took {measured_ms}ms of {budget_ms}ms"));
        }
        Ok(report)
    }
}

/// The declared millisecond budget of the smoke runtime target.
fn smoke_budget_ms(manifest: &toml::Table) -> Result<u64, GateError> {
    let (package, target) = SMOKE_TARGET;
    let budget = manifest
        .get("crates")
        .and_then(|crates| crates.get(package))
        .and_then(|entry| entry.get("targets"))
        .and_then(|targets| targets.get(target))
        .and_then(|row| row.get("budget"))
        .and_then(toml::Value::as_integer)
        .ok_or_else(|| {
            GateError::new(
                format!(
                    "`{PERF_TARGETS}` declares no integer budget for `crates.{package}.targets.{target}`"
                ),
                "declare the budget in the manifest; the gate enforces the published number and holds none of its own",
            )
        })?;
    u64::try_from(budget).map_err(|_| {
        GateError::new(
            format!("`{PERF_TARGETS}` declares a negative budget for `crates.{package}.targets.{target}`"),
            "declare a positive millisecond budget",
        )
    })
}

/// Total milliseconds across the harness-owned measured samples, rounded up.
fn smoke_measured_ms(stdout: &[u8]) -> Result<u64, GateError> {
    let report: serde_json::Value = serde_json::from_slice(stdout).map_err(|error| {
        GateError::new(
            format!("vyre-bench smoke run emitted invalid JSON: {error}"),
            "repair the benchmark report serializer; the gate cannot judge an unreadable runtime",
        )
    })?;
    let case = report
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .and_then(|cases| {
            cases
                .iter()
                .find(|case| case.get("id").and_then(serde_json::Value::as_str) == Some(SMOKE_CASE))
        })
        .ok_or_else(|| {
            GateError::new(
                format!("vyre-bench smoke report contains no `{SMOKE_CASE}` case"),
                "restore the canonical smoke case to the benchmark report",
            )
        })?;
    let wall = case.pointer("/metrics/wall_ns").ok_or_else(|| {
        GateError::new(
            format!("vyre-bench smoke case `{SMOKE_CASE}` has no `wall_ns` sample statistics"),
            "restore measured wall-clock statistics to the canonical smoke case",
        )
    })?;
    let mean_ns = wall
        .get("mean")
        .and_then(serde_json::Value::as_f64)
        .filter(|mean| mean.is_finite() && *mean >= 0.0)
        .ok_or_else(|| {
            GateError::new(
                format!(
                    "vyre-bench smoke case `{SMOKE_CASE}` has no finite nonnegative `wall_ns.mean`"
                ),
                "restore measured wall-clock statistics to the canonical smoke case",
            )
        })?;
    let samples = wall
        .get("samples")
        .and_then(serde_json::Value::as_u64)
        .filter(|samples| *samples >= 30)
        .ok_or_else(|| {
            GateError::new(
                format!("vyre-bench smoke case `{SMOKE_CASE}` records fewer than 30 wall-clock samples"),
                "restore the canonical 30-sample floor; the smoke budget may not be judged from fewer samples",
            )
        })?;
    let measured_ms = (mean_ns * samples as f64 / 1_000_000.0).ceil();
    if measured_ms > u64::MAX as f64 {
        return Err(GateError::new(
            format!("vyre-bench smoke case `{SMOKE_CASE}` reports a wall-clock total too large for milliseconds"),
            "repair the benchmark wall-clock statistics",
        ));
    }
    Ok(measured_ms as u64)
}

/// Build `vyre-bench` and return the binary cargo produced.
///
/// The binary is located and invoked directly rather than run through
/// `cargo run`, so the measured interval is the suite and not cargo's own
/// freshness check. The release profile is part of the measured target's
/// contract; every other build setting remains in workspace Cargo configuration.
fn build_bench_binary(root: &Path) -> Result<PathBuf, GateError> {
    let cargo = crate::cargo_runner::binary(root);
    let build = Command::new(&cargo)
        .args(["build", "-q", "--release", "-p", "vyre-bench"])
        .current_dir(root)
        .status()
        .map_err(|error| {
            GateError::new(
                format!(
                    "cannot run `{} build --release -p vyre-bench`: {error}",
                    cargo.display()
                ),
                "restore the cargo_full wrapper at the workspace root",
            )
        })?;
    if !build.success() {
        return Err(GateError::new(
            format!(
                "`cargo build --release -p vyre-bench` exited {}",
                build.code().unwrap_or(-1)
            ),
            "build vyre-bench by hand and fix what it reports; an unbuildable harness measures nothing",
        ));
    }
    let metadata = Command::new(&cargo)
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(root)
        .output()
        .map_err(|error| {
            GateError::new(
                format!("cannot run `{} metadata`: {error}", cargo.display()),
                "restore the cargo_full wrapper at the workspace root",
            )
        })?;
    if !metadata.status.success() {
        return Err(GateError::new(
            format!(
                "`cargo metadata` exited {}: {}",
                metadata.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&metadata.stderr).trim()
            ),
            "repair the workspace manifests so cargo can describe the workspace",
        ));
    }
    let described: serde_json::Value =
        serde_json::from_slice(&metadata.stdout).map_err(|error| {
            GateError::new(
                format!("`cargo metadata` did not emit JSON: {error}"),
                "run cargo metadata by hand and fix what it reports",
            )
        })?;
    let target_directory = described
        .get("target_directory")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            GateError::new(
                "`cargo metadata` named no target_directory",
                "repair the cargo configuration that declares the target directory",
            )
        })?;
    let binary = Path::new(target_directory)
        .join("release")
        .join("vyre-bench");
    if !binary.is_file() {
        return Err(GateError::new(
            format!(
                "`vyre-bench` built but no binary exists at `{}`",
                binary.display()
            ),
            "repair the cargo target directory, or the vyre-bench binary target",
        ));
    }
    Ok(binary)
}

/// Run the benchmark binary, failing the gate when it cannot run.
fn run_bench(root: &Path, binary: &Path, arguments: &[&str]) -> Result<Vec<u8>, GateError> {
    let output = Command::new(binary)
        .args(arguments)
        .current_dir(root)
        .output()
        .map_err(|error| {
            GateError::new(
                format!(
                    "cannot run `{} {}`: {error}",
                    binary.display(),
                    arguments.join(" ")
                ),
                "rebuild vyre-bench; a harness that cannot start measures nothing",
            )
        })?;
    if !output.status.success() {
        return Err(GateError::new(
            format!(
                "`vyre-bench {}` exited {}: {}",
                arguments.join(" "),
                output.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            "run the same command by hand and fix what it reports",
        ));
    }
    Ok(output.stdout)
}

/// One registered case per measured dimension.
///
/// A dimension whose case was renamed or deleted stops being measured, and the
/// harness says nothing, because a registry with fewer cases is still a valid
/// registry.
const REPRESENTATIVE: &[(&str, &str)] = &[
    ("latency", "runtime.megakernel.dispatch.256"),
    ("memory", "primitives.graph.frontier_step.1m"),
    ("optimizer", "foundation.optimizer.impact"),
    ("runtime_queueing", "runtime.megakernel.condition.64k"),
    ("throughput", "foundation.dfa_match.256k"),
];

/// The dimension whose evidence is a named test rather than a registered case.
const CACHE_CONTRACT: &str = "vyre-driver-cuda/tests/module_cache_contracts.rs";

/// The test that pins module-cache reuse.
const CACHE_TEST: &str = "repeated_dispatch_reuses_loaded_cuda_module";

/// File kinds where `--case` is an invocation rather than prose.
///
/// `--case` inside Rust source is a help or error string, never a command, so
/// those files are not scanned.
const REFERENCE_EXTENSIONS: &[&str] = &["yml", "yaml", "sh", "json", "toml", "md"];

/// Every measured dimension has a registered case, and every `--case` a tracked
/// file names resolves.
///
/// Most `--case` references live in `gpu-parity.yml`, which runs only on the
/// self-hosted GPU runner, and in release evidence manifests that run only at
/// release time. A rename breaks them where nobody is watching. Here it breaks
/// in PR CI, on a runner that needs no GPU, because listing the registry
/// measures nothing.
pub struct BenchCoverage;

impl crate::gate::GateBehavior for BenchCoverage {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let registered = registered_cases(&ctx.root)?;
        judge_coverage(&tree, &registered)
    }
}

/// The case ids the `vyre-bench` registry lists.
fn registered_cases(root: &Path) -> Result<BTreeSet<String>, GateError> {
    let cargo = crate::cargo_runner::binary(root);
    let listing = Command::new(&cargo)
        .args([
            "run",
            "-q",
            "-p",
            "vyre-bench",
            "--",
            "list",
            "--format",
            "json",
        ])
        .current_dir(root)
        .output()
        .map_err(|error| {
            GateError::new(
                format!(
                    "cannot run `{} run -p vyre-bench -- list`: {error}",
                    cargo.display()
                ),
                "restore the cargo_full wrapper at the workspace root",
            )
        })?;
    if !listing.status.success() {
        return Err(GateError::new(
            format!(
                "`cargo run -p vyre-bench -- list --format json` exited {}; a registry that cannot be listed is unmeasured, not covered: {}",
                listing.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&listing.stderr).trim()
            ),
            "run the same command by hand and fix what it reports",
        ));
    }
    let listed: serde_json::Value = serde_json::from_slice(&listing.stdout).map_err(|error| {
        GateError::new(
            format!("the vyre-bench registry listing is not JSON: {error}"),
            "make `vyre-bench list --format json` emit a JSON array of cases",
        )
    })?;
    let cases = listed.as_array().ok_or_else(|| {
        GateError::new(
            "the vyre-bench registry listing is not a JSON array",
            "make `vyre-bench list --format json` emit a JSON array of cases",
        )
    })?;
    let registered: BTreeSet<String> = cases
        .iter()
        .filter_map(|case| case.get("id").and_then(serde_json::Value::as_str))
        .map(str::to_string)
        .collect();
    if registered.is_empty() {
        return Err(GateError::new(
            "the vyre-bench registry lists no case with an `id`; every coverage rule below would pass vacuously",
            "register the benchmark cases, or repair the `id` field of the listing",
        ));
    }
    Ok(registered)
}

/// Judge one tree against one registry listing.
fn judge_coverage(tree: &Tree, registered: &BTreeSet<String>) -> Result<Report, GateError> {
    let mut report = Report::default();
    report.cover_complete("registered benchmark cases", registered.len());
    if let Some(note) = tree.absence_note() {
        report.note(note);
    }
    for (dimension, case) in REPRESENTATIVE {
        if !registered.contains(*case) {
            report.find(Finding::new(
                format!("the `{dimension}` dimension names vyre-bench case `{case}`, which is not registered"),
                "restore the case, or point the dimension at the case that replaced it; a dimension with no registered case is not measured",
            ));
        }
    }
    if tree.has(CACHE_CONTRACT) {
        if !tree.read(CACHE_CONTRACT)?.contains(CACHE_TEST) {
            report.find(Finding::in_file(
                CACHE_CONTRACT,
                format!("the compile_cache dimension names `{CACHE_TEST}`, which this file no longer defines"),
                "restore the test, or name the test that pins module-cache reuse now",
            ));
        }
    } else {
        report.find(Finding::in_file(
            CACHE_CONTRACT,
            "the compile_cache dimension names an executable cache contract that is not published",
            "restore the contract, or promote compile_cache to a vyre-bench case",
        ));
    }
    let mut references = 0_usize;
    let mut scanned = 0_usize;
    for path in tree.paths() {
        let extension = path.extension().and_then(|value| value.to_str());
        if !extension.is_some_and(|value| REFERENCE_EXTENSIONS.contains(&value)) {
            continue;
        }
        let Ok(text) = tree.read(path) else {
            continue;
        };
        if !text.contains("--case") {
            continue;
        }
        scanned += 1;
        for (number, line) in numbered(&text) {
            for case in cited_cases(line) {
                references += 1;
                if !registered.contains(&case) {
                    report.find(Finding::at(
                        path.clone(),
                        number,
                        format!("names `--case {case}`, which the vyre-bench registry does not contain"),
                        "use a registered case id; this invocation would fail wherever it runs, which for gpu-parity.yml and the release evidence manifests is far from PR CI",
                    ));
                }
            }
        }
    }
    report.note(format!(
        "{} dimensions covered by the {}-case registry, {references} `--case` reference(s) across {scanned} tracked file(s)",
        REPRESENTATIVE.len() + 1,
        registered.len()
    ));
    Ok(report)
}

/// Every case id a line names with `--case`.
///
/// Both spellings count: `--case id` and `--case=id`. A separator run is
/// consumed before the id so a wrapped invocation with several spaces reads the
/// same as a tight one.
fn cited_cases(line: &str) -> Vec<String> {
    let mut cases = Vec::new();
    let mut from = 0;
    while let Some(at) = line[from..].find("--case") {
        let after = from + at + "--case".len();
        let rest = &line[after..];
        let value = rest.trim_start_matches([' ', '\t', '=']);
        let consumed = rest.len() - value.len();
        if consumed == 0 {
            // `--cases` or `--case-list`: a different flag, not this one.
            from = after;
            continue;
        }
        let end = value
            .find(|character: char| !is_case_byte(character))
            .unwrap_or(value.len());
        if end > 0 {
            cases.push(value[..end].to_string());
        }
        from = after + consumed + end;
    }
    cases
}

/// Whether a character may appear in a benchmark case id.
fn is_case_byte(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-')
}

/// The dimensions this gate holds the registry to, for a caller that must cover
/// every one of them without restating the list.
#[must_use]
pub fn measured_dimensions() -> BTreeMap<&'static str, &'static str> {
    REPRESENTATIVE.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::GateBehavior;
    use crate::gates::fixture_checkout;

    /// WHY: the rule is "a crate `cargo bench` can run a target for", and the
    /// script it replaces asked "a directory called benches exists", which are
    /// different sets. A crate whose `benches/` holds only prose owes no
    /// measured number, and demanding one can only be satisfied by inventing
    /// it. Both target shapes cargo recognises must count, or a crate publishes
    /// a benchmark with no baseline.
    #[test]
    fn a_bench_target_and_not_a_bench_directory_owes_a_section() {
        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = [\"declared\", \"discovered\", \"prose\", \"opted-out\"]\n",
            ),
            (
                "declared/Cargo.toml",
                "[package]\nname = \"declared\"\nversion = \"0.0.0\"\n\n[[bench]]\nname = \"wire\"\nharness = false\n",
            ),
            ("declared/src/lib.rs", ""),
            (
                "discovered/Cargo.toml",
                "[package]\nname = \"discovered\"\nversion = \"0.0.0\"\n",
            ),
            ("discovered/src/lib.rs", ""),
            ("discovered/benches/throughput.rs", ""),
            (
                "prose/Cargo.toml",
                "[package]\nname = \"prose\"\nversion = \"0.0.0\"\n",
            ),
            ("prose/src/lib.rs", ""),
            ("prose/benches/README.md", "How to benchmark.\n"),
            (
                "opted-out/Cargo.toml",
                "[package]\nname = \"opted-out\"\nversion = \"0.0.0\"\nautobenches = false\n",
            ),
            ("opted-out/src/lib.rs", ""),
            ("opted-out/benches/legacy.rs", ""),
        ]);
        let tree = Tree::open(&root).unwrap();
        assert_eq!(
            bench_bearing_packages(&tree).unwrap(),
            vec!["declared".to_string(), "discovered".to_string()]
        );
    }

    /// WHY: a nested bench source is a target only in the `main.rs` shape, and
    /// the package name comes from the manifest. The script took the directory
    /// name two levels up, which for `crate/benches/group/case.rs` yielded
    /// `benches` and demanded a section for a crate that does not exist.
    #[test]
    fn a_nested_bench_target_is_named_by_its_manifest() {
        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = [\"nest/inner\"]\n",
            ),
            (
                "nest/inner/Cargo.toml",
                "[package]\nname = \"vyre-inner\"\nversion = \"0.0.0\"\n",
            ),
            ("nest/inner/src/lib.rs", ""),
            ("nest/inner/benches/group/main.rs", ""),
        ]);
        let tree = Tree::open(&root).unwrap();
        assert_eq!(
            bench_bearing_packages(&tree).unwrap(),
            vec!["vyre-inner".to_string()]
        );

        let (_other, shallow) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = [\"nest/inner\"]\n",
            ),
            (
                "nest/inner/Cargo.toml",
                "[package]\nname = \"vyre-inner\"\nversion = \"0.0.0\"\n",
            ),
            ("nest/inner/src/lib.rs", ""),
            ("nest/inner/benches/group/case.rs", ""),
        ]);
        let tree = Tree::open(&shallow).unwrap();
        assert!(
            bench_bearing_packages(&tree).unwrap().is_empty(),
            "a nested source that is not main.rs is not a bench target"
        );
    }

    /// WHY: the section rule is per crate, and a missing baseline header field
    /// is what makes a published median unreproducible. Both must be reported
    /// in one run: the script exited on the first failure, so a tree missing
    /// four things reported one and three more runs were needed to see them.
    #[test]
    fn every_missing_field_and_every_missing_section_is_reported_at_once() {
        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = [\"a\", \"b\"]\n",
            ),
            (
                "a/Cargo.toml",
                "[package]\nname = \"a\"\nversion = \"0.0.0\"\n",
            ),
            ("a/src/lib.rs", ""),
            ("a/benches/one.rs", ""),
            (
                "b/Cargo.toml",
                "[package]\nname = \"b\"\nversion = \"0.0.0\"\n",
            ),
            ("b/src/lib.rs", ""),
            ("b/benches/two.rs", ""),
            (
                "benches/RESULTS.md",
                "# Criterion baselines\n\nmachine: host\ncpu: cpu\ncommit: abc\n\n### a\n\n1ms\n| optimizer/pipeline/corpus/fixture | 1 us |\n",
            ),
            (
                OPTIMIZATION_FAMILY_MANIFEST,
                "{\"required_families\":[\"fixture\"]}\n",
            ),
        ]);
        let report = BenchBaselines
            .run(&GateCtx::new(root.clone(), Vec::new()))
            .unwrap();
        let rendered = report
            .findings
            .iter()
            .map(|finding| finding.message.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(report.count(), 3, "{rendered}");
        assert!(rendered.contains("`gpu:` field"), "{rendered}");
        assert!(rendered.contains("`rustc:` field"), "{rendered}");
        assert!(rendered.contains("### b"), "{rendered}");
    }

    /// WHY: one aggregate corpus id hid which semantic family regressed and
    /// made adding a family look like a slowdown. The roster comes from the
    /// generated manifest, so a new family must fail until it has its own
    /// measured row.
    #[test]
    fn every_registered_optimizer_family_owes_an_independent_baseline() {
        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = []\n",
            ),
            (
                "benches/RESULTS.md",
                "# Criterion baselines\n\nmachine: host\ngpu: gpu\ncpu: cpu\nrustc: 1.86\ncommit: abc\n\n| bench | median |\n| --- | --- |\n| optimizer/pipeline/corpus/scalar-algebra | 1 us |\n",
            ),
            (
                OPTIMIZATION_FAMILY_MANIFEST,
                "{\"required_families\":[\"scalar-algebra\",\"loop-transform\"]}\n",
            ),
        ]);
        let report = BenchBaselines.run(&GateCtx::new(root, Vec::new())).unwrap();
        assert_eq!(report.count(), 1);
        assert!(report.findings[0]
            .message
            .contains("`loop-transform` has no independent Criterion baseline"));
    }

    /// WHY: an empty generated roster must not make coverage vacuously green.
    /// The optimization producer owns the member set, but this consumer still
    /// rejects a manifest that gives it nothing to check.
    #[test]
    fn an_empty_optimizer_family_manifest_is_rejected() {
        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = []\n",
            ),
            (
                "benches/RESULTS.md",
                "# Criterion baselines\n\nmachine: host\ngpu: gpu\ncpu: cpu\nrustc: 1.86\ncommit: abc\n",
            ),
            (
                OPTIMIZATION_FAMILY_MANIFEST,
                "{\"required_families\":[]}\n",
            ),
        ]);
        let error = BenchBaselines
            .run(&GateCtx::new(root, Vec::new()))
            .expect_err("an empty optimizer family roster must fail closed");
        assert!(error
            .message
            .contains("declares no required optimizer families"));
    }

    /// WHY: a baseline nobody published is the whole gap this gate names, and it
    /// must not read as a clean tree.
    #[test]
    fn an_unpublished_baseline_is_one_finding() {
        let (_temporary, root) = fixture_checkout::checkout(&[(
            "Cargo.toml",
            "[workspace]\nresolver = \"2\"\nmembers = []\n",
        )]);
        let report = BenchBaselines.run(&GateCtx::new(root, Vec::new())).unwrap();
        assert_eq!(report.count(), 1);
        assert!(report.findings[0].message.contains("not published"));
    }

    /// WHY: the enforced budget must be the published one. A gate holding its
    /// own copy lets `contracts/perf_targets.toml` say 4000 while CI enforces
    /// something else, and the manifest is what a contributor reads.
    #[test]
    fn the_smoke_budget_comes_from_the_manifest() {
        let manifest: toml::Table = toml::from_str(
            "[crates.vyre-bench.targets.smoke_runtime]\nmetric = \"time_ms\"\nbudget = 4000\n",
        )
        .unwrap();
        assert_eq!(smoke_budget_ms(&manifest).unwrap(), 4000);

        let missing: toml::Table =
            toml::from_str("[crates.vyre-bench.targets.other]\nbudget = 1\n").unwrap();
        let error = smoke_budget_ms(&missing).unwrap_err();
        assert!(
            error
                .message
                .contains("crates.vyre-bench.targets.smoke_runtime"),
            "{}",
            error.message
        );

        let negative: toml::Table =
            toml::from_str("[crates.vyre-bench.targets.smoke_runtime]\nbudget = -1\n").unwrap();
        assert!(smoke_budget_ms(&negative).is_err());
    }

    /// WHY: the smoke ceiling applies to the 30 measured samples, not process
    /// startup or one-time backend preparation. Those setup costs varied from
    /// 4.2 to 14.5 seconds while the same 30 dispatch/readback samples remained
    /// below the four-second contract.
    #[test]
    fn smoke_budget_uses_all_harness_measured_samples() {
        let report = |mean, samples| {
            format!(
                r#"{{"cases":[{{"id":"{SMOKE_CASE}","metrics":{{"wall_ns":{{"mean":{mean},"samples":{samples}}}}}}}]}}"#
            )
        };
        assert_eq!(
            smoke_measured_ms(report(133_300_000, 30).as_bytes()).unwrap(),
            3999
        );
        assert_eq!(
            smoke_measured_ms(report(133_333_334, 30).as_bytes()).unwrap(),
            4001
        );
        assert!(smoke_measured_ms(report(1, 29).as_bytes()).is_err());
        assert!(smoke_measured_ms(br#"{"cases":[]}"#).is_err());
        assert!(smoke_measured_ms(b"not json").is_err());
    }

    /// WHY: the scan exists to catch a renamed case in a file only the GPU
    /// runner or a release run executes, so the extraction must read the
    /// spellings those files use and must not read a longer flag as this one.
    #[test]
    fn a_cited_case_is_read_in_every_spelling_a_command_uses() {
        assert_eq!(
            cited_cases("bench run --case foo.bar.1m"),
            vec!["foo.bar.1m"]
        );
        assert_eq!(cited_cases("bench run --case=foo.bar"), vec!["foo.bar"]);
        assert_eq!(
            cited_cases("--case a.b   --case   c.d"),
            vec!["a.b".to_string(), "c.d".to_string()]
        );
        assert_eq!(cited_cases("--case foo\" --case bar'"), vec!["foo", "bar"]);
        assert!(
            cited_cases("--cases foo").is_empty(),
            "a longer flag is not this flag"
        );
        assert!(cited_cases("--case").is_empty());
    }

    /// WHY: every dimension in the table must be judged, and the table is the
    /// only place the set is written. Enumerating it here rather than restating
    /// it means adding a dimension without a registered case turns this red.
    #[test]
    fn every_measured_dimension_is_judged_against_the_registry() {
        let cache = format!("fn {CACHE_TEST}() {{}}\n");
        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = []\n",
            ),
            (CACHE_CONTRACT, cache.as_str()),
        ]);
        let tree = Tree::open(&root).unwrap();

        let covered: BTreeSet<String> = measured_dimensions()
            .values()
            .map(|case| (*case).to_string())
            .collect();
        let report = judge_coverage(&tree, &covered).unwrap();
        assert_eq!(report.count(), 0, "{:?}", report.findings);

        for (dimension, case) in measured_dimensions() {
            let mut short = covered.clone();
            short.remove(case);
            let report = judge_coverage(&tree, &short).unwrap();
            let rendered = report
                .findings
                .iter()
                .map(|finding| finding.message.clone())
                .collect::<Vec<_>>()
                .join("\n");
            assert_eq!(report.count(), 1, "{dimension}: {rendered}");
            assert!(rendered.contains(dimension), "{dimension}: {rendered}");
        }
    }

    /// WHY: the compile_cache dimension is the one whose evidence is a test
    /// rather than a case, so deleting the file and deleting the test are two
    /// distinct ways to stop measuring it and both must fail.
    #[test]
    fn the_cache_dimension_fails_on_a_missing_file_and_on_a_missing_test() {
        let covered: BTreeSet<String> = measured_dimensions()
            .values()
            .map(|case| (*case).to_string())
            .collect();

        let (_absent, absent_root) = fixture_checkout::checkout(&[(
            "Cargo.toml",
            "[workspace]\nresolver = \"2\"\nmembers = []\n",
        )]);
        let report = judge_coverage(&Tree::open(&absent_root).unwrap(), &covered).unwrap();
        assert_eq!(report.count(), 1);
        assert!(report.findings[0].message.contains("not published"));

        let (_renamed, renamed_root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = []\n",
            ),
            (CACHE_CONTRACT, "fn something_else() {}\n"),
        ]);
        let report = judge_coverage(&Tree::open(&renamed_root).unwrap(), &covered).unwrap();
        assert_eq!(report.count(), 1);
        assert!(report.findings[0].message.contains(CACHE_TEST));
    }

    /// WHY: a `--case` in a workflow or an evidence manifest is an invocation
    /// that fails where nobody watches; a `--case` in Rust source is a help
    /// string. Scanning the second class reported the harness's own help text.
    #[test]
    fn a_cited_case_is_scanned_in_a_command_file_and_not_in_rust_source() {
        let covered: BTreeSet<String> = measured_dimensions()
            .values()
            .map(|case| (*case).to_string())
            .collect();
        let cache = format!("fn {CACHE_TEST}() {{}}\n");
        let (_temporary, root) = fixture_checkout::checkout(&[
            (
                "Cargo.toml",
                "[workspace]\nresolver = \"2\"\nmembers = []\n",
            ),
            (CACHE_CONTRACT, cache.as_str()),
            (
                ".github/workflows/gpu-parity.yml",
                "jobs:\n  run:\n    steps:\n      - run: bench run --case gone.case.1m\n",
            ),
            (
                "vyre-bench/src/help.rs",
                "const HELP: &str = \"pass --case also.gone to select one case\";\n",
            ),
        ]);
        let report = judge_coverage(&Tree::open(&root).unwrap(), &covered).unwrap();
        let rendered = report
            .findings
            .iter()
            .map(|finding| format!("{:?} {}", finding.file, finding.message))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(report.count(), 1, "{rendered}");
        assert!(rendered.contains("gone.case.1m"), "{rendered}");
        assert!(!rendered.contains("also.gone"), "{rendered}");
    }
}
