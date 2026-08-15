//! `vyre-lints` CLI. Runs the lint suite against a workspace.
//!
//! Usage:
//!   vyre-lints --workspace-root . [--allowlist vyre-lints/allowlist.toml] [--format json|text]
//!
//! Exits 0 if no violations (after allowlist filter). Exits 1 if any
//! violation. Exit 2 on I/O / parse failure.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::{Path, PathBuf};
use vyre_lints::{
    read_source_bounded, run_consumer_coupling, run_gpu_skip_guards, run_module_forks,
    run_production_cpu_fallbacks, run_raw_ir_in_libs, Violation, ViolationKind,
};

#[derive(Parser, Debug)]
#[command(
    name = "vyre-lints",
    version,
    about = "Lego-block enforcement lints for vyre"
)]
struct Cli {
    /// Workspace root (the dir containing vyre-libs/, vyre-foundation/, ...).
    #[arg(long, default_value = ".")]
    workspace_root: PathBuf,

    /// Allowlist file. If omitted, defaults to <workspace_root>/vyre-lints/allowlist.toml.
    #[arg(long)]
    allowlist: Option<PathBuf>,

    /// Output format: text (default) or json.
    #[arg(long, default_value = "text")]
    format: Format,

    /// Override the lib roots scanned. Defaults to vyre-libs/src.
    #[arg(long)]
    lib_root: Option<PathBuf>,

    /// Run the allowlist drift sentinel: fail if any allowlist entry
    /// is older than `--drift-budget-days` (default 14). Skips the
    /// raw-IR scan when set.
    #[arg(long)]
    check_drift: bool,

    /// Age budget for the drift sentinel, in days.
    #[arg(long, default_value_t = vyre_lints::drift::DEFAULT_AGE_BUDGET_DAYS)]
    drift_budget_days: i64,

    /// Today's date in YYYY-MM-DD form. Defaults to the OS clock.
    #[arg(long)]
    today: Option<String>,

    /// Run the production CPU fallback guard instead of the raw-IR lint.
    #[arg(long)]
    check_production_cpu_fallbacks: bool,

    /// Override production roots scanned by `--check-production-cpu-fallbacks`.
    /// Defaults to Vyre-owned production crates, excluding reference/conform crates.
    /// External consumers can be scanned by passing this flag repeatedly.
    #[arg(long)]
    production_root: Vec<PathBuf>,

    /// Run the consumer-name coupling guard over platform docs/comments.
    #[arg(long)]
    check_consumer_coupling: bool,

    /// Override roots scanned by `--check-consumer-coupling`.
    /// Defaults to current docs plus platform source crates.
    #[arg(long)]
    consumer_root: Vec<PathBuf>,

    /// Run the same-name module fork scanner over selected authority roots.
    #[arg(long)]
    check_module_forks: bool,

    /// Override roots scanned by `--check-module-forks`.
    /// Defaults to graph authority roots where fork drift has historically appeared.
    #[arg(long)]
    module_fork_root: Vec<PathBuf>,

    /// Run the GPU skip guard over CUDA/WGPU/runtime validation paths.
    #[arg(long)]
    check_gpu_skip_guards: bool,

    /// Override roots scanned by `--check-gpu-skip-guards`.
    #[arg(long)]
    gpu_skip_root: Vec<PathBuf>,

    /// Print the selected lint's default roots, one per line, and exit
    /// without scanning. Lets a caller check the declared scan scope without
    /// restating it.
    #[arg(long)]
    print_default_roots: bool,
}

#[derive(Clone, Debug, clap::ValueEnum)]
enum Format {
    Text,
    Json,
}

/// One lint the CLI can select, with everything that differs between lints.
///
/// Four `run_*_cli` functions used to hold this data inline, each repeating
/// the same five steps: pick overridden or default roots, fail closed on a
/// missing root, scan, emit, exit. Only the four fields below ever differed,
/// so they are the data and the driver below is the one copy of the steps.
struct Lint {
    /// `--check-...` flag that selects this lint, for diagnostics.
    flag: &'static str,
    /// Kinds this lint reports. The gate below requires every
    /// [`ViolationKind`] to be claimed by exactly one entry.
    kinds: &'static [ViolationKind],
    /// Roots scanned when the caller overrides none, resolved against the
    /// workspace root.
    ///
    /// A function rather than a list because a list of crate names goes stale
    /// the moment a crate is added, and it goes stale silently: the new crate
    /// is simply not scanned and the lint still reports success. That is not
    /// hypothetical here. `vyre-foundation` and `vyre-primitives` were never in
    /// the CPU-fallback list, which is how a `PolicyRoute::CpuSimd` variant, two
    /// threshold fields feeding it and a `-> bool { false }` selector sat in
    /// `vyre-foundation/src/execution_plan/policy.rs` with this guard green.
    default_roots: fn(&Path) -> Result<Vec<PathBuf>>,
    /// What a missing root means, appended to the fail-closed diagnostic.
    missing_root_fix: &'static str,
    /// How the fail-closed diagnostic names a missing root.
    root_noun: &'static str,
    /// Context added to a scan failure.
    context: &'static str,
    /// Scan entry point.
    scan: fn(&[&Path]) -> Result<Vec<Violation>>,
}

/// Flag-selected lints, in the order `main` tests them.
///
/// `raw_ir_in_libs` is absent on purpose: it is the default action rather than
/// a flag, and its roots come from the allowlist file rather than from a
/// declared list. The gate below records that, so its kinds still have to be
/// accounted for.
const LINTS: &[Lint] = &[
    Lint {
        flag: "check-production-cpu-fallbacks",
        kinds: &[ViolationKind::ProductionCpuFallback],
        default_roots: production_source_roots,
        missing_root_fix:
            "Fix: update the release CPU fallback guard roots instead of silently skipping this source tree.",
        root_noun: "production root",
        context: "running production CPU fallback guard",
        scan: run_production_cpu_fallbacks,
    },
    Lint {
        flag: "check-consumer-coupling",
        kinds: &[ViolationKind::ConsumerCoupling],
        default_roots: |workspace| {
            fixed_roots(
                workspace,
                &[
                    "docs",
                    "vyre/src",
                    "vyre-driver/src",
                    "vyre-driver-cuda/src",
                    "vyre-driver-wgpu/src",
                    "vyre-foundation/src",
                    "vyre-libs/src",
                    "vyre-lower/src",
                    "vyre-primitives/src",
                    "vyre-runtime/src",
                    "vyre-pass-engine/src",
                ],
            )
        },
        missing_root_fix:
            "Fix: update the platform doc/comment guard roots instead of silently shrinking scan coverage.",
        root_noun: "consumer coupling root",
        context: "running consumer-name coupling guard",
        scan: run_consumer_coupling,
    },
    Lint {
        flag: "check-module-forks",
        kinds: &[ViolationKind::ModuleFork],
        default_roots: |workspace| {
            fixed_roots(workspace, &["vyre-primitives/src/graph", "vyre-libs/src/graph"])
        },
        missing_root_fix:
            "Fix: update the duplicate-module scan roots instead of silently shrinking scan coverage.",
        root_noun: "module fork root",
        context: "running same-name module fork scanner",
        scan: run_module_forks,
    },
    Lint {
        flag: "check-gpu-skip-guards",
        kinds: &[ViolationKind::GpuSkipGuard],
        default_roots: |workspace| {
            fixed_roots(
                workspace,
                &[
                    "vyre-driver-cuda/src",
                    "vyre-driver-cuda/tests",
                    "vyre-driver-wgpu/src",
                    "vyre-driver-wgpu/tests",
                    "vyre-runtime/src",
                ],
            )
        },
        missing_root_fix:
            "Fix: update CUDA/WGPU validation roots instead of silently shrinking scan coverage.",
        root_noun: "GPU skip guard root",
        context: "running GPU skip guard",
        scan: run_gpu_skip_guards,
    },
];

/// Workspace-relative `roots` joined onto `workspace`.
fn fixed_roots(workspace: &Path, roots: &[&str]) -> Result<Vec<PathBuf>> {
    Ok(roots.iter().map(|root| workspace.join(root)).collect())
}

/// Crates exempt from the production CPU/reference guard, with the reason.
///
/// `vyre-reference` computes on the host by definition: it is the parity oracle
/// a conformance comparison reads. `vyre-test-support` and `vyre-lints` are
/// harness and tooling, not a dispatch path. `vyre-spec` and `vyre-macros`
/// declare rather than execute. Every other member is scanned, including one
/// added after this line was written.
const CPU_GUARD_EXEMPT_CRATES: &[(&str, &str)] = &[
    ("vyre-reference", "the host parity oracle, which is the one legitimate host execution"),
    ("vyre-test-support", "test harness, publish = false, never on a dispatch path"),
    ("vyre-lints", "this tool; its own fixtures name the patterns it forbids"),
    ("vyre-spec", "declares the frozen surface, executes nothing"),
    ("vyre-macros", "proc macros run at compile time, not on a dispatch path"),
];

/// The nearest ancestor of `start`, itself included, whose manifest declares a
/// `[workspace]` table.
///
/// The CLI defaults its root to the working directory, so a run from inside a
/// member directory read that member's manifest, found no members, and reported
/// the workspace as undeclared. Walking up answers the question the caller
/// meant, and a tree with no workspace manifest above it is an error naming
/// where the walk started rather than an empty scan.
fn workspace_manifest_root(start: &Path) -> Result<PathBuf> {
    let start = if start.is_absolute() {
        start.to_path_buf()
    } else {
        std::env::current_dir()
            .context("resolving the working directory for the workspace root walk")?
            .join(start)
    };
    for candidate in start.ancestors() {
        let manifest = candidate.join("Cargo.toml");
        if !manifest.is_file() {
            continue;
        }
        let text = read_source_bounded(&manifest)?;
        let parsed: toml::Table =
            toml::from_str(&text).with_context(|| format!("parsing {}", manifest.display()))?;
        if parsed.contains_key("workspace") {
            return Ok(candidate.to_path_buf());
        }
    }
    anyhow::bail!(
        "Fix: no Cargo.toml declaring `[workspace]` at or above {}. Pass --workspace-root.",
        start.display()
    )
}

/// Every workspace member's `src` directory, minus the recorded exemptions.
///
/// Read from the workspace manifest at run time. A crate added to the workspace
/// is scanned from its first commit without anyone editing this file, which is
/// the property a hardcoded list cannot have: the list that stood here omitted
/// `vyre-foundation` and `vyre-primitives`, so the guard reported success while
/// never reading either of them.
fn production_source_roots(workspace: &Path) -> Result<Vec<PathBuf>> {
    let workspace = workspace_manifest_root(workspace)?;
    let manifest = read_source_bounded(&workspace.join("Cargo.toml"))?;
    // `toml::Table` rather than `toml::Value`: with the pinned toml, a whole
    // document does not deserialize into `Value`, which is why the rest of the
    // workspace parses a table and reads fields off it.
    let manifest: toml::Table = toml::from_str(&manifest).context("parsing the workspace manifest")?;
    let members = manifest
        .get("workspace")
        .and_then(toml::Value::as_table)
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
        .context(
            "Fix: the workspace manifest declares no `workspace.members`, so the CPU fallback \
             guard cannot derive what to scan and will not fall back to a stale list",
        )?;

    let exempt: Vec<&str> = CPU_GUARD_EXEMPT_CRATES.iter().map(|(name, _)| *name).collect();
    let mut roots = Vec::with_capacity(members.len());
    for member in members {
        let Some(member) = member.as_str() else {
            anyhow::bail!("Fix: a `workspace.members` entry is not a string");
        };
        if exempt.contains(&member) {
            continue;
        }
        let source = workspace.join(member).join("src");
        if source.is_dir() {
            roots.push(source);
        }
    }
    anyhow::ensure!(
        !roots.is_empty(),
        "Fix: the CPU fallback guard derived no scan roots from the workspace manifest"
    );
    Ok(roots)
}

/// Kinds reported by the default `raw_ir_in_libs` run rather than by a
/// flag-selected lint.
const DEFAULT_RUN_KINDS: &[ViolationKind] = &[
    ViolationKind::RawNodeConstruction,
    ViolationKind::RawExprConstruction,
];

/// Run one lint: resolve its roots, fail closed on a missing one, emit, exit.
fn run_lint(cli: &Cli, lint: &Lint, overrides: &[PathBuf]) -> Result<()> {
    if cli.print_default_roots {
        // Printed relative to the workspace root: the roots are workspace
        // members, so a caller reading this list is reading member paths, not
        // paths into this checkout.
        let workspace = workspace_manifest_root(&cli.workspace_root)?;
        for root in (lint.default_roots)(&cli.workspace_root)? {
            let shown = root.strip_prefix(&workspace).unwrap_or(&root);
            println!("{}", shown.display());
        }
        return Ok(());
    }
    let roots: Vec<PathBuf> = if overrides.is_empty() {
        (lint.default_roots)(&cli.workspace_root)?
    } else {
        overrides.to_vec()
    };
    for root in &roots {
        if !root.exists() {
            anyhow::bail!(
                "{} not found: {}. {}",
                lint.root_noun,
                root.display(),
                lint.missing_root_fix
            );
        }
    }
    let root_refs: Vec<&Path> = roots.iter().map(PathBuf::as_path).collect();
    let violations = (lint.scan)(&root_refs).context(lint.context)?;
    report(cli, &violations)
}

/// Emit violations in the requested format and set the process exit status.
fn report(cli: &Cli, violations: &[Violation]) -> Result<()> {
    match cli.format {
        Format::Text => emit_text(violations),
        Format::Json => emit_json(violations)?,
    }
    if violations.is_empty() {
        Ok(())
    } else {
        std::process::exit(1);
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let allowlist = cli
        .allowlist
        .clone()
        .unwrap_or_else(|| cli.workspace_root.join("vyre-lints/allowlist.toml"));

    if cli.check_drift {
        return run_drift(&allowlist, cli.drift_budget_days, cli.today.as_deref());
    }

    // Selected flags, paired with the override list that belongs to each. The
    // pairing is here rather than in `LINTS` because the overrides live on
    // `Cli`, which clap owns.
    let selected: [(bool, &[PathBuf]); LINTS.len()] = [
        (
            cli.check_production_cpu_fallbacks,
            cli.production_root.as_slice(),
        ),
        (cli.check_consumer_coupling, cli.consumer_root.as_slice()),
        (cli.check_module_forks, cli.module_fork_root.as_slice()),
        (cli.check_gpu_skip_guards, cli.gpu_skip_root.as_slice()),
    ];
    for (lint, (requested, overrides)) in LINTS.iter().zip(selected) {
        if requested {
            return run_lint(&cli, lint, overrides);
        }
    }

    let allowlist_arg = if allowlist.exists() {
        Some(allowlist.as_path())
    } else {
        None
    };

    // A `--lib-root` override names one tree explicitly. Otherwise the trees
    // come from the lint's own configuration, so relocating a composition
    // domain between crates does not need a code change here.
    let roots: Vec<PathBuf> = match cli.lib_root.clone() {
        Some(lib_root) => vec![lib_root],
        None => {
            let configured = match allowlist_arg {
                Some(path) => vyre_lints::allowlist::load(path)?,
                None => vyre_lints::allowlist::Allowlist::empty(),
            };
            configured
                .measured_roots()
                .iter()
                .map(|measured_root| cli.workspace_root.join(measured_root))
                .collect()
        }
    };
    for root in &roots {
        if !root.exists() {
            anyhow::bail!("measured root not found: {}", root.display());
        }
    }

    let root_refs: Vec<&Path> = roots.iter().map(PathBuf::as_path).collect();
    let violations =
        run_raw_ir_in_libs(&root_refs, allowlist_arg).context("running raw_ir_in_libs lint")?;

    report(&cli, &violations)
}

fn run_drift(
    allowlist: &std::path::Path,
    budget_days: i64,
    today_override: Option<&str>,
) -> Result<()> {
    if !allowlist.exists() {
        anyhow::bail!("allowlist not found: {}", allowlist.display());
    }
    let today = match today_override {
        Some(s) => s.to_string(),
        None => current_iso_date(),
    };
    let resolver = vyre_lints::drift::GitBlameResolver::with_today(today);
    let findings = vyre_lints::drift::run(allowlist, budget_days, &resolver)
        .context("running allowlist drift sentinel")?;
    if findings.is_empty() {
        println!("vyre-lints drift: 0 stale entries (budget {budget_days} days)");
        return Ok(());
    }
    println!(
        "vyre-lints drift: {} stale entry(ies)  -  every entry should land its migration ticket within {budget_days} days.",
        findings.len()
    );
    for f in &findings {
        println!("{}", vyre_lints::drift::format_finding(f, budget_days));
    }
    std::process::exit(1);
}

fn current_iso_date() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = now.div_euclid(86_400);
    vyre_lints::drift::iso_from_days(days)
}

fn emit_text(violations: &[Violation]) {
    if violations.is_empty() {
        println!("vyre-lints: 0 violations");
        return;
    }
    for v in violations {
        println!("{}:{}:{}: {}", v.file, v.line, v.column, v.message);
    }
    println!("vyre-lints: {} violation(s)", violations.len());
}

fn emit_json(violations: &[Violation]) -> Result<()> {
    use std::fmt::Write;
    let mut out = String::from("[\n");
    for (i, v) in violations.iter().enumerate() {
        let kind = v.kind.as_str();
        if i > 0 {
            out.push_str(",\n");
        }
        write!(
            out,
            "  {{\"file\":{:?},\"line\":{},\"column\":{},\"kind\":{:?},\"message\":{:?}}}",
            v.file, v.line, v.column, kind, v.message
        )?;
    }
    out.push_str("\n]\n");
    print!("{out}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        production_source_roots, read_source_bounded, Lint, CPU_GUARD_EXEMPT_CRATES,
        DEFAULT_RUN_KINDS, LINTS,
    };
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use vyre_lints::ViolationKind;

    /// Lint recorded as reporting `kind`.
    ///
    /// Exhaustive on purpose: a `ViolationKind` added to the library does not
    /// compile here until someone records which lint reports it, and the tests
    /// below then require that decision to match the registry.
    fn recorded_reporter(kind: &ViolationKind) -> &'static str {
        match kind {
            ViolationKind::RawNodeConstruction | ViolationKind::RawExprConstruction => {
                "raw-ir-in-libs (default run)"
            }
            ViolationKind::ProductionCpuFallback => "check-production-cpu-fallbacks",
            ViolationKind::ConsumerCoupling => "check-consumer-coupling",
            ViolationKind::ModuleFork => "check-module-forks",
            ViolationKind::GpuSkipGuard => "check-gpu-skip-guards",
        }
    }

    #[test]
    fn every_violation_kind_reaches_a_registered_lint_or_the_default_run() {
        let mut reporter: BTreeMap<usize, &str> = BTreeMap::new();
        for lint in LINTS {
            for kind in lint.kinds {
                let previous = reporter.insert(kind.position(), lint.flag);
                assert_eq!(previous, None, "{kind:?} is claimed by two lints");
            }
        }
        for kind in DEFAULT_RUN_KINDS {
            let previous = reporter.insert(kind.position(), "raw-ir-in-libs (default run)");
            assert_eq!(previous, None, "{kind:?} is claimed by two lints");
        }

        for kind in ViolationKind::ALL {
            let found = reporter.get(&kind.position()).copied();
            assert_eq!(
                found,
                Some(recorded_reporter(kind)),
                "{kind:?} has no lint the CLI can run. Fix: register a lint in LINTS, \
                 or record it in DEFAULT_RUN_KINDS, and record the same decision in \
                 recorded_reporter."
            );
        }
    }

    #[test]
    fn every_registered_lint_declares_a_distinct_flag_and_resolves_real_roots() {
        let workspace = vyre_test_support::monorepo::vyre_workspace_root();
        let flags: std::collections::BTreeSet<_> = LINTS.iter().map(|lint| lint.flag).collect();
        assert_eq!(flags.len(), LINTS.len());

        for Lint {
            flag,
            kinds,
            default_roots,
            missing_root_fix,
            root_noun,
            context,
            ..
        } in LINTS
        {
            assert!(!kinds.is_empty(), "{flag} reports no violation kind");
            let roots = default_roots(&workspace)
                .unwrap_or_else(|error| panic!("{flag} cannot resolve its roots: {error}"));
            assert!(!roots.is_empty(), "{flag} scans nothing by default");
            for root in &roots {
                assert!(
                    root.exists(),
                    "{flag} declares a root that does not exist in this checkout: {}. \
                     A root that is missing is scanned as nothing, and the lint then reports \
                     success over a tree it never opened.",
                    root.display()
                );
            }
            assert!(
                missing_root_fix.starts_with("Fix: "),
                "{flag} missing-root diagnostic gives no corrective action"
            );
            assert!(
                root_noun.ends_with("root"),
                "{flag} names a missing root as `{root_noun}`, which does not read as a root"
            );
            assert!(!context.is_empty(), "{flag} adds no context to a failure");
        }
    }

    /// Every workspace member is either scanned for host execution or exempt
    /// with a reason.
    ///
    /// The class this closes: a crate that nobody remembered to add to the
    /// guard's root list. The list that used to stand in `LINTS` named nine
    /// crates and omitted `vyre-foundation`, which is where a `CpuSimd` route,
    /// its two threshold fields and its `-> bool { false }` selector lived while
    /// this guard reported success. The roster is now read from the manifest, so
    /// this test only has to prove the two sets partition the workspace, and a
    /// member added tomorrow lands on the scanned side by default.
    #[test]
    fn every_workspace_member_is_scanned_for_host_execution_or_exempt_with_a_reason() {
        let workspace = vyre_test_support::monorepo::vyre_workspace_root();
        let text = read_source_bounded(&workspace.join("Cargo.toml"))
            .expect("the workspace manifest is readable");
        let manifest: toml::Table =
            toml::from_str(&text).expect("the workspace manifest is valid TOML");
        let members: Vec<String> = manifest["workspace"]["members"]
            .as_array()
            .expect("the workspace declares members")
            .iter()
            .filter_map(|member| member.as_str().map(str::to_owned))
            .collect();
        assert!(members.len() > 20, "the workspace member list is truncated");

        let scanned: std::collections::BTreeSet<PathBuf> =
            production_source_roots(&workspace).expect("roots resolve").into_iter().collect();
        let exempt: std::collections::BTreeSet<&str> =
            CPU_GUARD_EXEMPT_CRATES.iter().map(|(name, _)| *name).collect();

        for member in &members {
            let source = workspace.join(member).join("src");
            if !source.is_dir() {
                continue;
            }
            let is_scanned = scanned.contains(&source);
            let is_exempt = exempt.contains(member.as_str());
            assert!(
                is_scanned != is_exempt,
                "Fix: workspace member `{member}` is {}. Every member either has its sources \
                 scanned for host execution or carries a row in CPU_GUARD_EXEMPT_CRATES saying \
                 why it does not.",
                if is_scanned {
                    "both scanned and exempt"
                } else {
                    "neither scanned nor exempt"
                }
            );
        }

        for (name, reason) in CPU_GUARD_EXEMPT_CRATES {
            assert!(
                members.iter().any(|member| member == name),
                "Fix: `{name}` is exempted from the host-execution guard but is not a workspace \
                 member, so the row exempts nothing and hides the next crate that takes its name"
            );
            assert!(
                reason.split_whitespace().count() >= 5,
                "Fix: the exemption for `{name}` has no reason worth reading: {reason:?}"
            );
        }
    }
}
