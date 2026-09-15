//! Source-derived platform support matrix gate.
//!
//! The cell space is read out of `vyre-foundation/src/platform/matrix.rs` by
//! parsing it: the `HostOs` and `HostArch` enums supply the members, their
//! `id`, `pointer_width` and `endianness` matches supply the facts, and the
//! `PlatformSupportMatrix::tier` match supplies what each cell claims. A cell
//! written by hand here would state a pairing the compiler never agreed to,
//! which is how this document once carried an x86_64 host with 32-bit
//! pointers.
//!
//! What a cell has been shown to do is separate from what it claims, and comes
//! only from `release/evidence/portability/host-identity.json`, which is
//! written by executing `cargo xtask portability-evidence --write`. A cell with
//! no run in that ledger is published as unproven. Device capability facts are
//! not stated here at all: subgroup width, shared memory size and tensor-core
//! presence are measurements, and a table of them in a gate that runs no probe
//! is a fabrication.
//!
//! The driver roster is read the same way. Each driver crate declares the
//! registry identifier it answers to as a `pub const` ending in
//! `_BACKEND_ID`, and that constant is the row. Payload formats, API families
//! and minimum API versions are not recorded, because no source states them
//! and the rows that once carried them were typed into this generated file by
//! hand.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use syn::{Expr, Fields, ImplItem, Item, Lit, Pat};

use crate::artifact_gate::{settle_inspection, Inspection};
use crate::gate::{Finding, GateBehavior, GateCtx, GateError, Report};
use crate::manifest_walk::{self, PackageManifest};

/// Canonical schema version for the platform support matrix document.
///
/// Version 2 replaced hand-written cells and evidence-class strings with cells
/// derived from the platform source and an evidence field that only an
/// executed run can raise above `none`.
pub const PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION: u32 = 2;

/// Artifact path owned by this gate.
pub const MANIFEST_PATH: &str = "docs/generated/platform-support-matrix.toml";

/// Source of the cell space, parsed rather than restated.
pub const MATRIX_SOURCE_PATH: &str = "vyre-foundation/src/platform/matrix.rs";

/// Record of what actually ran on each cell.
pub const LEDGER_PATH: &str = "release/evidence/portability/host-identity.json";

/// Schema version of the portability ledger this gate reads.
pub const PORTABILITY_LEDGER_SCHEMA_VERSION: u32 = 1;

/// How a cell was reached, strongest first.
///
/// `TypeChecked` is deliberately not evidence of support. A cross-compilation
/// proves the crate builds for a target and nothing about whether the wire
/// format round-trips there, which is the only interesting question on a
/// big-endian host.
pub const EVIDENCE_NATIVE: &str = "executed-native";
/// A run under a user-mode emulator. Proves host behavior and decoding, never
/// device support or performance.
pub const EVIDENCE_EMULATED: &str = "executed-emulated";
/// A cross-compilation only.
pub const EVIDENCE_TYPE_CHECKED: &str = "type-checked";
/// Nothing has been run or built for this cell.
pub const EVIDENCE_NONE: &str = "none";

/// Top-level platform support matrix document.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlatformSupportMatrixDocument {
    /// Schema version for fail-closed validation.
    pub schema_version: u32,
    /// Byte order every persisted payload is written in.
    pub canonical_endianness: String,
    /// Minimum Rust toolchain declared by the workspace manifest.
    pub canonical_rust_version: String,
    /// Every host cell the platform source declares.
    pub host_cells: Vec<HostCellEntry>,
    /// Every backend driver the workspace ships, as its own source declares it.
    pub drivers: Vec<DriverEntry>,
    /// Every declared feature of every workspace package.
    pub package_features: Vec<PackageFeatureEntry>,
}

/// One host execution cell.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HostCellEntry {
    /// Host operating system identifier.
    pub os: String,
    /// Host CPU architecture identifier.
    pub arch: String,
    /// Pointer width in bits, as the architecture defines it.
    pub pointer_width: String,
    /// Byte order, as the architecture defines it.
    pub endianness: String,
    /// What the platform source claims for this cell.
    pub tier: String,
    /// What has been shown, from the portability ledger.
    pub evidence: String,
    /// Target triple the recorded run used, empty when none ran.
    pub target_triple: String,
    /// Emulator the recorded run used, empty for a native run or none.
    pub emulator: String,
    /// Canonical corpus identity the run printed, empty when none ran.
    pub identity_digest: String,
    /// What would raise this cell's evidence, empty when the cell is excluded.
    pub requires: String,
}

/// One backend driver the workspace ships.
///
/// A driver crate registers under an identifier it declares in its own source
/// as a `pub const` ending in `_BACKEND_ID`. That constant is the only thing
/// that decides what the registry answers to, so it is what this document
/// records. The rows this replaced were written by hand into a generated file
/// and carried payload formats, API families and minimum API versions that no
/// source states, which made every one of them unfalsifiable.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DriverEntry {
    /// Registry identifier the crate declares for itself.
    pub id: String,
    /// Workspace package that declares it.
    pub package: String,
    /// Path of the source file the identifier is declared in.
    pub declared_in: String,
    /// Whether the package is published, and so claimed as a shipped driver.
    pub claimed: bool,
}

/// One package feature cell.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PackageFeatureEntry {
    /// Owning workspace package.
    pub package: String,
    /// Declared feature name.
    pub feature: String,
    /// Publication class the package manifest declares for itself.
    pub publication_class: String,
}

/// One recorded run, as `portability-evidence --write` writes it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PortabilityRun {
    /// Host operating system identifier the run covers.
    pub os: String,
    /// Host CPU architecture identifier the run covers.
    pub arch: String,
    /// Target triple the run used.
    pub target_triple: String,
    /// One of the evidence constants in this module.
    pub evidence: String,
    /// Emulator binary, empty for a native run.
    pub emulator: String,
    /// Identity of the canonical corpus, as the run printed it.
    pub canonical_identity: String,
    /// Identity of the structural-fallback corpus, as the run printed it.
    pub fallback_identity: String,
}

/// Record of what actually ran, per cell.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PortabilityLedger {
    /// Schema version for fail-closed validation.
    pub schema_version: u32,
    /// Every recorded run, in cell order.
    pub runs: Vec<PortabilityRun>,
}

/// Platform support matrix gate.
pub struct PlatformSupportMatrixGate;

impl GateBehavior for PlatformSupportMatrixGate {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let (matrix_doc, findings) = collect_support_matrix(&ctx.root)?;
        let rendered = render_matrix(&matrix_doc)?;

        let mut inspection = Inspection::new();
        for finding in findings {
            inspection.find(finding);
        }
        inspection.generates_document_text(MANIFEST_PATH, rendered);

        let proven = matrix_doc
            .host_cells
            .iter()
            .filter(|cell| is_executed(&cell.evidence))
            .count();
        let mut report = settle_inspection(ctx, ctx.gate_name()?, inspection);
        report.note(format!(
            "{} host cell(s), {proven} with an executed run, {} driver(s), {} package feature cell(s)",
            matrix_doc.host_cells.len(),
            matrix_doc.drivers.len(),
            matrix_doc.package_features.len()
        ));
        Ok(report)
    }
}

/// Whether an evidence value records a run rather than a build.
#[must_use]
pub fn is_executed(evidence: &str) -> bool {
    evidence == EVIDENCE_NATIVE || evidence == EVIDENCE_EMULATED
}

/// Render the canonical TOML representation of the platform support matrix.
///
/// # Errors
///
/// Returns `GateError` when the document cannot be serialized.
pub fn render_matrix(doc: &PlatformSupportMatrixDocument) -> Result<String, GateError> {
    let mut rendered = String::from(
        "# Generated by `cargo xtask platform-support-matrix --write`.\n\
         # Cells come from vyre-foundation/src/platform/matrix.rs; evidence comes from\n\
         # release/evidence/portability/host-identity.json, which only an executed run writes.\n",
    );
    let body = toml::to_string_pretty(doc).map_err(|error| {
        GateError::new(
            format!("failed to serialize platform support matrix: {error}"),
            "keep every document field serializable as TOML",
        )
    })?;
    rendered.push_str(&body);
    Ok(rendered)
}

/// The cell space and its facts, as parsed out of the platform source.
#[derive(Clone, Debug, Default)]
pub struct MatrixSource {
    /// `HostOs` variant ident to its stable identifier.
    pub os_ids: BTreeMap<String, String>,
    /// `HostArch` variant ident to its stable identifier.
    pub arch_ids: BTreeMap<String, String>,
    /// `HostArch` variant ident to its pointer width in bits.
    pub arch_pointer_width: BTreeMap<String, String>,
    /// `HostArch` variant ident to its byte order identifier.
    pub arch_endianness: BTreeMap<String, String>,
    /// `(HostOs, HostArch)` variant idents to the claimed tier identifier.
    pub tiers: BTreeMap<(String, String), String>,
    /// Declaration order of the `HostOs` variants.
    pub os_order: Vec<String>,
    /// Declaration order of the `HostArch` variants.
    pub arch_order: Vec<String>,
}

/// Parse the platform source into its declared cell space.
///
/// # Errors
///
/// Returns `GateError` when the source cannot be read or parsed as Rust.
pub fn parse_matrix_source(root: &Path) -> Result<(MatrixSource, Vec<Finding>), GateError> {
    let path = root.join(MATRIX_SOURCE_PATH);
    let text = std::fs::read_to_string(&path).map_err(|error| {
        GateError::new(
            format!("failed to read {MATRIX_SOURCE_PATH}: {error}"),
            "restore the platform matrix source that declares the host cell space",
        )
    })?;
    let file = syn::parse_file(&text).map_err(|error| {
        GateError::new(
            format!("failed to parse {MATRIX_SOURCE_PATH}: {error}"),
            "repair the platform matrix source so the cell space can be derived from it",
        )
    })?;

    let mut source = MatrixSource::default();
    let mut findings = Vec::new();

    source.os_order = enum_variants(&file, "HostOs");
    source.arch_order = enum_variants(&file, "HostArch");
    if source.os_order.is_empty() || source.arch_order.is_empty() {
        return Err(GateError::new(
            format!("{MATRIX_SOURCE_PATH} declares no HostOs or no HostArch variants"),
            "declare the host operating systems and architectures the workspace names",
        ));
    }

    source.os_ids = string_match_arms(&file, "HostOs", "id");
    source.arch_ids = string_match_arms(&file, "HostArch", "id");
    source.arch_pointer_width = path_match_arms(&file, "HostArch", "pointer_width")
        .into_iter()
        .map(|(variant, width)| (variant, width.trim_start_matches("Bits").to_string()))
        .collect();
    source.arch_endianness = path_match_arms(&file, "HostArch", "endianness")
        .into_iter()
        .map(|(variant, order)| (variant, snake_case(&order)))
        .collect();
    source.tiers = tier_match_arms(&file, &source.os_order, &source.arch_order);

    for os in &source.os_order {
        if !source.os_ids.contains_key(os) {
            findings.push(Finding::in_file(
                MATRIX_SOURCE_PATH,
                format!("HostOs::{os} has no arm in `HostOs::id`"),
                "give the variant a stable identifier so generated documents can name it",
            ));
        }
    }
    for arch in &source.arch_order {
        for (table, name) in [
            (&source.arch_ids, "id"),
            (&source.arch_pointer_width, "pointer_width"),
            (&source.arch_endianness, "endianness"),
        ] {
            if !table.contains_key(arch) {
                findings.push(Finding::in_file(
                    MATRIX_SOURCE_PATH,
                    format!("HostArch::{arch} has no arm in `HostArch::{name}`"),
                    "state the architecture fact for the new variant",
                ));
            }
        }
    }
    for os in &source.os_order {
        for arch in &source.arch_order {
            if !source.tiers.contains_key(&(os.clone(), arch.clone())) {
                findings.push(Finding::in_file(
                    MATRIX_SOURCE_PATH,
                    format!(
                        "cell (HostOs::{os}, HostArch::{arch}) has no arm in `PlatformSupportMatrix::tier`"
                    ),
                    "state a support tier for the cell; the match has no catch-all so this is also a compile error",
                ));
            }
        }
    }

    Ok((source, findings))
}

/// Read the portability ledger, keyed by `(os id, arch id)`.
///
/// # Errors
///
/// Returns `GateError` when the ledger exists but cannot be read or parsed.
pub fn read_ledger(
    root: &Path,
) -> Result<(BTreeMap<(String, String), PortabilityRun>, Vec<Finding>), GateError> {
    let path = root.join(LEDGER_PATH);
    let mut findings = Vec::new();
    if !path.exists() {
        return Ok((BTreeMap::new(), findings));
    }
    let text = std::fs::read_to_string(&path).map_err(|error| {
        GateError::new(
            format!("failed to read {LEDGER_PATH}: {error}"),
            "make the portability ledger readable or delete it and re-record the runs",
        )
    })?;
    let ledger: PortabilityLedger = serde_json::from_str(&text).map_err(|error| {
        GateError::new(
            format!("failed to parse {LEDGER_PATH}: {error}"),
            "re-record the runs with `cargo xtask portability-evidence --write`",
        )
    })?;

    if ledger.schema_version != PORTABILITY_LEDGER_SCHEMA_VERSION {
        findings.push(Finding::in_file(
            LEDGER_PATH,
            format!(
                "portability ledger schema version {} is not {PORTABILITY_LEDGER_SCHEMA_VERSION}",
                ledger.schema_version
            ),
            "re-record the runs with `cargo xtask portability-evidence --write`; a stale ledger states results under other rules",
        ));
        return Ok((BTreeMap::new(), findings));
    }

    let mut runs = BTreeMap::new();
    for run in ledger.runs {
        if is_executed(&run.evidence) && run.canonical_identity.is_empty() {
            findings.push(Finding::in_file(
                LEDGER_PATH,
                format!(
                    "run on {} records evidence `{}` with no identity digest",
                    run.target_triple, run.evidence
                ),
                "record the digest the run printed, or downgrade the entry to type-checked",
            ));
        }
        if run.evidence == EVIDENCE_EMULATED && run.emulator.is_empty() {
            findings.push(Finding::in_file(
                LEDGER_PATH,
                format!(
                    "run on {} records an emulated execution without naming the emulator",
                    run.target_triple
                ),
                "name the emulator binary; an emulated run and a native one are different evidence",
            ));
        }
        runs.insert((run.os.clone(), run.arch.clone()), run);
    }

    let executed: Vec<&PortabilityRun> = runs
        .values()
        .filter(|run| is_executed(&run.evidence))
        .collect();
    for pair in ["canonical", "fallback"] {
        let digests: BTreeSet<&str> = executed
            .iter()
            .map(|run| {
                if pair == "canonical" {
                    run.canonical_identity.as_str()
                } else {
                    run.fallback_identity.as_str()
                }
            })
            .collect();
        if digests.len() > 1 {
            let cells: Vec<String> = executed
                .iter()
                .map(|run| format!("{}={}", run.target_triple, run.canonical_identity))
                .collect();
            findings.push(Finding::in_file(
                LEDGER_PATH,
                format!(
                    "recorded {pair} identity is not byte-identical across executed cells: {}",
                    cells.join(", ")
                ),
                "a persisted identity that moves with pointer width or byte order is a wire defect; fix the encoder rather than the ledger",
            ));
        }
    }

    Ok((runs, findings))
}

/// Collect the source-derived platform support matrix.
///
/// # Errors
///
/// Returns `GateError` when the platform source, the root manifest or the
/// portability ledger cannot be read.
pub fn collect_support_matrix(
    root: &Path,
) -> Result<(PlatformSupportMatrixDocument, Vec<Finding>), GateError> {
    let (source, mut findings) = parse_matrix_source(root)?;
    let (ledger, ledger_findings) = read_ledger(root)?;
    findings.extend(ledger_findings);

    let rust_version = workspace_rust_version(root)?;

    let mut declared_cells: BTreeSet<(String, String)> = BTreeSet::new();
    for os in &source.os_order {
        for arch in &source.arch_order {
            declared_cells.insert((
                source.os_ids.get(os).cloned().unwrap_or_else(|| os.clone()),
                source
                    .arch_ids
                    .get(arch)
                    .cloned()
                    .unwrap_or_else(|| arch.clone()),
            ));
        }
    }
    for cell in ledger.keys() {
        if !declared_cells.contains(cell) {
            findings.push(Finding::in_file(
                LEDGER_PATH,
                format!(
                    "ledger records a run for {}/{}, which {MATRIX_SOURCE_PATH} does not declare",
                    cell.0, cell.1
                ),
                "declare the cell in HostOs and HostArch, or drop the ledger entry",
            ));
        }
    }

    let mut host_cells = Vec::with_capacity(source.os_order.len() * source.arch_order.len());
    for os in &source.os_order {
        for arch in &source.arch_order {
            let os_id = source.os_ids.get(os).cloned().unwrap_or_else(|| os.clone());
            let arch_id = source
                .arch_ids
                .get(arch)
                .cloned()
                .unwrap_or_else(|| arch.clone());
            let tier = source
                .tiers
                .get(&(os.clone(), arch.clone()))
                .cloned()
                .unwrap_or_else(|| "undeclared".to_string());
            let run = ledger.get(&(os_id.clone(), arch_id.clone()));

            if let Some(run) = run {
                if tier == "excluded" && is_executed(&run.evidence) {
                    findings.push(Finding::in_file(
                        MATRIX_SOURCE_PATH,
                        format!(
                            "cell {os_id}/{arch_id} is excluded from every claim, and the ledger records an executed run on it"
                        ),
                        "raise the cell to a tier the run supports, or remove the run from the ledger",
                    ));
                }
            }

            host_cells.push(HostCellEntry {
                os: os_id,
                arch: arch_id,
                pointer_width: source
                    .arch_pointer_width
                    .get(arch)
                    .cloned()
                    .unwrap_or_default(),
                endianness: source
                    .arch_endianness
                    .get(arch)
                    .cloned()
                    .unwrap_or_default(),
                requires: requirement_for(&tier, run.map(|run| run.evidence.as_str())),
                tier,
                evidence: run.map_or_else(|| EVIDENCE_NONE.to_string(), |run| run.evidence.clone()),
                target_triple: run.map(|run| run.target_triple.clone()).unwrap_or_default(),
                emulator: run.map(|run| run.emulator.clone()).unwrap_or_default(),
                identity_digest: run
                    .map(|run| run.canonical_identity.clone())
                    .unwrap_or_default(),
            });
        }
    }

    let drivers = collect_drivers(root, &mut findings);
    let package_features = collect_package_features(root, &mut findings);

    let document = PlatformSupportMatrixDocument {
        schema_version: PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION,
        canonical_endianness: "little_endian".to_string(),
        canonical_rust_version: rust_version,
        host_cells,
        drivers,
        package_features,
    };
    Ok((document, findings))
}

/// What would raise a cell's evidence, given what it claims and what ran.
fn requirement_for(tier: &str, evidence: Option<&str>) -> String {
    let evidence = evidence.unwrap_or(EVIDENCE_NONE);
    match (tier, evidence) {
        ("excluded", _) => String::new(),
        ("runtime", EVIDENCE_NATIVE) => String::new(),
        ("runtime", _) => {
            "a native run of the test suite on this cell; an emulator proves host behavior and decoding, never device support or performance"
                .to_string()
        }
        ("encoding", value) if is_executed(value) => String::new(),
        ("encoding", _) => {
            "an executed run of the host identity suite on this cell, native or under a user-mode emulator"
                .to_string()
        }
        _ => "a support tier for this cell in vyre_foundation::platform".to_string(),
    }
}

/// Read the minimum toolchain the workspace manifest declares.
fn workspace_rust_version(root: &Path) -> Result<String, GateError> {
    let path = root.join("Cargo.toml");
    let text = std::fs::read_to_string(&path).map_err(|error| {
        GateError::new(
            format!("failed to read root Cargo.toml: {error}"),
            "ensure the workspace manifest exists and is readable",
        )
    })?;
    let manifest: toml::Table = toml::from_str(&text).map_err(|error| {
        GateError::new(
            format!("failed to parse root Cargo.toml: {error}"),
            "repair the workspace manifest syntax",
        )
    })?;
    manifest
        .get("workspace")
        .and_then(|workspace| workspace.get("package"))
        .and_then(|package| package.get("rust-version"))
        .and_then(toml::Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| {
            GateError::new(
                "root Cargo.toml declares no workspace.package.rust-version",
                "declare the minimum toolchain once in the workspace manifest",
            )
        })
}

/// Prefix every driver package name carries.
const DRIVER_PACKAGE_PREFIX: &str = "vyre-driver-";

/// Suffix of the constant a driver crate declares its registry identifier in.
const BACKEND_ID_SUFFIX: &str = "_BACKEND_ID";

/// Every backend driver the workspace ships, read out of each driver crate.
///
/// The identifier is parsed rather than assembled from the package name. A
/// crate whose constant disagrees with its own package name is a registration
/// nothing can reach by the name the manifest advertises, and that is the
/// defect worth catching.
fn collect_drivers(root: &Path, findings: &mut Vec<Finding>) -> Vec<DriverEntry> {
    let mut located: Vec<(PackageManifest, std::path::PathBuf)> = Vec::new();
    let mut blockers: Vec<String> = Vec::new();
    manifest_walk::collect_manifests(
        root,
        "platform support matrix",
        &mut located,
        &mut blockers,
        |path| {
            let parsed = manifest_walk::parse_package_manifest(path, "platform support matrix")?;
            let Some(manifest) = parsed else {
                return Ok(None);
            };
            let dir = path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| root.to_path_buf());
            Ok(Some((manifest, dir)))
        },
    );
    for blocker in blockers {
        findings.push(Finding::in_file(
            MANIFEST_PATH,
            blocker,
            "repair the manifest syntax so the driver roster can be derived",
        ));
    }
    located.sort_by(|left, right| left.0.name.cmp(&right.0.name));

    let mut entries = Vec::new();
    for (manifest, dir) in &located {
        if !manifest.name.starts_with(DRIVER_PACKAGE_PREFIX) {
            continue;
        }
        let claimed = manifest
            .document
            .get("package")
            .and_then(|table| table.get("publish"))
            .and_then(toml::Value::as_bool)
            != Some(false);
        // A driver crate that declares no identifier registers no backend and
        // is absent from the roster rather than reported here. The reference
        // oracle is such a crate: it executes programs to give a device answer
        // something to be wrong against and is never dispatched to as a device.
        // Absence still costs a crate that claims one, because a
        // `target-support` contract is checked against this roster.
        let declared = backend_id_constants(&dir.join("src"));
        let ids: BTreeSet<&str> = declared.iter().map(|(id, _)| id.as_str()).collect();
        if ids.len() > 1 {
            findings.push(Finding::in_file(
                MANIFEST_PATH,
                format!(
                    "driver package `{}` declares {} different backend identifiers: {}",
                    manifest.name,
                    ids.len(),
                    ids.iter().copied().collect::<Vec<_>>().join(", ")
                ),
                "leave one registry identifier per driver crate",
            ));
        }
        for (id, declared_in) in declared {
            let relative = declared_in
                .strip_prefix(root)
                .unwrap_or(&declared_in)
                .to_string_lossy()
                .replace('\\', "/");
            entries.push(DriverEntry {
                id,
                package: manifest.name.clone(),
                declared_in: relative,
                claimed,
            });
        }
    }
    entries.sort_by(|left, right| {
        (&left.id, &left.package, &left.declared_in).cmp(&(
            &right.id,
            &right.package,
            &right.declared_in,
        ))
    });
    entries
}

/// Every `pub const *_BACKEND_ID: &str = "..."` under a crate source tree,
/// with the file it was declared in.
///
/// The constant is not always in `lib.rs`. One driver declares it in a
/// submodule and re-exports it, so a reader that opened only the crate root
/// would report that crate as declaring no identifier at all.
fn backend_id_constants(src: &Path) -> Vec<(String, std::path::PathBuf)> {
    let mut found = Vec::new();
    for entry in crate::tree_walk::pruned(src, crate::tree_walk::BUILD_OUTPUT_AND_VCS) {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let Ok(file) = syn::parse_file(&text) else {
            continue;
        };
        for item in &file.items {
            let Item::Const(item) = item else {
                continue;
            };
            if !matches!(item.vis, syn::Visibility::Public(_))
                || !item.ident.to_string().ends_with(BACKEND_ID_SUFFIX)
            {
                continue;
            }
            if let Expr::Lit(literal) = item.expr.as_ref() {
                if let Lit::Str(value) = &literal.lit {
                    found.push((value.value(), path.to_path_buf()));
                }
            }
        }
    }
    found.sort();
    found
}

/// Every declared feature of every workspace package, with the publication
/// class the package states for itself.
fn collect_package_features(root: &Path, findings: &mut Vec<Finding>) -> Vec<PackageFeatureEntry> {
    let mut manifests: Vec<PackageManifest> = Vec::new();
    let mut blockers: Vec<String> = Vec::new();
    manifest_walk::collect_manifests(
        root,
        "platform support matrix",
        &mut manifests,
        &mut blockers,
        |path| manifest_walk::parse_package_manifest(path, "platform support matrix"),
    );
    for blocker in blockers {
        findings.push(Finding::in_file(
            MANIFEST_PATH,
            blocker,
            "repair the manifest syntax so package features can be derived",
        ));
    }
    manifests.sort_by(|left, right| left.name.cmp(&right.name));

    let mut entries = Vec::new();
    for package in &manifests {
        let publication_class = package
            .document
            .get("package")
            .and_then(|table| table.get("metadata"))
            .and_then(|table| table.get("vyre"))
            .and_then(|table| table.get("publication_class"))
            .and_then(toml::Value::as_str)
            .unwrap_or("undeclared")
            .to_string();
        let features: BTreeSet<String> = package
            .document
            .get("features")
            .and_then(toml::Value::as_table)
            .map(|table| table.keys().cloned().collect())
            .unwrap_or_default();
        if features.is_empty() {
            entries.push(PackageFeatureEntry {
                package: package.name.clone(),
                feature: "default".to_string(),
                publication_class: publication_class.clone(),
            });
            continue;
        }
        for feature in features {
            entries.push(PackageFeatureEntry {
                package: package.name.clone(),
                feature,
                publication_class: publication_class.clone(),
            });
        }
    }
    entries
}

/// Variant idents of a unit-only enum, in declaration order.
fn enum_variants(file: &syn::File, name: &str) -> Vec<String> {
    for item in &file.items {
        if let Item::Enum(item) = item {
            if item.ident == name {
                return item
                    .variants
                    .iter()
                    .filter(|variant| matches!(variant.fields, Fields::Unit))
                    .map(|variant| variant.ident.to_string())
                    .collect();
            }
        }
    }
    Vec::new()
}

/// The `match self` in an inherent method, when it has exactly one.
fn self_match<'a>(
    file: &'a syn::File,
    type_name: &str,
    fn_name: &str,
) -> Option<&'a syn::ExprMatch> {
    for item in &file.items {
        let Item::Impl(item) = item else { continue };
        if item.trait_.is_some() || !type_path_is(&item.self_ty, type_name) {
            continue;
        }
        for member in &item.items {
            let ImplItem::Fn(member) = member else {
                continue;
            };
            if member.sig.ident != fn_name {
                continue;
            }
            for statement in &member.block.stmts {
                let expr = match statement {
                    syn::Stmt::Expr(expr, _) => expr,
                    _ => continue,
                };
                if let Expr::Match(expr) = expr {
                    return Some(expr);
                }
            }
        }
    }
    None
}

/// The expression an arm evaluates to, seeing through a braced body.
///
/// An arm whose pattern is long enough is wrapped by the formatter into
/// `pattern => { value }`, which parses as a block and not as the literal or
/// path the readers below match on. Reading only the unwrapped form made the
/// widest arm in `PlatformSupportMatrix::tier` invisible, so the gate reported
/// six cells as having no arm in a match that has no catch-all and therefore
/// could not have compiled with a cell missing.
fn arm_value(body: &Expr) -> &Expr {
    let Expr::Block(block) = body else {
        return body;
    };
    let [syn::Stmt::Expr(inner, None)] = block.block.stmts.as_slice() else {
        return body;
    };
    arm_value(inner)
}

/// Variant ident to string literal, from `Self::Variant => "lit"` arms.
fn string_match_arms(file: &syn::File, type_name: &str, fn_name: &str) -> BTreeMap<String, String> {
    let mut arms = BTreeMap::new();
    let Some(expr) = self_match(file, type_name, fn_name) else {
        return arms;
    };
    for arm in &expr.arms {
        let Expr::Lit(literal) = arm_value(arm.body.as_ref()) else {
            continue;
        };
        let Lit::Str(value) = &literal.lit else {
            continue;
        };
        for variant in pattern_variants(&arm.pat) {
            arms.insert(variant, value.value());
        }
    }
    arms
}

/// Variant ident to the final path segment of the arm body, from
/// `Self::Variant => Enum::Member` arms.
fn path_match_arms(file: &syn::File, type_name: &str, fn_name: &str) -> BTreeMap<String, String> {
    let mut arms = BTreeMap::new();
    let Some(expr) = self_match(file, type_name, fn_name) else {
        return arms;
    };
    for arm in &expr.arms {
        let Expr::Path(path) = arm_value(arm.body.as_ref()) else {
            continue;
        };
        let Some(last) = path.path.segments.last() else {
            continue;
        };
        let member = last.ident.to_string();
        for variant in pattern_variants(&arm.pat) {
            arms.insert(variant, member.clone());
        }
    }
    arms
}

/// `(HostOs, HostArch)` variant pairs to the tier the first matching arm
/// states, following match order so an earlier arm wins exactly as it does at
/// run time.
fn tier_match_arms(
    file: &syn::File,
    os_order: &[String],
    arch_order: &[String],
) -> BTreeMap<(String, String), String> {
    let mut tiers = BTreeMap::new();
    let Some(expr) = self_match(file, "PlatformSupportMatrix", "tier") else {
        return tiers;
    };
    for arm in &expr.arms {
        let Expr::Path(path) = arm_value(arm.body.as_ref()) else {
            continue;
        };
        let Some(tier) = path.path.segments.last() else {
            continue;
        };
        let tier = snake_case(&tier.ident.to_string());
        let Pat::Tuple(tuple) = &arm.pat else {
            continue;
        };
        let mut elements = tuple.elems.iter();
        let (Some(os_pat), Some(arch_pat)) = (elements.next(), elements.next()) else {
            continue;
        };
        let matched_os = pattern_members(os_pat, os_order);
        let matched_arch = pattern_members(arch_pat, arch_order);
        for os in &matched_os {
            for arch in &matched_arch {
                tiers
                    .entry((os.clone(), arch.clone()))
                    .or_insert_with(|| tier.clone());
            }
        }
    }
    tiers
}

/// Variant idents a pattern names, following `|` alternatives.
fn pattern_variants(pat: &Pat) -> Vec<String> {
    match pat {
        Pat::Path(path) => path
            .path
            .segments
            .last()
            .map(|segment| vec![segment.ident.to_string()])
            .unwrap_or_default(),
        Pat::Ident(ident) => vec![ident.ident.to_string()],
        Pat::Or(alternatives) => alternatives
            .cases
            .iter()
            .flat_map(pattern_variants)
            .collect(),
        _ => Vec::new(),
    }
}

/// Members a pattern covers, with a wildcard covering every declared member.
fn pattern_members(pat: &Pat, declared: &[String]) -> Vec<String> {
    if matches!(pat, Pat::Wild(_)) {
        return declared.to_vec();
    }
    pattern_variants(pat)
}

/// Whether a type is the named path, ignoring generics.
fn type_path_is(ty: &syn::Type, name: &str) -> bool {
    let syn::Type::Path(path) = ty else {
        return false;
    };
    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == name)
}

/// `LittleEndian` to `little_endian`, matching the serde rename the platform
/// enums declare.
fn snake_case(ident: &str) -> String {
    let mut out = String::with_capacity(ident.len() + 2);
    for (index, ch) in ident.char_indices() {
        if ch.is_ascii_uppercase() {
            if index != 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
/// Contract tests for the platform support matrix gate.
pub mod tests {
    use super::*;

    /// WHY: every driver crate the workspace ships must appear in the roster
    /// under the identifier its own source declares. This closes the class the
    /// hand-written rows left open, where a driver could be added, renamed or
    /// removed and the document kept claiming whatever was typed into it. The
    /// crate set and the identifier are both read at run time, so adding a
    /// driver crate that declares an identifier changes this document with no
    /// edit here.
    ///
    /// What it does not catch: a crate that declares the constant and never
    /// registers it with the backend registry. Reaching that needs the
    /// registry to run, which is what the backend matrix evidence does.
    #[test]
    pub fn every_driver_crate_reaches_the_roster_under_its_declared_identifier() {
        let root = crate::checkout::checkout_root();
        let (document, _) = collect_support_matrix(&root).expect("matrix collection must succeed");

        let mut expected: BTreeSet<(String, String)> = BTreeSet::new();
        for entry in std::fs::read_dir(&root).expect("the checkout root must be readable") {
            let entry = entry.expect("a checkout root entry must be readable");
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with(DRIVER_PACKAGE_PREFIX) || !entry.path().is_dir() {
                continue;
            }
            for (id, _) in backend_id_constants(&entry.path().join("src")) {
                expected.insert((id, name.clone()));
            }
        }
        assert!(
            !expected.is_empty(),
            "the workspace must ship at least one driver crate declaring a backend identifier"
        );

        let recorded: BTreeSet<(String, String)> = document
            .drivers
            .iter()
            .map(|driver| (driver.id.clone(), driver.package.clone()))
            .collect();
        assert_eq!(
            recorded, expected,
            "the driver roster must state exactly the identifiers the driver crates declare"
        );
        for driver in &document.drivers {
            assert!(
                root.join(&driver.declared_in).is_file(),
                "driver `{}` names a declaration site that does not exist: {}",
                driver.id,
                driver.declared_in
            );
        }
    }

    /// WHY: the cell space must come from the platform source, so that adding
    /// a `HostOs` or `HostArch` variant changes this document rather than
    /// leaving a hand-written list two behind its enum. This closes the class
    /// by reading the enums and the tier match at run time; it does not name a
    /// single cell.
    ///
    /// What it does not catch: a variant declared with an `id` arm and a tier
    /// arm but never reachable from `current()`. That pairing is a compile
    /// error in the platform source, not a document defect.
    #[test]
    pub fn every_declared_variant_reaches_the_document() {
        let root = crate::checkout::checkout_root();
        let (source, source_findings) =
            parse_matrix_source(&root).expect("the platform source must parse");
        assert!(
            source_findings.is_empty(),
            "the platform source must declare an id, an architecture fact and a tier for every variant: {source_findings:?}"
        );

        let (document, findings) =
            collect_support_matrix(&root).expect("matrix collection must succeed");
        assert!(
            findings.is_empty(),
            "a consistent tree must produce no findings: {findings:?}"
        );
        assert_eq!(
            document.host_cells.len(),
            source.os_order.len() * source.arch_order.len(),
            "the document must carry the full cross product of the declared cell space"
        );

        for os in &source.os_order {
            let os_id = &source.os_ids[os];
            assert!(
                document.host_cells.iter().any(|cell| cell.os == *os_id),
                "HostOs::{os} is declared in source and absent from the document"
            );
        }
        for arch in &source.arch_order {
            let arch_id = &source.arch_ids[arch];
            let cell = document
                .host_cells
                .iter()
                .find(|cell| cell.arch == *arch_id)
                .unwrap_or_else(|| panic!("HostArch::{arch} is absent from the document"));
            assert_eq!(cell.pointer_width, source.arch_pointer_width[arch]);
            assert_eq!(cell.endianness, source.arch_endianness[arch]);
        }
    }

    /// WHY: an arm wide enough to wrap is written `pattern => { value }` by
    /// the formatter, and a reader that matched only the unwrapped body read
    /// it as absent. That inverted the gate: it reported six cells as having
    /// no arm in a match with no catch-all, which could not have compiled.
    /// Both body shapes are read here, in the same match, so a reader that
    /// handles one and not the other fails.
    ///
    /// What it does not catch: an arm whose body computes the tier instead of
    /// naming it. The platform source states tiers as constants.
    #[test]
    pub fn a_braced_arm_body_states_its_tier() {
        let file: syn::File = syn::parse_str(
            "impl PlatformSupportMatrix {
                pub const fn tier(os: HostOs, arch: HostArch) -> HostSupportTier {
                    match (os, arch) {
                        (HostOs::Linux | HostOs::MacOS, HostArch::X86_64 | HostArch::AArch64) => {
                            HostSupportTier::Runtime
                        }
                        (HostOs::Linux, HostArch::S390x) => HostSupportTier::Encoding,
                        (_, _) => HostSupportTier::Excluded,
                    }
                }
            }",
        )
        .expect("the fixture parses");
        let os = ["Linux".to_string(), "MacOS".to_string()];
        let arch = [
            "X86_64".to_string(),
            "AArch64".to_string(),
            "S390x".to_string(),
        ];
        let tiers = tier_match_arms(&file, &os, &arch);

        for cell in [("Linux", "X86_64"), ("MacOS", "AArch64")] {
            assert_eq!(
                tiers.get(&(cell.0.to_string(), cell.1.to_string())),
                Some(&"runtime".to_string()),
                "the braced cross-product arm must state a tier for {cell:?}"
            );
        }
        assert_eq!(
            tiers.get(&("Linux".to_string(), "S390x".to_string())),
            Some(&"encoding".to_string()),
            "the unwrapped arm must still be read"
        );
        assert_eq!(
            tiers.get(&("MacOS".to_string(), "S390x".to_string())),
            Some(&"excluded".to_string()),
            "a wildcard arm must still cover the cells no earlier arm claimed"
        );
    }

    /// WHY: the defect this gate shipped was claiming evidence for cells that
    /// never ran. Evidence is raised only by a ledger entry, and a cell with
    /// no entry publishes as unproven with the run it still needs.
    #[test]
    pub fn a_cell_with_no_recorded_run_is_published_as_unproven() {
        let root = crate::checkout::checkout_root();
        let (document, _) = collect_support_matrix(&root).expect("matrix collection must succeed");
        let (ledger, _) = read_ledger(&root).expect("the ledger must be readable");

        for cell in &document.host_cells {
            let recorded = ledger.get(&(cell.os.clone(), cell.arch.clone()));
            match recorded {
                Some(run) => assert_eq!(cell.evidence, run.evidence),
                None => assert_eq!(
                    cell.evidence, EVIDENCE_NONE,
                    "cell {}/{} claims evidence `{}` with no recorded run",
                    cell.os, cell.arch, cell.evidence
                ),
            }
            if is_executed(&cell.evidence) {
                assert!(
                    !cell.identity_digest.is_empty(),
                    "cell {}/{} records an executed run with no identity digest",
                    cell.os,
                    cell.arch
                );
            }
            let needs_proof = !is_executed(&cell.evidence) && cell.tier != "excluded";
            assert_eq!(
                needs_proof,
                !cell.requires.is_empty(),
                "cell {}/{} at tier {} states requires=`{}` for evidence `{}`",
                cell.os,
                cell.arch,
                cell.tier,
                cell.requires,
                cell.evidence
            );
        }
    }

    /// WHY: a cross-compilation is not evidence of support. A ledger entry
    /// recording only a type check must leave its cell unproven, and an
    /// executed entry with no digest must be rejected rather than trusted.
    #[test]
    pub fn a_type_checked_run_never_counts_as_evidence() {
        assert!(!is_executed(EVIDENCE_TYPE_CHECKED));
        assert!(!is_executed(EVIDENCE_NONE));
        assert!(is_executed(EVIDENCE_NATIVE));
        assert!(is_executed(EVIDENCE_EMULATED));

        assert!(!requirement_for("runtime", Some(EVIDENCE_TYPE_CHECKED)).is_empty());
        assert!(!requirement_for("encoding", Some(EVIDENCE_TYPE_CHECKED)).is_empty());
        assert!(!requirement_for("runtime", Some(EVIDENCE_EMULATED)).is_empty());
        assert!(requirement_for("runtime", Some(EVIDENCE_NATIVE)).is_empty());
        assert!(requirement_for("encoding", Some(EVIDENCE_EMULATED)).is_empty());
        assert!(requirement_for("excluded", None).is_empty());
    }

    /// WHY: the identity digest is the whole point of the encoding-tier cells.
    /// Two executed cells that disagree mean a persisted identity moves with
    /// pointer width or byte order, and the gate must say so rather than
    /// publish both.
    #[test]
    pub fn disagreeing_identity_digests_are_a_finding() {
        let root = crate::checkout::checkout_root();
        let (recorded, findings) = read_ledger(&root).expect("the ledger must be readable");
        assert!(
            findings.is_empty(),
            "the recorded ledger must be self-consistent: {findings:?}"
        );

        let digests: BTreeSet<&str> = recorded
            .values()
            .filter(|run| is_executed(&run.evidence))
            .map(|run| run.canonical_identity.as_str())
            .collect();
        assert!(
            digests.len() <= 1,
            "executed cells recorded different canonical identities: {digests:?}"
        );
    }

    /// WHY: the tier match is the compile-time closure that forces a decision
    /// for a new cell. Parsing it must follow match order, because an earlier
    /// arm wins at run time and a parser that let a later wildcard overwrite
    /// it would publish a tier the code does not use.
    #[test]
    pub fn tier_parsing_follows_match_order() {
        let file: syn::File = syn::parse_str(
            r#"
            enum HostOs { Linux, Windows }
            enum HostArch { X86_64, Wasm32 }
            impl PlatformSupportMatrix {
                pub const fn tier(os: HostOs, arch: HostArch) -> HostSupportTier {
                    match (os, arch) {
                        (HostOs::Linux, HostArch::X86_64) => HostSupportTier::Runtime,
                        (HostOs::Linux | HostOs::Windows, _) => HostSupportTier::Excluded,
                    }
                }
            }
            "#,
        )
        .expect("the fixture must parse");
        let os_order = vec!["Linux".to_string(), "Windows".to_string()];
        let arch_order = vec!["X86_64".to_string(), "Wasm32".to_string()];
        let tiers = tier_match_arms(&file, &os_order, &arch_order);

        assert_eq!(
            tiers[&("Linux".to_string(), "X86_64".to_string())],
            "runtime"
        );
        assert_eq!(
            tiers[&("Linux".to_string(), "Wasm32".to_string())],
            "excluded"
        );
        assert_eq!(
            tiers[&("Windows".to_string(), "X86_64".to_string())],
            "excluded"
        );
        assert_eq!(tiers.len(), 4);
    }

    /// WHY: the matrix exists to say what is supported and on what evidence, so
    /// the two ways a cell can be empty are the two the gate must never pass
    /// silently. A cell the platform source states no tier for arrives here as
    /// `undeclared`, and a cell whose tier outranks the evidence recorded for
    /// it is a claim with nothing behind it. Both have to name what would
    /// raise them.
    ///
    /// The tier list is read from the platform source at run time rather than
    /// written down here, so a tier added there with no arm in
    /// `requirement_for` falls to the catch-all and turns this red instead of
    /// being reported as satisfied.
    ///
    /// What this does not catch: whether the evidence a ledger row records was
    /// truthfully produced. That is the attribution gate's subject.
    #[test]
    pub fn unclaimed_or_missing_cell_fails_gate() {
        for evidence in [
            None,
            Some(EVIDENCE_NONE),
            Some(EVIDENCE_TYPE_CHECKED),
            Some(EVIDENCE_EMULATED),
            Some(EVIDENCE_NATIVE),
        ] {
            assert!(
                !requirement_for("undeclared", evidence).is_empty(),
                "a cell with no declared tier states no requirement at evidence {evidence:?}"
            );
        }

        let root = crate::checkout::checkout_root();
        let (source, _) = parse_matrix_source(&root).expect("the platform source parses");
        let declared: std::collections::BTreeSet<&str> =
            source.tiers.values().map(String::as_str).collect();
        assert!(
            !declared.is_empty(),
            "the platform source declared no tier at all, so nothing below is measured"
        );

        for tier in declared {
            let satisfied = requirement_for(tier, Some(EVIDENCE_NATIVE));
            assert!(
                satisfied.is_empty(),
                "tier `{tier}` is not satisfied by a native run, so it has no arm in \
                 requirement_for and every cell claiming it reads as unraisable: {satisfied}"
            );
            if tier == "excluded" {
                continue;
            }
            assert!(
                !requirement_for(tier, Some(EVIDENCE_NONE)).is_empty(),
                "tier `{tier}` claims support with no evidence and asks for nothing"
            );
        }
    }
}
