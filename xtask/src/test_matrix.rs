//! Test architecture evidence for the Vyre/Weir release.

use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::Serialize;
use walkdir::WalkDir;

#[derive(Debug, Serialize)]
struct TestMatrix {
    schema_version: u32,
    test_files: usize,
    vyre_test_files: usize,
    weir_test_files: usize,
    vyrec_test_files: usize,
    layers: Vec<String>,
    surface_coverages: Vec<SurfaceCoverage>,
    modular_directories: Vec<ModularDirectory>,
    oversized_files: Vec<OversizedFile>,
    god_test_candidates: Vec<String>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SurfaceCoverage {
    surface: &'static str,
    file_count: usize,
    assertion_count: usize,
    entrypoint_count: usize,
    layers: Vec<String>,
    required_layers: Vec<&'static str>,
    missing_layers: Vec<&'static str>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct OversizedFile {
    path: String,
    lines: usize,
    lines_over_threshold: usize,
    recommended_split: Vec<String>,
    release_blocker: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ModularDirectory {
    surface: &'static str,
    layer: &'static str,
    path: String,
    exists: bool,
}

#[derive(Debug, Clone, Serialize)]
struct TestFileRecord {
    path: String,
    layers: Vec<String>,
    lines: usize,
    has_test_entrypoint: bool,
    assertion_count: usize,
    oversized: bool,
    god_test_candidate: bool,
    recommended_split: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ModularizationMap {
    schema_version: u32,
    directories: Vec<ModularDirectory>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OversizedTestClosure {
    schema_version: u32,
    threshold_lines: usize,
    closed: bool,
    total_oversized_files: usize,
    total_god_test_candidates: usize,
    oversized_files: Vec<OversizedFile>,
    god_test_candidates: Vec<String>,
    required_split_count: usize,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SuiteEvidence {
    schema_version: u32,
    suite: String,
    file_count: usize,
    vyre_file_count: usize,
    dataflow_consumer_file_count: usize,
    vyrec_file_count: usize,
    files: Vec<TestFileRecord>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SurfaceCoverageArtifact {
    schema_version: u32,
    surfaces: Vec<SurfaceCoverage>,
    blockers: Vec<String>,
}

const REQUIRED_LAYERS: &[&str] = &[
    "unit",
    "integration",
    "property",
    "adversarial",
    "corpus",
    "benchmark",
    "conformance",
    "gap",
    "fuzz",
];

const REQUIRED_MODULAR_DIRS: &[(&str, &str)] = &[
    ("fixtures", "tests/fixtures"),
    ("contracts", "tests/contracts"),
    ("properties", "tests/properties"),
    ("backends", "tests/backends"),
    ("corpus", "tests/corpus"),
    ("bench", "benches"),
    ("regression", "tests/regression"),
];

const MAX_TEST_SOURCE_BYTES: u64 = 2_097_152;

const RELEASE_SURFACES: &[(&str, &[&str])] = &[
    ("vyre", REQUIRED_LAYERS),
    ("weir", REQUIRED_LAYERS),
    ("vyrec", REQUIRED_LAYERS),
];

const OVERSIZED_TEST_THRESHOLD_LINES: usize = 500;

pub(crate) fn run(args: &[String]) {
    let output = match parse_output(args) {
        Ok(output) => output,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let vyre_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let santh_root = vyre_root
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| vyre_root.clone());
    let modular_roots = [
        ("vyre", vyre_root.clone()),
        ("weir", santh_root.join("libs/dataflow/weir")),
        ("vyrec", santh_root.join("tools/vyrec")),
    ];
    let test_roots = [
        vyre_root,
        santh_root.join("libs/dataflow/weir"),
        santh_root.join("tools/vyrec"),
    ];
    let mut test_files = 0usize;
    let mut layers = BTreeSet::new();
    let mut oversized_files = Vec::new();
    let mut modular_directories = Vec::new();
    let mut file_records = Vec::new();
    let mut scan_blockers = Vec::new();
    for root in &test_roots {
        scan_tests(
            root,
            &mut test_files,
            &mut layers,
            &mut oversized_files,
            &mut file_records,
            &mut scan_blockers,
        );
    }
    for (surface, root) in &modular_roots {
        collect_modular_dirs(surface, root, &mut modular_directories);
    }
    let mut blockers = Vec::new();
    blockers.extend(scan_blockers);
    for required in REQUIRED_LAYERS {
        if !layers.contains(*required) {
            blockers.push(format!("missing required test layer `{required}`"));
        }
    }
    if !oversized_files.is_empty() {
        blockers.push(format!(
            "{} test file(s) exceed the {OVERSIZED_TEST_THRESHOLD_LINES}-line modularity threshold",
            oversized_files.len()
        ));
    }
    let god_test_candidates = file_records
        .iter()
        .filter(|file| file.god_test_candidate)
        .map(|file| file.path.clone())
        .collect::<Vec<_>>();
    if !god_test_candidates.is_empty() {
        blockers.push(format!(
            "{} monolithic test file(s) must be split into modular test layers",
            god_test_candidates.len()
        ));
    }
    for directory in &modular_directories {
        if !directory.exists {
            blockers.push(format!(
                "missing modular test directory `{}` for `{}`",
                directory.path, directory.layer
            ));
        }
    }
    let vyre_test_files = file_records
        .iter()
        .filter(|file| {
            !file.path.contains("/libs/dataflow/weir/") && !file.path.contains("/tools/vyrec/")
        })
        .count();
    let weir_test_files = file_records
        .iter()
        .filter(|file| file.path.contains("/libs/dataflow/weir/"))
        .count();
    let vyrec_test_files = file_records
        .iter()
        .filter(|file| file.path.contains("/tools/vyrec/"))
        .count();
    if vyre_test_files == 0 {
        blockers.push("test matrix has zero Vyre release-surface test files".to_string());
    }
    if weir_test_files == 0 {
        blockers.push("test matrix has zero Weir release-surface test files".to_string());
    }
    if vyrec_test_files == 0 {
        blockers.push("test matrix has zero tools/vyrec release-surface test files".to_string());
    }
    let surface_coverages = release_surface_coverages(&file_records);
    for surface in &surface_coverages {
        for blocker in &surface.blockers {
            blockers.push(blocker.clone());
        }
    }
    let matrix = TestMatrix {
        schema_version: 1,
        test_files,
        vyre_test_files,
        weir_test_files,
        vyrec_test_files,
        layers: layers.into_iter().map(String::from).collect(),
        surface_coverages,
        modular_directories,
        oversized_files,
        god_test_candidates,
        blockers,
    };
    let json = match serde_json::to_string_pretty(&matrix) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize test matrix: {error}");
            std::process::exit(1);
        }
    };
    if let Some(parent) = output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    if let Err(error) = fs::write(&output, format!("{json}\n")) {
        eprintln!("Fix: failed to write `{}`: {error}", output.display());
        std::process::exit(1);
    }
    write_sibling_artifacts(&output, &matrix, &file_records);
    println!("test-matrix: wrote {}", output.display());
    if !matrix.blockers.is_empty() {
        std::process::exit(1);
    }
}

fn write_sibling_artifacts(output: &Path, matrix: &TestMatrix, files: &[TestFileRecord]) {
    let Some(parent) = output.parent() else {
        eprintln!(
            "Fix: test matrix output `{}` has no parent directory.",
            output.display()
        );
        std::process::exit(1);
    };
    let modular_blockers = matrix
        .modular_directories
        .iter()
        .filter(|directory| !directory.exists)
        .map(|directory| {
            format!(
                "missing modular test directory `{}` for `{}`",
                directory.path, directory.layer
            )
        })
        .collect::<Vec<_>>();
    write_json(
        &parent.join("modularization-map.json"),
        &ModularizationMap {
            schema_version: 1,
            directories: matrix.modular_directories.clone(),
            blockers: modular_blockers,
        },
    );
    let oversized_blockers =
        if matrix.oversized_files.is_empty() && matrix.god_test_candidates.is_empty() {
            Vec::new()
        } else {
            let mut blockers = vec![format!(
            "{} test file(s) exceed the {OVERSIZED_TEST_THRESHOLD_LINES}-line modularity threshold",
            matrix.oversized_files.len()
        )];
            if !matrix.god_test_candidates.is_empty() {
                blockers.push(format!(
                    "{} monolithic tests.rs file(s) still need modularization",
                    matrix.god_test_candidates.len()
                ));
            }
            blockers
        };
    write_json(
        &parent.join("oversized-test-closure.json"),
        &OversizedTestClosure {
            schema_version: 1,
            threshold_lines: OVERSIZED_TEST_THRESHOLD_LINES,
            closed: matrix.oversized_files.is_empty() && matrix.god_test_candidates.is_empty(),
            total_oversized_files: matrix.oversized_files.len(),
            total_god_test_candidates: matrix.god_test_candidates.len(),
            required_split_count: matrix
                .oversized_files
                .iter()
                .map(|file| file.recommended_split.len())
                .sum(),
            oversized_files: matrix.oversized_files.clone(),
            god_test_candidates: matrix.god_test_candidates.clone(),
            blockers: oversized_blockers,
        },
    );
    for (suite, artifact) in [
        ("unit", "unit-suite.json"),
        ("adversarial", "adversarial-suite.json"),
        ("property", "property-suite.json"),
        ("conformance", "conformance-suite.json"),
        ("corpus", "corpus-suite.json"),
        ("benchmark", "benchmark-suite.json"),
        ("gap", "gap-suite.json"),
        ("fuzz", "fuzz-suite.json"),
    ] {
        write_suite_artifact(parent, suite, artifact, files);
    }
    write_json(
        &parent.join("release-surface-suite-coverage.json"),
        &SurfaceCoverageArtifact {
            schema_version: 1,
            surfaces: matrix.surface_coverages.clone(),
            blockers: matrix
                .surface_coverages
                .iter()
                .flat_map(|surface| surface.blockers.iter().cloned())
                .collect(),
        },
    );
}

fn write_suite_artifact(parent: &Path, suite: &str, artifact: &str, files: &[TestFileRecord]) {
    let suite_files = files
        .iter()
        .filter(|file| file.layers.iter().any(|layer| layer == suite))
        .cloned()
        .collect::<Vec<_>>();
    let blockers = if suite_files.is_empty() {
        vec![format!("test suite `{suite}` has zero files")]
    } else {
        let mut blockers = Vec::new();
        let vyre_file_count = suite_files
            .iter()
            .filter(|file| {
                !file.path.contains("/libs/dataflow/weir/") && !file.path.contains("/tools/vyrec/")
            })
            .count();
        let dataflow_consumer_file_count = suite_files
            .iter()
            .filter(|file| file.path.contains("/libs/dataflow/weir/"))
            .count();
        let vyrec_file_count = suite_files
            .iter()
            .filter(|file| file.path.contains("/tools/vyrec/"))
            .count();
        if vyre_file_count == 0 {
            blockers.push(format!("test suite `{suite}` has zero Vyre-side files"));
        }
        if dataflow_consumer_file_count == 0 {
            blockers.push(format!("test suite `{suite}` has zero Weir-side files"));
        }
        if vyrec_file_count == 0 {
            blockers.push(format!(
                "test suite `{suite}` has zero tools/vyrec-side files"
            ));
        }
        let asserted_files = suite_files
            .iter()
            .filter(|file| {
                file.assertion_count > 0 || file.layers.iter().any(|layer| layer == "benchmark")
            })
            .count();
        if asserted_files == 0 {
            blockers.push(format!(
                "test suite `{suite}` has no files with assertions or benchmark bodies"
            ));
        }
        let entrypoint_files = suite_files
            .iter()
            .filter(|file| {
                file.has_test_entrypoint || file.layers.iter().any(|layer| layer == "benchmark")
            })
            .count();
        if entrypoint_files == 0 {
            blockers.push(format!(
                "test suite `{suite}` has no #[test], proptest!, criterion, or bench entrypoints"
            ));
        }
        blockers
    };
    let vyre_file_count = suite_files
        .iter()
        .filter(|file| {
            !file.path.contains("/libs/dataflow/weir/") && !file.path.contains("/tools/vyrec/")
        })
        .count();
    let dataflow_consumer_file_count = suite_files
        .iter()
        .filter(|file| file.path.contains("/libs/dataflow/weir/"))
        .count();
    let vyrec_file_count = suite_files
        .iter()
        .filter(|file| file.path.contains("/tools/vyrec/"))
        .count();
    write_json(
        &parent.join(artifact),
        &SuiteEvidence {
            schema_version: 1,
            suite: suite.to_string(),
            file_count: suite_files.len(),
            vyre_file_count,
            dataflow_consumer_file_count,
            vyrec_file_count,
            files: suite_files,
            blockers,
        },
    );
}


fn release_surface_coverages(files: &[TestFileRecord]) -> Vec<SurfaceCoverage> {
    RELEASE_SURFACES
        .iter()
        .map(|&(surface, required_layers)| {
            let surface_files = files
                .iter()
                .filter(|file| file_belongs_to_surface(&file.path, surface))
                .collect::<Vec<_>>();
            let mut layers = BTreeSet::new();
            let mut assertion_count = 0usize;
            let mut entrypoint_count = 0usize;
            for file in &surface_files {
                assertion_count += file.assertion_count;
                if file.has_test_entrypoint
                    || file.layers.iter().any(|layer| layer == "benchmark")
                {
                    entrypoint_count += 1;
                }
                for layer in &file.layers {
                    layers.insert(layer.as_str());
                }
            }
            let missing_layers = required_layers
                .iter()
                .copied()
                .filter(|layer| !layers.contains(layer))
                .collect::<Vec<_>>();
            let mut blockers = Vec::new();
            if surface_files.is_empty() {
                blockers.push(format!("release surface `{surface}` has zero test files"));
            }
            if assertion_count == 0 {
                blockers.push(format!(
                    "release surface `{surface}` has no assertions across its test files"
                ));
            }
            if entrypoint_count == 0 {
                blockers.push(format!(
                    "release surface `{surface}` has no executable test, proptest, fuzz, or benchmark entrypoints"
                ));
            }
            for layer in &missing_layers {
                blockers.push(format!(
                    "release surface `{surface}` is missing required `{layer}` test coverage"
                ));
            }
            SurfaceCoverage {
                surface,
                file_count: surface_files.len(),
                assertion_count,
                entrypoint_count,
                layers: layers.into_iter().map(String::from).collect(),
                required_layers: required_layers.to_vec(),
                missing_layers,
                blockers,
            }
        })
        .collect()
}

fn file_belongs_to_surface(path: &str, surface: &str) -> bool {
    match surface {
        "weir" => path.contains("/libs/dataflow/weir/"),
        "vyrec" => path.contains("/tools/vyrec/"),
        "vyre" => !path.contains("/libs/dataflow/weir/") && !path.contains("/tools/vyrec/"),
        _ => false,
    }
}

fn write_json(path: &Path, value: &impl Serialize) {
    let json = match serde_json::to_string_pretty(value) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize `{}`: {error}", path.display());
            std::process::exit(1);
        }
    };
    if let Err(error) = fs::write(path, format!("{json}\n")) {
        eprintln!("Fix: failed to write `{}`: {error}", path.display());
        std::process::exit(1);
    }
}

fn collect_modular_dirs(
    surface: &'static str,
    root: &Path,
    modular_directories: &mut Vec<ModularDirectory>,
) {
    for &(layer, relative) in REQUIRED_MODULAR_DIRS {
        let path = root.join(relative);
        modular_directories.push(ModularDirectory {
            surface,
            layer,
            path: path.display().to_string(),
            exists: path.is_dir(),
        });
    }
}

fn scan_tests(
    root: &Path,
    test_files: &mut usize,
    layers: &mut BTreeSet<&'static str>,
    oversized_files: &mut Vec<OversizedFile>,
    file_records: &mut Vec<TestFileRecord>,
    blockers: &mut Vec<String>,
) {
    for entry in WalkDir::new(root).into_iter().filter_entry(|entry| {
        let name = entry.file_name().to_string_lossy();
        !matches!(
            name.as_ref(),
            "target" | "target-codex" | "target_tests" | ".git" | ".cargo-target" | "release"
        )
    }) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                blockers.push(format!(
                    "failed to walk test evidence root `{}`: {error}",
                    error
                        .path()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| root.display().to_string())
                ));
                continue;
            }
        };
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let path_string = path.display().to_string();
        let text = match read_text_bounded(path) {
            Ok(text) => text,
            Err(error) => {
                blockers.push(format!(
                    "failed to read test evidence file `{}`: {error}",
                    path.display()
                ));
                continue;
            }
        };
        let is_test_file = path_string.contains("/tests/")
            || path_string.contains("/benches/")
            || path_string.contains("/fuzz/fuzz_targets/")
            || path_string.ends_with("/tests.rs")
            || path_string.ends_with("_tests.rs")
            || path_string.ends_with("_test.rs")
            || path_string.contains("_tests_")
            || path_string.contains("_test_")
            || has_test_entrypoint(&text);
        if !is_test_file {
            continue;
        }
        *test_files += 1;
        let lines = text.lines().count();
        let file_layers = classify_file_layers(&path_string, &text);
        for layer in &file_layers {
            layers.insert(*layer);
        }
        let oversized = lines > OVERSIZED_TEST_THRESHOLD_LINES;
        let recommended_split = recommended_split(&path_string, &file_layers, lines);
        let god_test_candidate = oversized || path_string.ends_with("/tests.rs");
        if oversized {
            oversized_files.push(OversizedFile {
                path: path_string.clone(),
                lines,
                lines_over_threshold: lines - OVERSIZED_TEST_THRESHOLD_LINES,
                recommended_split: recommended_split.clone(),
                release_blocker: true,
            });
        }
        file_records.push(TestFileRecord {
            path: path_string,
            layers: file_layers.into_iter().map(String::from).collect(),
            lines,
            has_test_entrypoint: has_test_entrypoint(&text),
            assertion_count: assertion_count(&text),
            oversized,
            god_test_candidate,
            recommended_split,
        });
    }
}

fn recommended_split(path: &str, layers: &BTreeSet<&'static str>, lines: usize) -> Vec<String> {
    if lines <= OVERSIZED_TEST_THRESHOLD_LINES && !path.ends_with("/tests.rs") {
        return Vec::new();
    }
    let mut splits = Vec::new();
    if path.ends_with("/tests.rs") {
        splits.push(
            "move monolithic src/tests.rs coverage into focused tests/<domain>/ files".to_string(),
        );
    }
    for layer in layers {
        match *layer {
            "property" => {
                splits.push("extract property invariants into tests/properties/".to_string())
            }
            "adversarial" => {
                splits.push("extract hostile-input cases into tests/adversarial/".to_string())
            }
            "corpus" => splits.push("extract fixture-driven cases into tests/corpus/".to_string()),
            "conformance" => {
                splits.push("extract backend/op parity cases into tests/conformance/".to_string())
            }
            "benchmark" => splits.push(
                "move timing-only checks into benches/ or release benchmark cases".to_string(),
            ),
            "gap" => splits.push("extract expected-failure coverage into tests/gap/".to_string()),
            _ => {}
        }
    }
    if splits.is_empty() {
        splits
            .push("split by API contract into tests/contracts/ and tests/regression/".to_string());
    }
    splits.sort();
    splits.dedup();
    splits
}

fn has_test_entrypoint(text: &str) -> bool {
    text.contains("#[test]")
        || text.contains("#[tokio::test]")
        || text.contains("proptest!")
        || text.contains("criterion_group!")
        || text.contains("fuzz_target!")
        || text.contains("#[bench]")
}

fn assertion_count(text: &str) -> usize {
    [
        "assert!(",
        "assert_eq!(",
        "assert_ne!(",
        "prop_assert!(",
        "prop_assert_eq!(",
    ]
    .iter()
    .map(|needle| text.matches(needle).count())
    .sum()
}

fn classify_file_layers(path: &str, text: &str) -> BTreeSet<&'static str> {
    let mut layers = BTreeSet::new();
    let lowered = text.to_ascii_lowercase();
    layers.insert("unit");
    if path.contains("/tests/") {
        layers.insert("integration");
    }
    if path.contains("/benches/")
        || path.contains("bench")
        || path.contains("perf")
        || lowered.contains("criterion_group!")
        || lowered.contains("#[bench]")
        || lowered.contains("benchmark")
    {
        layers.insert("benchmark");
    }
    if path.contains("property") || path.contains("proptest") || lowered.contains("proptest!") {
        layers.insert("property");
    }
    if path.contains("adversarial")
        || path.contains("malformed")
        || path.contains("hostile")
        || lowered.contains("hostile")
        || lowered.contains("malformed")
        || lowered.contains("fail closed")
    {
        layers.insert("adversarial");
    }
    if path.contains("corpus")
        || path.contains("linux")
        || lowered.contains("corpus")
        || lowered.contains("linux subsystem")
    {
        layers.insert("corpus");
    }
    if path.contains("conform")
        || path.contains("parity")
        || path.contains("cross_backend")
        || lowered.contains("conformance")
        || lowered.contains("parity")
        || lowered.contains("frontend api handoff")
    {
        layers.insert("conformance");
    }
    if path.contains("gap")
        || path.contains("blocker")
        || lowered.contains("gap contract")
        || lowered.contains("missing source")
    {
        layers.insert("gap");
    }
    if path.contains("fuzz") || lowered.contains("fuzz") || lowered.contains("hostile_arg") {
        layers.insert("fuzz");
    }
    layers
}

fn parse_output(args: &[String]) -> Result<PathBuf, String> {
    let mut output = None;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--output" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("Fix: --output requires a path.".to_string());
                };
                output = Some(PathBuf::from(path));
                index += 2;
            }
            "--help" | "-h" => {
                println!(
                    "USAGE:\n  cargo_full run --bin xtask -- test-matrix [--output PATH]\n\n\
                     Writes Vyre/Weir test architecture evidence."
                );
                std::process::exit(0);
            }
            other => return Err(format!("Fix: unknown test-matrix option `{other}`.")),
        }
    }
    Ok(output.unwrap_or_else(default_output))
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/evidence/tests/test-matrix.json"))
        .unwrap_or_else(|| PathBuf::from("release/evidence/tests/test-matrix.json"))
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_TEST_SOURCE_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_TEST_SOURCE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_TEST_SOURCE_BYTES} byte release test-source read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}

