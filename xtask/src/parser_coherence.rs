//! Distributed C parser ownership evidence.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::Serialize;

#[derive(Debug, Serialize)]
struct ParserCoherence {
    schema_version: u32,
    components: Vec<ParserComponent>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ParserComponent {
    id: &'static str,
    role: &'static str,
    #[serde(flatten)]
    location: ArtifactLocation,
    exists: bool,
    required_files: Vec<ComponentFile>,
    required_terms: Vec<&'static str>,
    missing_terms: Vec<&'static str>,
    required_contract_topics: Vec<&'static str>,
    missing_contract_topics: Vec<&'static str>,
    required_test_categories: Vec<&'static str>,
    missing_test_categories: Vec<&'static str>,
    required_evidence_trees: Vec<ComponentEvidenceTree>,
    unresolved_ownership_markers: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct ComponentFile {
    #[serde(flatten)]
    location: ArtifactLocation,
    exists: bool,
    read_error: Option<String>,
    source_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactLocation {
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_path: Option<String>,
}

impl ArtifactLocation {
    fn classify(path: &Path, exists: bool) -> Self {
        let rendered = path.display().to_string();
        if exists {
            Self {
                path: Some(rendered),
                expected_path: None,
            }
        } else {
            Self {
                path: None,
                expected_path: Some(rendered),
            }
        }
    }

    fn display(&self) -> &str {
        self.path
            .as_deref()
            .or(self.expected_path.as_deref())
            .expect("ArtifactLocation always owns one rendered path")
    }
}

#[derive(Debug, Serialize)]
struct ComponentContract {
    schema_version: u32,
    component_id: String,
    role: String,
    root: String,
    required_files: Vec<ComponentFile>,
    required_terms: Vec<&'static str>,
    missing_terms: Vec<&'static str>,
    required_contract_topics: Vec<&'static str>,
    missing_contract_topics: Vec<&'static str>,
    required_test_categories: Vec<&'static str>,
    missing_test_categories: Vec<&'static str>,
    required_evidence_trees: Vec<ComponentEvidenceTree>,
    unresolved_ownership_markers: Vec<&'static str>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ComponentEvidenceTree {
    tree: &'static str,
    #[serde(flatten)]
    location: ArtifactLocation,
    exists: bool,
    source_bytes: usize,
    unreadable_file_count: usize,
}

type ComponentSpec = (
    &'static str,
    &'static str,
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
    &'static [&'static str],
    &'static [&'static str],
);

const COMPONENTS: &[ComponentSpec] = &[
    (
        "vyre-frontend-c",
        "Core GPU-first C frontend pipeline, parser contracts, object container, C fixture tests",
        "libs/performance/matching/vyre/vyre-frontend-c",
        &["Cargo.toml", "src/lib.rs", "README.md"],
        &["parser", "compile", "object"],
        &[
            "syntax",
            "ast",
            "diagnostic",
            "span",
            "preprocessor",
            "gnu",
            "unsupported",
        ],
        REQUIRED_PARSER_TEST_CATEGORIES,
    ),
    (
        "vyrec",
        "CLI/compiler user workflow over vyre-frontend-c",
        "tools/vyrec",
        &[
            "Cargo.toml",
            "src/main.rs",
            "README.md",
            "tests/cli_contracts.rs",
            "tests/adversarial_cli_contracts.rs",
            "tests/property_cli_contracts.rs",
            "tests/corpus_linux_contracts.rs",
            "tests/benchmark_cli_contracts.rs",
            "tests/conformance_cli_contracts.rs",
            "tests/gap_cli_contracts.rs",
            "tests/fuzz_cli_contracts.rs",
        ],
        &["vyre", "compile", "cli", "evidence"],
        &[
            "cli",
            "diagnostic",
            "include",
            "macro",
            "corpus",
            "cuda",
            "fuzz",
            "gap",
            "conformance",
            "fix:",
        ],
        REQUIRED_PARSER_TEST_CATEGORIES,
    ),
    (
        "external-dataflow",
        "Dataflow facts consumed by parser/compiler optimization and downstream analysis",
        "libs/dataflow/weir",
        &["Cargo.toml", "src/lib.rs", "README.md"],
        &["dataflow", "analysis", "program"],
        &["parser", "dataflow", "alias", "reaching", "callgraph"],
        REQUIRED_PARSER_TEST_CATEGORIES,
    ),
    (
        "compiler-consumer-grammar-gen",
        "Shared grammar generation substrate",
        "libs/performance/matching/vyre/vyre-grammar-gen",
        &["Cargo.toml", "src/lib.rs", "README.md"],
        &["grammar", "generate"],
        &["grammar", "generate", "token", "parser"],
        REQUIRED_PARSER_TEST_CATEGORIES,
    ),
];

/// Ids of every parser-ownership component, in declaration order.
///
/// Release completion and artifact registries derive their component lists
/// from this function so a removed or renamed surface cannot remain required
/// by a second hardcoded inventory.
pub(crate) fn component_ids() -> Vec<&'static str> {
    COMPONENTS.iter().map(|component| component.0).collect()
}

const REQUIRED_PARSER_TEST_CATEGORIES: &[&str] = &[
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

const REQUIRED_PARSER_EVIDENCE_TREES: &[&str] = &["tests", "benches", "fuzz"];

const MAX_PARSER_CONTRACT_FILE_BYTES: u64 = 2_097_152;
const UNRESOLVED_OWNERSHIP_MARKERS: &[&str] = &[
    "owner: tbd",
    "ownership: tbd",
    "owner: unknown",
    "ownership: unknown",
    "unresolved ownership",
    "placeholder",
    "todo",
    "fixme",
];

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
    let mut components = Vec::new();
    let mut blockers = Vec::new();
    for &(
        id,
        role,
        relative,
        required_files,
        required_terms,
        required_contract_topics,
        required_test_categories,
    ) in COMPONENTS
    {
        let path = santh_root.join(relative);
        let exists = path.exists();
        if !exists {
            blockers.push(format!(
                "parser component `{id}` is missing at {}",
                path.display()
            ));
        }
        let mut component_text = String::new();
        let mut ownership_text = String::new();
        let required_files = required_files
            .iter()
            .map(|required| {
                let file_path = path.join(required);
                let exists = file_path.is_file();
                let (text, read_error) = if exists {
                    match read_text_bounded(&file_path) {
                        Ok(text) => (text, None),
                        Err(error) => {
                            blockers.push(format!(
                                "parser component `{id}` required file {} could not be read: {error}",
                                file_path.display()
                            ));
                            (String::new(), Some(error.to_string()))
                        }
                    }
                } else {
                    (String::new(), None)
                };
                component_text.push_str(&text);
                ownership_text.push_str(&text);
                if !exists {
                    blockers.push(format!(
                        "parser component `{id}` is missing required file {}",
                        file_path.display()
                    ));
                } else if text.trim().is_empty() {
                    blockers.push(format!(
                        "parser component `{id}` required file {} is empty",
                        file_path.display()
                    ));
                }
                ComponentFile {
                    location: ArtifactLocation::classify(&file_path, exists),
                    exists,
                    read_error,
                    source_bytes: text.len(),
                }
            })
            .collect();
        let component_test_unreadable = append_component_test_text(&path, &mut component_text);
        if component_test_unreadable != 0 {
            blockers.push(format!(
                "parser component `{id}` test/bench/fuzz evidence has {component_test_unreadable} unreadable source file(s)"
            ));
        }
        let lowered = component_text.to_ascii_lowercase();
        let missing_terms = required_terms
            .iter()
            .copied()
            .filter(|term| !lowered.contains(term))
            .collect::<Vec<_>>();
        for term in &missing_terms {
            blockers.push(format!(
                "parser component `{id}` does not document or expose required term `{term}`"
            ));
        }
        let missing_contract_topics = required_contract_topics
            .iter()
            .copied()
            .filter(|topic| !lowered.contains(topic))
            .collect::<Vec<_>>();
        for topic in &missing_contract_topics {
            blockers.push(format!(
                "parser component `{id}` does not document parser contract topic `{topic}`"
            ));
        }
        let missing_test_categories = required_test_categories
            .iter()
            .copied()
            .filter(|category| !lowered.contains(category))
            .collect::<Vec<_>>();
        for category in &missing_test_categories {
            blockers.push(format!(
                "parser component `{id}` does not expose required test category `{category}`"
            ));
        }
        let required_evidence_trees = REQUIRED_PARSER_EVIDENCE_TREES
            .iter()
            .map(|tree| {
                let tree = *tree;
                let tree_path = path.join(tree);
                let exists = tree_path.is_dir();
                let (source_bytes, unreadable_file_count) = tree_source_bytes(&tree_path);
                if !exists {
                    blockers.push(format!(
                        "parser component `{id}` is missing required `{tree}` evidence tree"
                    ));
                } else if unreadable_file_count != 0 {
                    blockers.push(format!(
                        "parser component `{id}` required `{tree}` evidence tree has {unreadable_file_count} unreadable source file(s)"
                    ));
                } else if source_bytes == 0 {
                    blockers.push(format!(
                        "parser component `{id}` required `{tree}` evidence tree is empty"
                    ));
                }
                ComponentEvidenceTree {
                    tree,
                    location: ArtifactLocation::classify(&tree_path, exists),
                    exists,
                    source_bytes,
                    unreadable_file_count,
                }
            })
            .collect::<Vec<_>>();
        let lowered_ownership = normalized_ownership_text(&ownership_text);
        let unresolved_ownership_markers = UNRESOLVED_OWNERSHIP_MARKERS
            .iter()
            .copied()
            .filter(|marker| lowered_ownership.contains(marker))
            .collect::<Vec<_>>();
        for marker in &unresolved_ownership_markers {
            blockers.push(format!(
                "parser component `{id}` contains unresolved ownership marker `{marker}`"
            ));
        }
        components.push(ParserComponent {
            id,
            role,
            location: ArtifactLocation::classify(&path, exists),
            exists,
            required_files,
            required_terms: required_terms.to_vec(),
            missing_terms,
            required_contract_topics: required_contract_topics.to_vec(),
            missing_contract_topics,
            required_test_categories: required_test_categories.to_vec(),
            missing_test_categories,
            required_evidence_trees,
            unresolved_ownership_markers,
        });
    }
    let matrix = ParserCoherence {
        schema_version: 1,
        components,
        blockers,
    };

    if let Some(parent) = output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    crate::output_arg::write_json(&output, &matrix);
    write_sibling_contracts(&output, &matrix);
    println!("parser-coherence: wrote {}", output.display());
    if !matrix.blockers.is_empty() {
        std::process::exit(1);
    }
}

fn write_sibling_contracts(output: &Path, matrix: &ParserCoherence) {
    let Some(parent) = output.parent() else {
        eprintln!(
            "Fix: parser coherence output `{}` has no parent directory.",
            output.display()
        );
        std::process::exit(1);
    };
    for component in &matrix.components {
        let blockers = component
            .required_files
            .iter()
            .filter(|file| !file.exists)
            .map(|file| {
                format!(
                    "parser component `{}` is missing required file {}",
                    component.id,
                    file.location.display()
                )
            })
            .chain((!component.exists).then(|| {
                format!(
                    "parser component `{}` is missing at {}",
                    component.id,
                    component.location.display()
                )
            }))
            .chain(component.required_files.iter().filter(|file| file.source_bytes == 0).map(
                |file| {
                    format!(
                        "parser component `{}` required file {} is empty",
                        component.id,
                        file.location.display()
                    )
                },
            ))
            .chain(component.missing_terms.iter().map(|term| {
                format!(
                    "parser component `{}` is missing required term `{term}`",
                    component.id
                )
            }))
            .chain(component.missing_contract_topics.iter().map(|topic| {
                format!(
                    "parser component `{}` is missing parser contract topic `{topic}`",
                    component.id
                )
            }))
            .chain(component.missing_test_categories.iter().map(|category| {
                format!(
                    "parser component `{}` is missing test category `{category}`",
                    component.id
                )
            }))
            .chain(component.required_evidence_trees.iter().filter(|tree| !tree.exists).map(
                |tree| {
                    format!(
                        "parser component `{}` is missing required `{}` evidence tree {}",
                        component.id,
                        tree.tree,
                        tree.location.display()
                    )
                },
            ))
            .chain(component.required_evidence_trees.iter().filter(|tree| tree.unreadable_file_count != 0).map(
                |tree| {
                    format!(
                        "parser component `{}` required `{}` evidence tree {} has {} unreadable source file(s)",
                        component.id,
                        tree.tree,
                        tree.location.display(),
                        tree.unreadable_file_count
                    )
                },
            ))
            .chain(component.required_evidence_trees.iter().filter(|tree| tree.source_bytes == 0).map(
                |tree| {
                    format!(
                        "parser component `{}` required `{}` evidence tree {} is empty",
                        component.id,
                        tree.tree,
                        tree.location.display()
                    )
                },
            ))
            .chain(component.unresolved_ownership_markers.iter().map(|marker| {
                format!(
                    "parser component `{}` contains unresolved ownership marker `{marker}`",
                    component.id
                )
            }))
            .collect::<Vec<_>>();
        write_json(
            &parent.join(component_contract_artifact(component.id)),
            &ComponentContract {
                schema_version: 1,
                component_id: component.id.to_string(),
                role: component.role.to_string(),
                root: component.location.display().to_string(),
                required_files: component.required_files.clone(),
                required_terms: component.required_terms.clone(),
                missing_terms: component.missing_terms.clone(),
                required_contract_topics: component.required_contract_topics.clone(),
                missing_contract_topics: component.missing_contract_topics.clone(),
                required_test_categories: component.required_test_categories.clone(),
                missing_test_categories: component.missing_test_categories.clone(),
                required_evidence_trees: component.required_evidence_trees.clone(),
                unresolved_ownership_markers: component.unresolved_ownership_markers.clone(),
                blockers,
            },
        );
    }
}

/// File name of the per-component contract artifact for `component_id`.
///
/// The single owner of the id-to-file-name mapping. The release gate used to
/// re-derive the component id from the file name by stripping `-contracts.json`
/// and special-casing `vyrec`, which is the same rule written twice: when the
/// dataflow component id and its artifact name diverged, the gate reported
/// `component_id `weir`, expected `external-dataflow`` for an artifact the
/// generator had produced correctly.
pub(crate) fn component_contract_artifact(component_id: &str) -> String {
    match component_id {
        "vyrec" => "vyrec-cli-contracts.json".to_string(),
        other => format!("{other}-contracts.json"),
    }
}

/// Component id that owns the contract artifact named `artifact`.
///
/// Inverse of [`component_contract_artifact`], resolved against the component
/// table rather than by string surgery, so a component whose artifact name is
/// not simply `<id>-contracts.json` still resolves.
pub(crate) fn component_id_for_contract_artifact(artifact: &str) -> Option<&'static str> {
    component_ids()
        .into_iter()
        .find(|id| component_contract_artifact(id) == artifact)
}

/// Contract artifact file names for every parser-ownership component, in
/// declaration order.
pub(crate) fn component_contract_artifacts() -> Vec<String> {
    component_ids()
        .into_iter()
        .map(component_contract_artifact)
        .collect()
}

fn write_json(path: &Path, value: &impl Serialize) {
    crate::output_arg::write_json(path, value);
}

fn append_component_test_text(root: &Path, output: &mut String) -> usize {
    let mut unreadable = 0usize;
    for relative in ["tests", "benches", "fuzz"] {
        unreadable = unreadable.saturating_add(append_tree_text(&root.join(relative), output));
    }
    unreadable
}

/// Lowercase the ownership text with lint configuration removed.
///
/// A lint that DENIES `todo!()` is the opposite of an unresolved ownership marker, so it
/// must not read as one. This handled the attribute spelling (`clippy::todo`) but not the
/// manifest spelling: `todo = "deny"` under `[lints.clippy]` in `tools/vyrec/Cargo.toml`
/// made the `vyrec` parser component permanently report an unresolved `todo` marker.
/// Dropping whole lint-level lines covers both spellings and any future lint named after a
/// marker word.
fn normalized_ownership_text(text: &str) -> String {
    text.to_ascii_lowercase()
        .lines()
        .filter(|line| !is_lint_level_line(line))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("clippy::todo", "clippy-lint")
}

/// True when `line` sets a lint level, in either the manifest or attribute spelling.
fn is_lint_level_line(line: &str) -> bool {
    const LINT_LEVELS: &[&str] = &["deny", "forbid", "warn", "allow"];
    let trimmed = line.trim();
    if let Some((_name, value)) = trimmed.split_once('=') {
        let value = value.trim().trim_end_matches(',').trim_matches('"');
        if LINT_LEVELS.contains(&value) {
            return true;
        }
    }
    trimmed.starts_with("#![") || trimmed.starts_with("#[")
}

fn tree_source_bytes(root: &Path) -> (usize, usize) {
    let mut text = String::new();
    let unreadable = append_tree_text(root, &mut text);
    (text.len(), unreadable)
}

fn append_tree_text(root: &Path, output: &mut String) -> usize {
    let Ok(entries) = fs::read_dir(root) else {
        return usize::from(root.exists());
    };
    let mut unreadable = 0usize;
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => {
                unreadable = unreadable.saturating_add(1);
                continue;
            }
        };
        let path = entry.path();
        if path.is_dir() {
            unreadable = unreadable.saturating_add(append_tree_text(&path, output));
            continue;
        }
        let extension = path.extension().and_then(|extension| extension.to_str());
        if !matches!(extension, Some("rs" | "toml" | "md" | "c" | "h")) {
            continue;
        }
        if let Ok(text) = read_text_bounded(&path) {
            output.push('\n');
            output.push_str(&text);
        } else {
            unreadable = unreadable.saturating_add(1);
        }
    }
    unreadable
}

fn parse_output(args: &[String]) -> Result<PathBuf, String> {
    crate::output_arg::parse_output_arg(
        args,
        "parser-coherence",
        "Writes distributed C parser ownership evidence.",
        default_output,
    )
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/evidence/parser/distributed-parser-map.json"))
        .unwrap_or_else(|| PathBuf::from("release/evidence/parser/distributed-parser-map.json"))
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_PARSER_CONTRACT_FILE_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_PARSER_CONTRACT_FILE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_PARSER_CONTRACT_FILE_BYTES} byte parser contract read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lint that denies `todo!()` must not read as an unresolved ownership marker.
    ///
    /// `tools/vyrec/Cargo.toml` sets `todo = "deny"` under `[lints.clippy]`, the strongest
    /// possible statement that todos are banned. The scan matched the word anyway, so the
    /// `vyrec` parser component reported a permanent unresolved-ownership blocker that no
    /// amount of real work could clear.
    #[test]
    fn manifest_lint_levels_are_not_unresolved_ownership_markers() {
        let manifest = "[lints.clippy]\ntodo = \"deny\"\nunimplemented = \"deny\"\n";
        let normalized = normalized_ownership_text(manifest);

        for marker in UNRESOLVED_OWNERSHIP_MARKERS {
            assert!(
                !normalized.contains(marker),
                "Fix: a `[lints.clippy]` level must not count as unresolved ownership marker `{marker}`; normalized={normalized:?}"
            );
        }
    }

    /// The attribute spelling of the same lint is also not a marker.
    #[test]
    fn attribute_lint_levels_are_not_unresolved_ownership_markers() {
        let source = "#![deny(clippy::todo)]\n#[allow(clippy::fixme)]\nfn main() {}\n";
        let normalized = normalized_ownership_text(source);

        for marker in UNRESOLVED_OWNERSHIP_MARKERS {
            assert!(
                !normalized.contains(marker),
                "Fix: a lint attribute must not count as unresolved ownership marker `{marker}`; normalized={normalized:?}"
            );
        }
    }

    /// A real unresolved marker in prose is still reported.
    ///
    /// The filter above must narrow the scan to lint configuration only. If it swallowed
    /// prose too, a component could ship with `Owner: TBD` in its README and pass.
    #[test]
    fn prose_ownership_markers_are_still_reported() {
        let readme = "# vyrec\n\nOwner: TBD\n\nTODO: wire the parser.\n";
        let normalized = normalized_ownership_text(readme);

        assert!(
            normalized.contains("owner: tbd"),
            "Fix: an unresolved owner in prose must still be reported; normalized={normalized:?}"
        );
        assert!(
            normalized.contains("todo"),
            "Fix: a TODO in prose must still be reported; normalized={normalized:?}"
        );
    }

    /// Every component's contract artifact resolves back to that component.
    ///
    /// The release gate looks a component up by artifact file name. Before the
    /// mapping had one owner it stripped `-contracts.json` instead, so the
    /// dataflow component, whose artifact is `external-dataflow-contracts.json`,
    /// resolved to a component id that did not exist and the gate reported a
    /// permanent `expected` mismatch on an artifact that was in fact correct.
    #[test]
    fn every_component_contract_artifact_round_trips_to_its_component_id() {
        for id in component_ids() {
            let artifact = component_contract_artifact(id);
            assert_eq!(
                component_id_for_contract_artifact(&artifact),
                Some(id),
                "Fix: artifact `{artifact}` must resolve back to component `{id}`."
            );
        }
    }

    /// The parser release inventory contains only components owned by this
    /// release train. A removed sibling consumer must not keep publication
    /// blocked through a synthetic parser contract.
    #[test]
    fn release_parser_components_match_live_owned_surfaces() {
        assert_eq!(
            component_ids(),
            vec![
                "vyre-frontend-c",
                "vyrec",
                "external-dataflow",
                "compiler-consumer-grammar-gen",
            ]
        );
        assert_eq!(
            component_id_for_contract_artifact("compiler-consumer-contracts.json"),
            None
        );
    }

    /// An artifact no component owns resolves to nothing rather than to a
    /// plausible-looking id derived from its name.
    #[test]
    fn an_unowned_contract_artifact_resolves_to_no_component() {
        assert_eq!(
            component_id_for_contract_artifact("weir-contracts.json"),
            None,
            "Fix: `weir-contracts.json` is the pre-0.7.0 name of the dataflow \
             component's artifact and no component owns it now."
        );
        assert_eq!(component_id_for_contract_artifact("contracts.json"), None);
    }

    /// The component ids are neutral capability names, not sibling crate names.
    ///
    /// The artifact file name derives from the id, and every expected-artifact
    /// list in the release gates names `external-dataflow-contracts.json`. An id
    /// carrying a crate name would silently rename the artifact and break those
    /// lists.
    #[test]
    fn component_ids_are_neutral_capability_names() {
        for id in component_ids() {
            for crate_name in ["weir", "surgec", "gossan", "keyhog"] {
                assert_ne!(
                    id, crate_name,
                    "Fix: component id `{id}` must be a neutral capability name."
                );
            }
        }
        assert!(
            component_contract_artifacts()
                .contains(&"external-dataflow-contracts.json".to_string()),
            "Fix: the dataflow component must own `external-dataflow-contracts.json`."
        );
    }

    /// Missing prerequisites remain explicit without masquerading as live path citations.
    #[test]
    fn missing_artifact_locations_serialize_as_expected_paths() {
        let location = ArtifactLocation::classify(Path::new("/missing/compiler/Cargo.toml"), false);
        let value = serde_json::to_value(location)
            .expect("Fix: an expected parser component path must serialize");

        assert_eq!(value.get("path"), None);
        assert_eq!(
            value
                .get("expected_path")
                .and_then(serde_json::Value::as_str),
            Some("/missing/compiler/Cargo.toml")
        );
    }

    /// Existing evidence remains a live `path` citation that the filesystem gate verifies.
    #[test]
    fn existing_artifact_locations_serialize_as_live_paths() {
        let location = ArtifactLocation::classify(Path::new("/present/parser/src/lib.rs"), true);
        let value = serde_json::to_value(location)
            .expect("Fix: a live parser component path must serialize");

        assert_eq!(
            value.get("path").and_then(serde_json::Value::as_str),
            Some("/present/parser/src/lib.rs")
        );
        assert_eq!(value.get("expected_path"), None);
    }
}
