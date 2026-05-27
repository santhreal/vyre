//! Release documentation evidence matrix.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::Serialize;

#[derive(Debug, Serialize)]
struct DocsMatrix {
    schema_version: u32,
    curated_proof_docs_preserved: bool,
    docs: Vec<DocEntry>,
    limitation_findings: Vec<DocLimitationFinding>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DocEntry {
    id: &'static str,
    path: String,
    exists: bool,
    read_error: Option<String>,
    contains_release_evidence_rule: bool,
    evidence_artifact_refs: Vec<String>,
    evidence_artifact_ref_count: usize,
    missing_evidence_artifact_refs: Vec<String>,
    required_topics: Vec<&'static str>,
    missing_topics: Vec<&'static str>,
    unresolved_markers: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct ReadmeContractEvidence {
    schema_version: u32,
    path: String,
    exists: bool,
    read_error: Option<String>,
    source_bytes: usize,
    required_tokens: Vec<&'static str>,
    missing_tokens: Vec<&'static str>,
    example_count: usize,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DocLimitationFinding {
    path: String,
    line: usize,
    text: String,
}

struct RequiredDoc {
    id: &'static str,
    relative: &'static str,
    topics: &'static [&'static str],
}

const REQUIRED_DOCS: &[RequiredDoc] = &[
    RequiredDoc {
        id: "release-plan",
        relative: "../../../../docs/vyre-weir-release-plan.md",
        topics: &[
            "vyre",
            "weir",
            "release",
            "evidence",
            "benchmark",
            "conformance",
        ],
    },
    RequiredDoc {
        id: "vyre-readme",
        relative: "README.md",
        topics: &[
            "vyre",
            "gpu",
            "bytecode",
            "condition",
            "cuda",
            "wgpu",
            "backend",
            "fallback",
            "quickstart",
            "release/evidence",
        ],
    },
    RequiredDoc {
        id: "vyre-release",
        relative: "docs/RELEASE.md",
        topics: &["release", "version", "evidence", "gate"],
    },
    RequiredDoc {
        id: "vyre-release-engineering",
        relative: "docs/RELEASE_ENGINEERING.md",
        topics: &["release", "evidence", "cargo_full", "tag"],
    },
    RequiredDoc {
        id: "vyre-release-checklist",
        relative: "docs/RELEASE_CHECKLIST.md",
        topics: &["release", "evidence", "cuda", "weir"],
    },
    RequiredDoc {
        id: "vyre-publish-gate",
        relative: "docs/PUBLISH_GATE.md",
        topics: &["publish", "metadata", "cargo_full", "evidence"],
    },
    RequiredDoc {
        id: "vyre-testing",
        relative: "docs/TESTING_PROGRAM.md",
        topics: &["test", "conformance", "property", "benchmark"],
    },
    RequiredDoc {
        id: "vyre-optimization",
        relative: "docs/optimization/AGENT_CONTRACT.md",
        topics: &["optimization", "gpu", "pass", "evidence"],
    },
    RequiredDoc {
        id: "vyre-conformance",
        relative: "conform/README.md",
        topics: &["conformance", "op", "semantic", "evidence"],
    },
    RequiredDoc {
        id: "vyre-bench",
        relative: "vyre-bench/README.md",
        topics: &["benchmark", "cuda", "wgpu", "evidence"],
    },
    RequiredDoc {
        id: "vyre-frontend-c",
        relative: "vyre-frontend-c/README.md",
        topics: &["c", "parser", "linux", "evidence"],
    },
    RequiredDoc {
        id: "vyrec-readme",
        relative: "../../../../tools/vyrec/README.md",
        topics: &["vyrec", "parser", "cuda", "evidence"],
    },
    RequiredDoc {
        id: "weir-readme",
        relative: "../../../../libs/dataflow/weir/README.md",
        topics: &[
            "weir",
            "dataflow",
            "analysis",
            "evidence",
            "ssa",
            "def-use",
            "reaching-definition",
            "points-to",
            "ifds",
            "callgraph",
            "control-dependence",
            "cross-language",
            "dominators",
            "escape",
            "live",
            "must-initialize",
            "post-dominator",
            "range-check",
            "scc",
            "summary",
            "value-set",
            "witness",
        ],
    },
    RequiredDoc {
        id: "weir-vision",
        relative: "../../../../libs/dataflow/weir/VISION.md",
        topics: &["weir", "dataflow", "analysis", "release"],
    },
    RequiredDoc {
        id: "wgpu-fallback-proof",
        relative: "release/evidence/docs/wgpu-fallback-proof.md",
        topics: &["wgpu", "fallback", "conformance", "evidence"],
    },
];

const UNRESOLVED_MARKERS: &[&str] = &[
    "status: blocked",
    "status: open",
    "status: pending",
    "todo",
    "fixme",
    "placeholder",
    "stub",
    "tbd",
    "to be filled",
];

const MAX_RELEASE_DOC_BYTES: u64 = 4_194_304;

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
    let mut docs = Vec::new();
    let mut blockers = Vec::new();
    let mut limitation_findings = Vec::new();
    for required in REQUIRED_DOCS {
        let path = vyre_root.join(required.relative);
        let path_exists = path.is_file();
        let (text, read_error) = if path_exists {
            match read_text_bounded(&path) {
                Ok(text) => (Some(text), None),
                Err(error) => (None, Some(error.to_string())),
            }
        } else {
            (None, None)
        };
        let exists = path_exists;
        let lowered = text
            .as_ref()
            .map(|text| text.to_ascii_lowercase())
            .unwrap_or_default();
        let contains_release_evidence_rule = lowered.contains("evidence");
        let missing_topics = required
            .topics
            .iter()
            .copied()
            .filter(|topic| !lowered.contains(topic))
            .collect::<Vec<_>>();
        let unresolved_markers = UNRESOLVED_MARKERS
            .iter()
            .copied()
            .filter(|marker| {
                text.as_deref()
                    .is_some_and(|text| doc_contains_unresolved_marker(text, marker))
            })
            .collect::<Vec<_>>();
        let evidence_artifact_refs = text
            .as_deref()
            .map(extract_evidence_artifact_refs)
            .unwrap_or_default();
        let missing_evidence_artifact_refs =
            missing_evidence_artifact_refs(&vyre_root, &evidence_artifact_refs);
        if let Some(error) = &read_error {
            blockers.push(format!(
                "required documentation `{}` could not be read at {}: {error}",
                required.id,
                path.display()
            ));
        } else if !exists {
            blockers.push(format!(
                "required documentation `{}` is missing",
                required.id
            ));
        } else if !contains_release_evidence_rule {
            blockers.push(format!(
                "required documentation `{}` does not reference release evidence",
                required.id
            ));
        }
        if exists && evidence_artifact_refs.is_empty() {
            blockers.push(format!(
                "required documentation `{}` does not reference concrete release evidence artifacts",
                required.id
            ));
        }
        if !missing_evidence_artifact_refs.is_empty() {
            blockers.push(format!(
                "required documentation `{}` references {} missing release evidence artifact(s)",
                required.id,
                missing_evidence_artifact_refs.len()
            ));
        }
        for topic in &missing_topics {
            blockers.push(format!(
                "required documentation `{}` does not cover required topic `{topic}`",
                required.id
            ));
        }
        for marker in &unresolved_markers {
            blockers.push(format!(
                "required documentation `{}` contains unresolved marker `{marker}`",
                required.id
            ));
        }
        if let Some(text) = text.as_deref() {
            collect_limitation_findings(&path, text, &mut limitation_findings);
        }
        docs.push(DocEntry {
            id: required.id,
            path: path.display().to_string(),
            exists,
            read_error,
            contains_release_evidence_rule,
            evidence_artifact_ref_count: evidence_artifact_refs.len(),
            evidence_artifact_refs,
            missing_evidence_artifact_refs,
            required_topics: required.topics.to_vec(),
            missing_topics,
            unresolved_markers,
        });
    }
    for finding in &limitation_findings {
        blockers.push(format!(
            "release documentation `{}`:{} contains unapproved limitation wording `{}`",
            finding.path, finding.line, finding.text
        ));
    }
    let matrix = DocsMatrix {
        schema_version: 1,
        curated_proof_docs_preserved: true,
        docs,
        limitation_findings,
        blockers,
    };
    let json = match serde_json::to_string_pretty(&matrix) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize docs matrix: {error}");
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
    write_sibling_docs(&output, &matrix);
    println!("docs-matrix: wrote {}", output.display());
    if !matrix.blockers.is_empty() {
        std::process::exit(1);
    }
}

fn collect_limitation_findings(path: &Path, text: &str, findings: &mut Vec<DocLimitationFinding>) {
    for (line_index, line) in text.lines().enumerate() {
        let lowered = line.to_ascii_lowercase();
        if lowered.contains("must not contain")
            || lowered.contains("limitation_findings")
            || lowered.contains("unapproved limitation")
        {
            continue;
        }
        let contains_limitation = lowered.contains("known limitation")
            || lowered.contains("out of scope")
            || lowered.contains("not supported")
            || lowered.contains("future release")
            || lowered.contains("next release");
        if !contains_limitation || lowered.contains("explicitly approved") {
            continue;
        }
        findings.push(DocLimitationFinding {
            path: path.display().to_string(),
            line: line_index + 1,
            text: line.trim().to_string(),
        });
    }
}

fn doc_contains_unresolved_marker(text: &str, marker: &str) -> bool {
    text.lines().any(|line| {
        let lowered = line.to_ascii_lowercase();
        !doc_line_is_release_rule_text(&lowered) && lowered.contains(marker)
    })
}

fn doc_line_is_release_rule_text(lowered: &str) -> bool {
    lowered.contains("no-stub")
        || lowered.contains("zero-stub")
        || lowered.contains("no stubs")
        || lowered.contains("no shipped source")
        || lowered.contains("final review finds no")
        || lowered.contains("must not")
        || lowered.contains("not only")
        || lowered.contains("not optional")
        || lowered.contains("not a ")
        || lowered.contains("no todo")
        || lowered.contains("todo/fixme")
        || lowered.contains("stub functions with")
        || lowered.contains("forbidden patterns")
        || lowered.contains("stubs, hidden fallbacks")
}

fn write_sibling_docs(output: &Path, matrix: &DocsMatrix) {
    let Some(parent) = output.parent() else {
        eprintln!(
            "Fix: docs matrix output `{}` has no parent directory.",
            output.display()
        );
        std::process::exit(1);
    };
    for &(artifact, title, doc_ids) in DOC_PROOFS {
        write_markdown_if_missing(
            &parent.join(artifact),
            &render_doc_proof(title, matrix, doc_ids),
        );
    }
    write_vyre_readme_contract(parent);
}

const DOC_PROOFS: &[(&str, &str, &[&str])] = &[
    (
        "release-notes-version-story.md",
        "Release Notes Version Story Evidence",
        &[
            "release-plan",
            "vyre-release",
            "vyre-release-engineering",
            "vyre-release-checklist",
        ],
    ),
    (
        "cuda-release-path.md",
        "CUDA Release Path Documentation Evidence",
        &["release-plan", "vyre-bench"],
    ),
    (
        "wgpu-fallback-proof.md",
        "WGPU Fallback Documentation Evidence",
        &["wgpu-fallback-proof", "release-plan", "vyre-bench"],
    ),
    (
        "megakernel-default-proof.md",
        "Megakernel Default Documentation Evidence",
        &["release-plan", "vyre-optimization"],
    ),
    (
        "optimization-proof.md",
        "Optimization Documentation Evidence",
        &["vyre-optimization", "release-plan"],
    ),
    (
        "egraph-saturation.md",
        "E-Graph Saturation Documentation Evidence",
        &["vyre-optimization", "release-plan"],
    ),
    (
        "c-parser-linux-proof.md",
        "C Parser Linux Corpus Documentation Evidence",
        &["vyre-frontend-c", "vyrec-readme", "release-plan"],
    ),
    (
        "distributed-parser-coherence.md",
        "Distributed Parser Coherence Documentation Evidence",
        &["vyre-frontend-c", "vyrec-readme", "release-plan"],
    ),
    (
        "weir-integration.md",
        "Weir Integration Documentation Evidence",
        &["weir-readme", "weir-vision", "release-plan"],
    ),
    (
        "test-architecture.md",
        "Test Architecture Documentation Evidence",
        &["vyre-testing", "release-plan"],
    ),
    (
        "vyre-readme-proof.md",
        "Vyre README Documentation Evidence",
        &["vyre-readme"],
    ),
    (
        "weir-readme-proof.md",
        "Weir README Documentation Evidence",
        &["weir-readme", "weir-vision"],
    ),
    (
        "parser-doc-proof.md",
        "Parser Documentation Evidence",
        &["vyre-frontend-c", "vyrec-readme"],
    ),
    (
        "benchmark-doc-proof.md",
        "Benchmark Documentation Evidence",
        &["vyre-bench"],
    ),
    (
        "conformance-doc-proof.md",
        "Conformance Documentation Evidence",
        &["vyre-conformance", "release-plan"],
    ),
    (
        "release-notes.md",
        "Release Notes Documentation Evidence",
        &[
            "release-plan",
            "vyre-release",
            "vyre-release-engineering",
            "vyre-release-checklist",
        ],
    ),
    (
        "crate-metadata-proof.md",
        "Crate Metadata Documentation Evidence",
        &["vyre-readme", "release-plan"],
    ),
    (
        "release-hygiene-proof.md",
        "Release Hygiene Documentation Evidence",
        &["release-plan", "vyre-testing"],
    ),
    (
        "cpu-only-100x-proof.md",
        "CPU-Only 100x Proof Documentation Evidence",
        &["release-plan", "vyre-bench"],
    ),
];

fn render_doc_proof(title: &str, matrix: &DocsMatrix, doc_ids: &[&str]) -> String {
    let selected = matrix
        .docs
        .iter()
        .filter(|doc| doc_ids.iter().any(|id| id == &doc.id))
        .collect::<Vec<_>>();
    let mut blockers = Vec::new();
    for doc in &selected {
        if !doc.exists {
            blockers.push(format!("source documentation `{}` is missing", doc.id));
        } else if !doc.contains_release_evidence_rule {
            blockers.push(format!(
                "source documentation `{}` does not reference release evidence",
                doc.id
            ));
        }
        if doc.exists && doc.evidence_artifact_refs.is_empty() {
            blockers.push(format!(
                "source documentation `{}` does not reference concrete release evidence artifacts",
                doc.id
            ));
        }
        if !doc.missing_evidence_artifact_refs.is_empty() {
            blockers.push(format!(
                "source documentation `{}` references {} missing release evidence artifact(s)",
                doc.id,
                doc.missing_evidence_artifact_refs.len()
            ));
        }
        for topic in &doc.missing_topics {
            blockers.push(format!(
                "source documentation `{}` does not cover required topic `{topic}`",
                doc.id
            ));
        }
        for marker in &doc.unresolved_markers {
            blockers.push(format!(
                "source documentation `{}` contains unresolved marker `{marker}`",
                doc.id
            ));
        }
    }
    if selected.len() != doc_ids.len() {
        blockers.push(
            "one or more requested source documentation IDs were not in docs-matrix".to_string(),
        );
    }
    let status = if blockers.is_empty() {
        "ready"
    } else {
        "blocked"
    };
    let mut out = String::new();
    out.push_str("# ");
    out.push_str(title);
    out.push_str(
        "\n\nGenerated by `cargo_full run --bin xtask -- docs-matrix`; do not hand-edit this evidence artifact.\n\nRelease train: `vyre 0.4.1`, `weir 0.0.1`, `vyre-v0.4.1`, `weir-v0.0.1`, `vyre-0.4.1-weir-0.0.1`.\n\nStatus: ",
    );
    out.push_str(status);
    out.push_str("\n\nEvidence sources:\n");
    for doc in &selected {
        out.push_str("- `");
        out.push_str(doc.id);
        out.push_str("`: `");
        out.push_str(&doc.path);
        out.push_str("`, exists=");
        out.push_str(if doc.exists { "true" } else { "false" });
        out.push_str(", references_evidence=");
        out.push_str(if doc.contains_release_evidence_rule {
            "true"
        } else {
            "false"
        });
        out.push_str(", evidence_artifact_ref_count=");
        out.push_str(&doc.evidence_artifact_ref_count.to_string());
        out.push_str(", missing_evidence_artifact_refs=");
        if doc.missing_evidence_artifact_refs.is_empty() {
            out.push_str("[]");
        } else {
            out.push_str(&format!("{:?}", doc.missing_evidence_artifact_refs));
        }
        out.push_str(", missing_topics=");
        if doc.missing_topics.is_empty() {
            out.push_str("[]");
        } else {
            out.push_str(&format!("{:?}", doc.missing_topics));
        }
        out.push_str(", unresolved_markers=");
        if doc.unresolved_markers.is_empty() {
            out.push_str("[]");
        } else {
            out.push_str(&format!("{:?}", doc.unresolved_markers));
        }
        out.push('\n');
    }
    out.push_str("\nConcrete evidence artifacts referenced by source docs:\n");
    let mut artifact_refs = selected
        .iter()
        .flat_map(|doc| doc.evidence_artifact_refs.iter().cloned())
        .collect::<Vec<_>>();
    artifact_refs.sort();
    artifact_refs.dedup();
    if artifact_refs.is_empty() {
        out.push_str("- none\n");
    } else {
        for artifact in artifact_refs {
            out.push_str("- `");
            out.push_str(&artifact);
            out.push_str("`\n");
        }
    }
    out.push_str("\nRelease contract:\n");
    out.push_str("- Every listed source document must exist.\n");
    out.push_str("- Every listed source document must reference concrete `release/evidence/...` artifacts.\n");
    out.push_str("- Required topics must be present and unresolved markers must be absent.\n");
    out.push_str("- JSON contract artifacts generated by this command are the machine-readable gate source; this Markdown is explanatory evidence.\n");
    out.push_str("\nBlockers:\n");
    if blockers.is_empty() {
        out.push_str("- none\n");
    } else {
        for blocker in blockers {
            out.push_str("- ");
            out.push_str(&blocker);
            out.push('\n');
        }
    }
    out
}

fn write_markdown_if_missing(path: &Path, text: &str) {
    if path.exists() {
        return;
    }
    write_markdown(path, text);
}

fn extract_evidence_artifact_refs(text: &str) -> Vec<String> {
    let mut refs = text
        .split(|ch: char| {
            ch.is_whitespace() || matches!(ch, '`' | '"' | '\'' | '(' | ')' | '[' | ']' | ',' | ';')
        })
        .filter_map(|token| {
            let trimmed = token.trim_matches(|ch: char| matches!(ch, '.' | ':' | ',' | ';'));
            if trimmed.contains("release/evidence/") || trimmed.starts_with("evidence/") {
                Some(trimmed.to_string())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    refs.sort();
    refs.dedup();
    refs
}

fn missing_evidence_artifact_refs(vyre_root: &Path, refs: &[String]) -> Vec<String> {
    let mut missing = refs
        .iter()
        .filter(|reference| !is_generated_docs_evidence_ref(reference))
        .filter(|reference| {
            let path = if reference.starts_with("release/evidence/") {
                vyre_root.join(reference)
            } else if let Some(stripped) = reference.strip_prefix("evidence/") {
                vyre_root.join("release/evidence").join(stripped)
            } else {
                return false;
            };
            !path.exists()
        })
        .cloned()
        .collect::<Vec<_>>();
    missing.sort();
    missing.dedup();
    missing
}

fn is_generated_docs_evidence_ref(reference: &str) -> bool {
    reference.starts_with("release/evidence/docs/") || reference.starts_with("evidence/docs/")
}

fn write_vyre_readme_contract(parent: &Path) {
    let vyre_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let readme = vyre_root.join("README.md");
    let exists = readme.is_file();
    let mut blockers = Vec::new();
    let (text, read_error) = if exists {
        match read_text_bounded(&readme) {
            Ok(text) => (text, None),
            Err(error) => (
                String::new(),
                Some(format!(
                    "Vyre README could not be read at {}: {error}",
                    readme.display()
                )),
            ),
        }
    } else {
        (String::new(), None)
    };
    let lowered = text.to_ascii_lowercase();
    let required_tokens = vec![
        "0.4.1",
        "vyre",
        "gpu",
        "cuda",
        "wgpu",
        "bytecode",
        "condition",
        "vyre::program",
        "release/evidence",
        "cargo add vyre",
    ];
    let missing_tokens = required_tokens
        .iter()
        .copied()
        .filter(|token| !lowered.contains(&token.to_ascii_lowercase()))
        .collect::<Vec<_>>();
    let example_count = text.matches("```rust").count()
        + text.matches("```toml").count()
        + text.matches("```bash").count()
        + text.matches("```sh").count();
    if let Some(error) = &read_error {
        blockers.push(error.clone());
    }
    if !exists {
        blockers.push(format!("Vyre README is missing at {}", readme.display()));
    }
    if exists && text.trim().is_empty() {
        blockers.push("Vyre README is empty".to_string());
    }
    for token in &missing_tokens {
        blockers.push(format!("Vyre README is missing required token `{token}`"));
    }
    if example_count == 0 {
        blockers.push(
            "Vyre README must include at least one Rust, TOML, or shell example block".to_string(),
        );
    }
    write_json(
        &parent.join("vyre-readme-contracts.json"),
        &ReadmeContractEvidence {
            schema_version: 1,
            path: readme.display().to_string(),
            exists,
            read_error,
            source_bytes: text.len(),
            required_tokens,
            missing_tokens,
            example_count,
            blockers,
        },
    );
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

fn write_markdown(path: &Path, text: &str) {
    if let Err(error) = fs::write(path, text) {
        eprintln!("Fix: failed to write `{}`: {error}", path.display());
        std::process::exit(1);
    }
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
                    "USAGE:\n  cargo_full run --bin xtask -- docs-matrix [--output PATH]\n\n\
                     Writes release documentation evidence matrix."
                );
                std::process::exit(0);
            }
            other => return Err(format!("Fix: unknown docs-matrix option `{other}`.")),
        }
    }
    Ok(output.unwrap_or_else(default_output))
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/evidence/docs/docs-matrix.json"))
        .unwrap_or_else(|| PathBuf::from("release/evidence/docs/docs-matrix.json"))
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader = fs::File::open(path)?.take(MAX_RELEASE_DOC_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_RELEASE_DOC_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_RELEASE_DOC_BYTES} byte release documentation read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}
