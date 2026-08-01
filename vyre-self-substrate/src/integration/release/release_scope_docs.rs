//! Release scope documentation validation.

/// Release documentation scope declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReleaseScopeDoc<'a> {
    /// Document path.
    pub path: &'a str,
    /// Whether stable Vyre capabilities are listed separately.
    pub stable_vyre_section: bool,
    /// Whether stable Dataflow consumer capabilities are listed separately.
    pub stable_section: bool,
    /// Whether Vyrec beta status is explicit.
    pub vyrec_beta_section: bool,
    /// Whether unsupported lower steps are the only documented beta limitation.
    pub lower_steps_only_limitation: bool,
    /// Whether parser and semantic gaps are linked to parity/gap evidence.
    pub parser_semantic_gap_links: bool,
}

/// Release scope documentation proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReleaseScopeDocProof {
    /// Number of docs validated.
    pub doc_count: usize,
}

/// Validated committed release-scope documentation artifacts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReleaseScopeArtifactProof {
    /// Number of committed scope artifacts validated.
    pub artifact_count: usize,
}

/// Release scope documentation errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReleaseScopeDocError {
    /// No docs were supplied.
    EmptyDocs,
    /// Document path is empty.
    EmptyPath,
    /// Stable Vyre section is missing.
    MissingStableVyreSection {
        /// Document path.
        path: String,
    },
    /// Stable Dataflow consumer section is missing.
    MissingStableDataflowSection {
        /// Document path.
        path: String,
    },
    /// Vyrec beta section is missing.
    MissingVyrecBetaSection {
        /// Document path.
        path: String,
    },
    /// Docs hide parser or semantic gaps as beta limitations.
    HidesParserOrSemanticGaps {
        /// Document path.
        path: String,
    },
    /// Parser/semantic gaps are not linked to evidence.
    MissingGapEvidenceLinks {
        /// Document path.
        path: String,
    },
    /// Committed release-scope artifact is missing required evidence.
    ArtifactMissingEvidence {
        /// Missing evidence.
        evidence: &'static str,
    },
}

impl std::fmt::Display for ReleaseScopeDocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDocs => write!(
                f,
                "release scope docs are empty. Fix: provide docs separating stable Vyre/Dataflow consumer capability from Vyrec beta scope."
            ),
            Self::EmptyPath => write!(
                f,
                "release scope doc has empty path. Fix: record the doc artifact path."
            ),
            Self::MissingStableVyreSection { path } => write!(
                f,
                "release doc `{path}` lacks stable Vyre section. Fix: list stable Vyre capabilities separately."
            ),
            Self::MissingStableDataflowSection { path } => write!(
                f,
                "release doc `{path}` lacks stable Dataflow consumer section. Fix: list stable Dataflow consumer capabilities separately."
            ),
            Self::MissingVyrecBetaSection { path } => write!(
                f,
                "release doc `{path}` lacks Vyrec beta section. Fix: state Vyrec is active C compiler frontend beta."
            ),
            Self::HidesParserOrSemanticGaps { path } => write!(
                f,
                "release doc `{path}` hides parser or semantic gaps as beta limitations. Fix: beta limitations may cover lower steps only; parser/semantic gaps need explicit parity evidence."
            ),
            Self::MissingGapEvidenceLinks { path } => write!(
                f,
                "release doc `{path}` lacks parser/semantic gap evidence links. Fix: link to clang parity dashboard or gap findings."
            ),
            Self::ArtifactMissingEvidence { evidence } => write!(
                f,
                "release scope artifact is missing {evidence}. Fix: docs must separate stable platform, stable dataflow-consumer, compiler-frontend beta lower-step scope, and parser/semantic evidence."
            ),
        }
    }
}

impl std::error::Error for ReleaseScopeDocError {}

/// Validate release scope documentation.
pub fn validate_release_scope_docs(
    docs: &[ReleaseScopeDoc<'_>],
) -> Result<ReleaseScopeDocProof, ReleaseScopeDocError> {
    if docs.is_empty() {
        return Err(ReleaseScopeDocError::EmptyDocs);
    }
    for doc in docs {
        if doc.path.trim().is_empty() {
            return Err(ReleaseScopeDocError::EmptyPath);
        }
        if !doc.stable_vyre_section {
            return Err(ReleaseScopeDocError::MissingStableVyreSection {
                path: doc.path.to_owned(),
            });
        }
        if !doc.stable_section {
            return Err(ReleaseScopeDocError::MissingStableDataflowSection {
                path: doc.path.to_owned(),
            });
        }
        if !doc.vyrec_beta_section {
            return Err(ReleaseScopeDocError::MissingVyrecBetaSection {
                path: doc.path.to_owned(),
            });
        }
        if !doc.lower_steps_only_limitation {
            return Err(ReleaseScopeDocError::HidesParserOrSemanticGaps {
                path: doc.path.to_owned(),
            });
        }
        if !doc.parser_semantic_gap_links {
            return Err(ReleaseScopeDocError::MissingGapEvidenceLinks {
                path: doc.path.to_owned(),
            });
        }
    }
    Ok(ReleaseScopeDocProof {
        doc_count: docs.len(),
    })
}

type ArtifactRequirements = &'static [(&'static str, &'static str)];

const VYRE_README_REQUIREMENTS: ArtifactRequirements = &[
    ("stable Vyre README proof", "# Vyre README proof"),
    ("Vyre CUDA/WGPU release path", "CUDA-first/WGPU-fallback"),
    (
        "Vyre evidence-backed claims",
        "concrete release evidence artifacts",
    ),
];

const DATAFLOW_README_REQUIREMENTS: ArtifactRequirements = &[
    (
        "stable Dataflow consumer README proof",
        "# Dataflow consumer README proof",
    ),
    ("Dataflow consumer 0.1.0 API surface", "0.1.0"),
    (
        "standalone Dataflow consumer APIs",
        "standalone Dataflow consumer APIs",
    ),
];

const PARSER_DOC_REQUIREMENTS: ArtifactRequirements = &[
    ("parser documentation proof", "# Parser documentation proof"),
    (
        "parser object/VAST/semantic boundary",
        "parsing, object emission, VAST, semantic graph, and future compiler lowering",
    ),
    ("not full C compiler claim", "not a full C compiler claim"),
    (
        "unsupported feature handling",
        "unsupported-feature handling",
    ),
    (
        "distributed parser artifacts",
        "release/evidence/parser/distributed-parser-map.json",
    ),
];

const C_PARSER_LINUX_REQUIREMENTS: ArtifactRequirements = &[
    ("Linux C parser proof", "# C parser Linux subsystem proof"),
    ("semantic graph evidence", "semantic graph"),
    ("zero failed files", "failed files must be zero"),
    ("full corpus floor", "250"),
];

const RELEASE_NOTES_REQUIREMENTS: ArtifactRequirements = &[
    ("release notes evidence title", "# Release notes evidence"),
    ("Vyre release train", "vyre-driver-cuda@0.7.0"),
    ("Vyrec release surface", "vyrec"),
    ("Weir release version", "`weir` is present at `0.1.0`"),
    ("completion audit precondition", "completion audit"),
];

/// Validate committed release documentation artifacts for honest scope boundaries.
pub fn validate_committed_release_scope_artifacts(
    vyre_readme_proof: &str,
    readme_proof: &str,
    parser_doc_proof: &str,
    c_parser_linux_proof: &str,
    release_notes: &str,
) -> Result<ReleaseScopeArtifactProof, ReleaseScopeDocError> {
    for (artifact, requirements) in [
        (vyre_readme_proof, VYRE_README_REQUIREMENTS),
        (readme_proof, DATAFLOW_README_REQUIREMENTS),
        (parser_doc_proof, PARSER_DOC_REQUIREMENTS),
        (c_parser_linux_proof, C_PARSER_LINUX_REQUIREMENTS),
        (release_notes, RELEASE_NOTES_REQUIREMENTS),
    ] {
        for &(evidence, needle) in requirements {
            artifact_contains(artifact, evidence, needle)?;
        }
    }

    Ok(ReleaseScopeArtifactProof { artifact_count: 5 })
}

fn artifact_contains(
    artifact: &str,
    evidence: &'static str,
    needle: &str,
) -> Result<(), ReleaseScopeDocError> {
    if artifact.contains(needle) {
        Ok(())
    } else {
        Err(ReleaseScopeDocError::ArtifactMissingEvidence { evidence })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_scope_docs_accept_stable_and_beta_separation() {
        let proof = validate_release_scope_docs(&[doc()])
            .expect("Fix: valid release scope doc should pass");

        assert_eq!(proof.doc_count, 1);
    }

    #[test]
    fn release_scope_docs_reject_hidden_parser_semantic_gaps() {
        let mut bad = doc();
        bad.lower_steps_only_limitation = false;

        assert_eq!(
            validate_release_scope_docs(&[bad])
                .expect_err("hidden parser/semantic gap should fail"),
            ReleaseScopeDocError::HidesParserOrSemanticGaps {
                path: "release/docs/scope.md".to_owned(),
            }
        );
    }

    #[test]
    fn release_scope_docs_require_gap_links() {
        let mut bad = doc();
        bad.parser_semantic_gap_links = false;

        assert_eq!(
            validate_release_scope_docs(&[bad]).expect_err("missing gap links should fail"),
            ReleaseScopeDocError::MissingGapEvidenceLinks {
                path: "release/docs/scope.md".to_owned(),
            }
        );
    }

    #[test]
    fn release_scope_docs_accept_committed_scope_artifacts() {
        let proof = validate_committed_release_scope_artifacts(
            include_str!("../../../../release/evidence/docs/vyre-readme-proof.md"),
            include_str!("../../../../release/evidence/dataflow/readme-proof.md"),
            include_str!("../../../../release/evidence/docs/parser-doc-proof.md"),
            include_str!("../../../../release/evidence/docs/c-parser-linux-proof.md"),
            include_str!("../../../../release/evidence/docs/release-notes-platform.md"),
        )
        .expect("Fix: committed release scope docs should preserve stable/beta boundaries");

        assert_eq!(proof.artifact_count, 5);
    }

    #[test]
    fn release_scope_docs_reject_full_compiler_parser_claims() {
        let parser_doc =
            "# Parser documentation proof\nC parser release proof is a full C compiler claim.";
        let err = validate_committed_release_scope_artifacts(
            include_str!("../../../../release/evidence/docs/vyre-readme-proof.md"),
            include_str!("../../../../release/evidence/dataflow/readme-proof.md"),
            parser_doc,
            include_str!("../../../../release/evidence/docs/c-parser-linux-proof.md"),
            include_str!("../../../../release/evidence/docs/release-notes-platform.md"),
        )
        .expect_err("parser docs must reject full compiler scope claims");

        assert_eq!(
            err,
            ReleaseScopeDocError::ArtifactMissingEvidence {
                evidence: "parser object/VAST/semantic boundary",
            }
        );
    }

    fn doc() -> ReleaseScopeDoc<'static> {
        ReleaseScopeDoc {
            path: "release/docs/scope.md",
            stable_vyre_section: true,
            stable_section: true,
            vyrec_beta_section: true,
            lower_steps_only_limitation: true,
            parser_semantic_gap_links: true,
        }
    }
}
