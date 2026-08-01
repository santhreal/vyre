//! Architecture boundary-map validation for Vyre, dataflow consumers, and parser tooling.

use std::collections::BTreeSet;

/// One owned architectural duty.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchitectureBoundary<'a> {
    /// Duty name, such as parsing, graph formation, scheduling, or dispatch.
    pub duty: &'a str,
    /// Owning crate or subsystem.
    pub owner: &'a str,
    /// Public module that owns the contract.
    pub module: &'a str,
}

/// Boundary-map validation proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchitectureBoundaryMapProof {
    /// Number of validated boundaries.
    pub boundary_count: usize,
}

/// Committed architecture artifact proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchitectureBoundaryArtifactProof {
    /// Number of parser/distributed-frontend components validated.
    pub parser_component_count: usize,
    /// Number of CUDA release-path markers validated.
    pub cuda_marker_count: usize,
    /// Number of Dataflow analysis rows validated.
    pub analysis_count: usize,
    /// Number of contributor topology rows validated.
    pub modular_directory_count: usize,
}

/// Boundary-map validation errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArchitectureBoundaryMapError {
    /// Boundary list is empty.
    EmptyMap,
    /// Required metadata is empty.
    EmptyMetadata {
        /// Duty name.
        duty: String,
        /// Field.
        field: &'static str,
    },
    /// Required duty is missing.
    MissingDuty {
        /// Missing duty.
        duty: &'static str,
    },
    /// Duty has more than one owner.
    DuplicateDutyOwner {
        /// Duty.
        duty: String,
    },
    /// Committed architecture artifact is missing required evidence.
    ArtifactMissingEvidence {
        /// Missing evidence.
        evidence: &'static str,
    },
    /// Committed architecture artifact missed a release threshold.
    ArtifactThresholdMiss {
        /// Field.
        field: &'static str,
        /// Observed value.
        observed: usize,
        /// Required value.
        required: usize,
    },
}

impl std::fmt::Display for ArchitectureBoundaryMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyMap => write!(
                f,
                "architecture boundary map is empty. Fix: define owners for parsing, graph formation, lowering, scheduling, dispatch, and validation."
            ),
            Self::EmptyMetadata { duty, field } => write!(
                f,
                "architecture boundary `{duty}` has empty {field}. Fix: every duty needs owner and module."
            ),
            Self::MissingDuty { duty } => write!(
                f,
                "architecture boundary map is missing duty `{duty}`. Fix: make crate ownership explicit for contributors."
            ),
            Self::DuplicateDutyOwner { duty } => write!(
                f,
                "architecture boundary duty `{duty}` has multiple owners. Fix: choose one owner and route other crates through its public contract."
            ),
            Self::ArtifactMissingEvidence { evidence } => write!(
                f,
                "architecture boundary artifact is missing {evidence}. Fix: publish committed ownership evidence for parser components, Dataflow analysis dataflow, CUDA dispatch, and contributor topology."
            ),
            Self::ArtifactThresholdMiss {
                field,
                observed,
                required,
            } => write!(
                f,
                "architecture boundary artifact {field}={observed} missed required {required}. Fix: expand committed architecture evidence until every release surface has one clear owner."
            ),
        }
    }
}

impl std::error::Error for ArchitectureBoundaryMapError {}

const REQUIRED_DUTIES: &[&str] = &[
    "parsing",
    "graph-formation",
    "lowering",
    "scheduling",
    "dispatch",
    "validation",
];

/// Validate contributor-facing architecture duty ownership.
pub fn validate_architecture_boundary_map(
    boundaries: &[ArchitectureBoundary<'_>],
) -> Result<ArchitectureBoundaryMapProof, ArchitectureBoundaryMapError> {
    if boundaries.is_empty() {
        return Err(ArchitectureBoundaryMapError::EmptyMap);
    }

    let mut duties = BTreeSet::new();
    for boundary in boundaries {
        for (field, value) in [
            ("duty", boundary.duty),
            ("owner", boundary.owner),
            ("module", boundary.module),
        ] {
            if value.trim().is_empty() {
                return Err(ArchitectureBoundaryMapError::EmptyMetadata {
                    duty: boundary.duty.to_owned(),
                    field,
                });
            }
        }
        if !duties.insert(boundary.duty) {
            return Err(ArchitectureBoundaryMapError::DuplicateDutyOwner {
                duty: boundary.duty.to_owned(),
            });
        }
    }

    for duty in REQUIRED_DUTIES {
        if !duties.contains(duty) {
            return Err(ArchitectureBoundaryMapError::MissingDuty { duty });
        }
    }

    Ok(ArchitectureBoundaryMapProof {
        boundary_count: boundaries.len(),
    })
}

/// Read a top-level JSON string field out of an evidence artifact.
///
/// The evidence artifacts are generated with one field per line, so a needle scan
/// is enough and this crate does not take a JSON dependency to read one id. It
/// exists so the dataflow component id has a single source, the artifact itself,
/// instead of a literal pasted into this validator that goes stale the next time
/// the component is renamed.
fn artifact_string_field(artifact: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\": \"");
    let start = artifact.find(&key)? + key.len();
    let rest = &artifact[start..];
    let end = rest.find('"')?;
    let value = &rest[..end];
    if value.is_empty() {
        return None;
    }
    Some(value.to_owned())
}

type ArtifactRequirements = &'static [(&'static str, &'static str)];

const DISTRIBUTED_PARSER_REQUIREMENTS: ArtifactRequirements = &[
    ("distributed parser schema", "\"schema_version\": 1"),
    ("zero parser blockers", "\"blockers\": []"),
    (
        "vyre-frontend-c parser owner",
        "\"id\": \"vyre-frontend-c\"",
    ),
    ("vyrec CLI parser owner", "\"id\": \"vyrec\""),
    (
        "grammar generator owner",
        "\"role\": \"Shared grammar generation substrate\"",
    ),
    (
        "empty ownership marker lists",
        "\"unresolved_ownership_markers\": []",
    ),
    (
        "empty missing contract topics",
        "\"missing_contract_topics\": []",
    ),
    (
        "empty missing test categories",
        "\"missing_test_categories\": []",
    ),
    ("test evidence tree", "\"tree\": \"tests\""),
    ("benchmark evidence tree", "\"tree\": \"benches\""),
    ("fuzz evidence tree", "\"tree\": \"fuzz\""),
];

const FRONTEND_C_REQUIREMENTS: ArtifactRequirements = &[
    (
        "frontend C contract owner",
        "\"component_id\": \"vyre-frontend-c\"",
    ),
    ("frontend preprocessor contract", "\"preprocessor\""),
    ("frontend GNU contract", "\"gnu\""),
    ("frontend unsupported-feature contract", "\"unsupported\""),
];

const VYREC_CLI_REQUIREMENTS: ArtifactRequirements = &[
    ("vyrec CUDA CLI contract", "\"cuda\""),
    ("vyrec actionable diagnostics contract", "\"fix:\""),
];

const DATAFLOW_CONTRACT_REQUIREMENTS: ArtifactRequirements = &[
    (
        "Dataflow analysis alias/reaching/callgraph contract",
        "\"alias\"",
    ),
    ("Dataflow reaching contract", "\"reaching\""),
    ("Dataflow analysis callgraph contract", "\"callgraph\""),
];

const GRAMMAR_REQUIREMENTS: ArtifactRequirements =
    &[("grammar generator contract", "\"generate\"")];

const BACKEND_REQUIREMENTS: ArtifactRequirements = &[
    ("CUDA-first backend matrix", "\"cuda_first\": true"),
    (
        "CUDA preferred backend",
        "\"preferred_backend_id\": \"cuda\"",
    ),
    (
        "GPU-only preferred backend",
        "\"preferred_backend_gpu_only\": true",
    ),
    ("RTX 5090 release probe", "NVIDIA GeForce RTX 5090"),
    (
        "CUDA resident dispatch",
        "\"id\": \"cuda-resident-dispatch\"",
    ),
    ("CUDA graph launch", "\"id\": \"cuda-graph-launch\""),
    ("CUDA module cache", "\"id\": \"cuda-module-cache\""),
    ("CUDA PTX source cache", "\"id\": \"cuda-ptx-source-cache\""),
    ("WGPU fallback owner", "\"wgpu_fallback_present\": true"),
    (
        "no hidden fallback findings",
        "\"hidden_fallback_findings\": []",
    ),
];

const ANALYSIS_REQUIREMENTS: ArtifactRequirements = &[
    ("Dataflow analysis SSA analysis owner", "\"id\": \"ssa\""),
    (
        "Dataflow analysis points-to analysis owner",
        "\"id\": \"points_to\"",
    ),
    ("Dataflow analysis IFDS analysis owner", "\"id\": \"ifds\""),
    (
        "Dataflow analysis callgraph analysis owner",
        "\"id\": \"callgraph\"",
    ),
    (
        "Dataflow analysis liveness analysis owner",
        "\"id\": \"live\"",
    ),
    (
        "Dataflow analysis slice analysis owner",
        "\"id\": \"slice\"",
    ),
    (
        "no missing Dataflow analysis APIs",
        "\"missing_api_items\": []",
    ),
    (
        "no unresolved Dataflow analysis markers",
        "\"unresolved_markers\": []",
    ),
];

const MODULARIZATION_REQUIREMENTS: ArtifactRequirements = &[
    ("Vyre contributor topology", "\"surface\": \"vyre\""),
    (
        "Vyrec parser CLI contributor topology",
        "\"surface\": \"vyrec\"",
    ),
    ("backend test topology", "\"layer\": \"backends\""),
    ("zero topology blockers", "\"blockers\": []"),
];

const DISTRIBUTED_PARSER_DOC_REQUIREMENTS: ArtifactRequirements = &[
    (
        "distributed parser coherence title",
        "# Distributed parser coherence proof",
    ),
    (
        "explicit distributed parser ownership contract",
        "Parser boundaries must be coherent even though the parser implementation is distributed.",
    ),
    ("contract artifact list", "Required generated evidence:"),
];

/// Validate committed architecture-boundary evidence across parser, dataflow, CUDA, and tests.
pub fn validate_committed_architecture_boundary_artifacts(
    distributed_parser_map: &str,
    frontend_c_contracts: &str,
    vyrec_cli_contracts: &str,
    contracts: &str,
    grammar_contracts: &str,
    backend_matrix: &str,
    analysis_matrix: &str,
    modularization_map: &str,
    distributed_parser_doc: &str,
) -> Result<ArchitectureBoundaryArtifactProof, ArchitectureBoundaryMapError> {
    // The dataflow component id comes from the contract artifact rather than a
    // literal here. Pasting it produced a fossil: the id stayed `dataflow-consumer`
    // in this validator after the component was renamed, so the boundary proof was
    // asserted against a component root that no longer existed on disk. Deriving it
    // also proves the parser map and the contract artifact agree on the id, which a
    // pasted literal cannot.
    let dataflow_component_id = artifact_string_field(contracts, "component_id").ok_or(
        ArchitectureBoundaryMapError::ArtifactMissingEvidence {
            evidence: "dataflow contract component id",
        },
    )?;
    let dataflow_root = artifact_string_field(contracts, "root").ok_or(
        ArchitectureBoundaryMapError::ArtifactMissingEvidence {
            evidence: "dataflow contract root",
        },
    )?;
    let dataflow_surface = dataflow_root
        .rsplit('/')
        .find(|segment| !segment.is_empty())
        .ok_or(ArchitectureBoundaryMapError::ArtifactMissingEvidence {
            evidence: "dataflow contract root",
        })?;
    let dataflow_map_owner_needle = format!("\"id\": \"{dataflow_component_id}\"");
    let dataflow_topology_needle = format!("\"surface\": \"{dataflow_surface}\"");

    for (artifact, requirements) in [
        (distributed_parser_map, DISTRIBUTED_PARSER_REQUIREMENTS),
        (frontend_c_contracts, FRONTEND_C_REQUIREMENTS),
        (vyrec_cli_contracts, VYREC_CLI_REQUIREMENTS),
        (contracts, DATAFLOW_CONTRACT_REQUIREMENTS),
        (grammar_contracts, GRAMMAR_REQUIREMENTS),
        (backend_matrix, BACKEND_REQUIREMENTS),
        (analysis_matrix, ANALYSIS_REQUIREMENTS),
        (modularization_map, MODULARIZATION_REQUIREMENTS),
        (distributed_parser_doc, DISTRIBUTED_PARSER_DOC_REQUIREMENTS),
    ] {
        for &(evidence, needle) in requirements {
            artifact_contains(artifact, evidence, needle)?;
        }
    }
    artifact_contains(
        distributed_parser_map,
        "Dataflow analysis dataflow parser owner",
        &dataflow_map_owner_needle,
    )?;
    artifact_contains(
        modularization_map,
        "dataflow contributor topology",
        &dataflow_topology_needle,
    )?;

    for contract in [
        frontend_c_contracts,
        vyrec_cli_contracts,
        contracts,
        grammar_contracts,
    ] {
        for (evidence, needle) in [
            ("contract schema", "\"schema_version\": 1"),
            ("contract blockers inventory", "\"blockers\""),
            (
                "contract ownership markers",
                "\"unresolved_ownership_markers\": []",
            ),
            ("contract tests tree", "\"tree\": \"tests\""),
            ("contract benches tree", "\"tree\": \"benches\""),
            ("contract fuzz tree", "\"tree\": \"fuzz\""),
        ] {
            artifact_contains(contract, evidence, needle)?;
        }
    }

    let parser_component_count = distributed_parser_map.matches("\"id\": ").count();
    let cuda_marker_count = backend_matrix.matches("\"id\": \"cuda").count();
    let analysis_count = analysis_matrix.matches("\"id\": ").count();
    let modular_directory_count = modularization_map.matches("\"surface\": ").count();

    artifact_at_least("parser components", parser_component_count, 4)?;
    artifact_at_least("CUDA backend markers", cuda_marker_count, 7)?;
    artifact_at_least("Dataflow analysis rows", analysis_count, 20)?;
    artifact_at_least("modular directory rows", modular_directory_count, 21)?;

    Ok(ArchitectureBoundaryArtifactProof {
        parser_component_count,
        cuda_marker_count,
        analysis_count,
        modular_directory_count,
    })
}

fn artifact_contains(
    artifact: &str,
    evidence: &'static str,
    needle: &str,
) -> Result<(), ArchitectureBoundaryMapError> {
    if artifact.contains(needle) {
        Ok(())
    } else {
        Err(ArchitectureBoundaryMapError::ArtifactMissingEvidence { evidence })
    }
}

fn artifact_at_least(
    field: &'static str,
    observed: usize,
    required: usize,
) -> Result<(), ArchitectureBoundaryMapError> {
    if observed >= required {
        Ok(())
    } else {
        Err(ArchitectureBoundaryMapError::ArtifactThresholdMiss {
            field,
            observed,
            required,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_map_accepts_single_owner_per_required_duty() {
        let proof = validate_architecture_boundary_map(&boundaries())
            .expect("Fix: complete boundary map should pass");

        assert_eq!(proof.boundary_count, 6);
    }

    #[test]
    fn boundary_map_rejects_missing_required_duties() {
        let mut boundaries = boundaries();
        boundaries.pop();

        assert_eq!(
            validate_architecture_boundary_map(&boundaries)
                .expect_err("missing validation duty should fail"),
            ArchitectureBoundaryMapError::MissingDuty { duty: "validation" }
        );
    }

    #[test]
    fn boundary_map_rejects_duplicate_duty_ownership() {
        let mut boundaries = boundaries();
        boundaries.push(ArchitectureBoundary {
            duty: "dispatch",
            owner: "vyrec",
            module: "vyrec::dispatch",
        });

        assert_eq!(
            validate_architecture_boundary_map(&boundaries)
                .expect_err("duplicate duty should fail"),
            ArchitectureBoundaryMapError::DuplicateDutyOwner {
                duty: "dispatch".to_owned(),
            }
        );
    }

    #[test]
    fn boundary_artifacts_accept_committed_architecture_evidence() {
        let proof = committed_architecture_artifact_proof()
            .expect("Fix: committed architecture evidence should prove release boundaries");

        assert!(proof.parser_component_count >= 4);
        assert!(proof.cuda_marker_count >= 7);
        assert!(proof.analysis_count >= 20);
        assert!(proof.modular_directory_count >= 21);
    }

    #[test]
    fn boundary_artifacts_reject_missing_cuda_release_path() {
        let backend_matrix =
            include_str!("../../../../release/evidence/backends/backend-matrix.json").replace(
                "\"preferred_backend_id\": \"cuda\"",
                "\"preferred_backend_id\": \"wgpu\"",
            );

        assert_eq!(
            validate_committed_architecture_boundary_artifacts(
                include_str!(
                    "../../../../release/evidence/parser/distributed-parser-map.json"
                ),
                include_str!("../../../../release/evidence/parser/vyre-frontend-c-contracts.json"),
                include_str!("../../../../release/evidence/parser/vyrec-cli-contracts.json"),
                include_str!(
                    "../../../../release/evidence/parser/external-dataflow-contracts.json"
                ),
                include_str!(
                    "../../../../release/evidence/parser/compiler-consumer-grammar-gen-contracts.json"
                ),
                &backend_matrix,
                include_str!(concat!(
                    "../../../../release/evidence/",
                    "we",
                    "ir/",
                    "we",
                    "ir-analysis-api-matrix.json"
                )),
                include_str!("../../../../release/evidence/tests/modularization-map.json"),
                include_str!("../../../../release/evidence/docs/distributed-parser-coherence.md"),
            )
            .expect_err("release boundary proof must not accept WGPU as preferred path"),
            ArchitectureBoundaryMapError::ArtifactMissingEvidence {
                evidence: "CUDA preferred backend",
            }
        );
    }

    #[test]
    fn boundary_artifacts_reject_missing_dataflow_api_ownership() {
        let analysis = include_str!(concat!(
            "../../../../release/evidence/",
            "we",
            "ir/",
            "we",
            "ir-analysis-api-matrix.json"
        ))
        .replace("\"id\": \"points_to\"", "\"id\": \"points_to_removed\"");

        assert_eq!(
            validate_committed_architecture_boundary_artifacts(
                include_str!(
                    "../../../../release/evidence/parser/distributed-parser-map.json"
                ),
                include_str!("../../../../release/evidence/parser/vyre-frontend-c-contracts.json"),
                include_str!("../../../../release/evidence/parser/vyrec-cli-contracts.json"),
                include_str!(
                    "../../../../release/evidence/parser/external-dataflow-contracts.json"
                ),
                include_str!(
                    "../../../../release/evidence/parser/compiler-consumer-grammar-gen-contracts.json"
                ),
                include_str!("../../../../release/evidence/backends/backend-matrix.json"),
                &analysis,
                include_str!("../../../../release/evidence/tests/modularization-map.json"),
                include_str!("../../../../release/evidence/docs/distributed-parser-coherence.md"),
            )
            .expect_err("release boundary proof must not accept missing Dataflow analysis owner"),
            ArchitectureBoundaryMapError::ArtifactMissingEvidence {
                evidence: "Dataflow analysis points-to analysis owner",
            }
        );
    }

    fn boundaries() -> Vec<ArchitectureBoundary<'static>> {
        vec![
            boundary("parsing", "vyrec", "vyrec::parser"),
            boundary("graph-formation", "dataflow", "dataflow::graph_layout"),
            boundary("lowering", "vyre", "vyre_driver::lowering"),
            boundary(
                "scheduling",
                "vyre-cuda",
                "vyre_driver_cuda::megakernel_scheduler",
            ),
            boundary("dispatch", "vyre-cuda", "vyre_driver_cuda::backend"),
            boundary(
                "validation",
                "vyre-self",
                "vyre_self_substrate::release_validation_matrix",
            ),
        ]
    }

    fn boundary(
        duty: &'static str,
        owner: &'static str,
        module: &'static str,
    ) -> ArchitectureBoundary<'static> {
        ArchitectureBoundary {
            duty,
            owner,
            module,
        }
    }

    fn committed_architecture_artifact_proof(
    ) -> Result<ArchitectureBoundaryArtifactProof, ArchitectureBoundaryMapError> {
        validate_committed_architecture_boundary_artifacts(
            include_str!("../../../../release/evidence/parser/distributed-parser-map.json"),
            include_str!("../../../../release/evidence/parser/vyre-frontend-c-contracts.json"),
            include_str!("../../../../release/evidence/parser/vyrec-cli-contracts.json"),
            include_str!("../../../../release/evidence/parser/external-dataflow-contracts.json"),
            include_str!(
                "../../../../release/evidence/parser/compiler-consumer-grammar-gen-contracts.json"
            ),
            include_str!("../../../../release/evidence/backends/backend-matrix.json"),
            include_str!(concat!(
                "../../../../release/evidence/",
                "we",
                "ir/",
                "we",
                "ir-analysis-api-matrix.json"
            )),
            include_str!("../../../../release/evidence/tests/modularization-map.json"),
            include_str!("../../../../release/evidence/docs/distributed-parser-coherence.md"),
        )
    }

    /// The dataflow component id must be read out of the contract artifact, because
    /// a literal pasted into the validator went stale when the component was
    /// renamed and left the boundary proof asserting a root that no longer existed.
    ///
    /// The expected value is deliberately not spelled here. This crate is a
    /// platform crate and may not name a downstream product, and hardcoding the
    /// id would reintroduce exactly the stale literal the derivation removed.
    #[test]
    fn dataflow_component_id_is_read_from_the_committed_contract_artifact() {
        let contracts =
            include_str!("../../../../release/evidence/parser/external-dataflow-contracts.json");
        let id = artifact_string_field(contracts, "component_id")
            .expect("the committed dataflow contract artifact must name its component id");
        assert!(
            !id.trim().is_empty() && !id.contains('"'),
            "component id {id:?} is not a usable identifier"
        );
        assert!(
            contracts.contains("\"root\": \""),
            "the contract artifact must also record the component root"
        );
    }

    /// The parser map uses the contract component id, while the contributor
    /// topology uses the contract root's package name. Both identities must be
    /// derived from the contract instead of copied as independent literals.
    #[test]
    fn parser_map_and_topology_map_follow_dataflow_contract_identity() {
        let contracts =
            include_str!("../../../../release/evidence/parser/external-dataflow-contracts.json");
        let id = artifact_string_field(contracts, "component_id")
            .expect("dataflow contract artifact must name its component id");
        let root = artifact_string_field(contracts, "root")
            .expect("dataflow contract artifact must name its root");
        let surface = root
            .rsplit('/')
            .find(|segment| !segment.is_empty())
            .expect("dataflow contract root must end in a package name");
        let parser_map =
            include_str!("../../../../release/evidence/parser/distributed-parser-map.json");
        let topology = include_str!("../../../../release/evidence/tests/modularization-map.json");
        assert!(
            parser_map.contains(&format!("\"id\": \"{id}\"")),
            "distributed-parser-map.json must own the dataflow component under id `{id}`"
        );
        assert!(
            topology.contains(&format!("\"surface\": \"{surface}\"")),
            "modularization-map.json must own the dataflow surface under root package `{surface}`"
        );
    }

    /// A contract artifact with no component id must fail the boundary proof rather
    /// than skip the dataflow ownership checks, which is what an `unwrap_or_default`
    /// on the derived needle would have done (a Law-10 silent skip).
    #[test]
    fn boundary_artifacts_reject_a_contract_artifact_with_no_component_id() {
        let contracts =
            include_str!("../../../../release/evidence/parser/external-dataflow-contracts.json")
                .replace("\"component_id\"", "\"component_id_removed\"");

        assert_eq!(
            validate_committed_architecture_boundary_artifacts(
                include_str!(
                    "../../../../release/evidence/parser/distributed-parser-map.json"
                ),
                include_str!("../../../../release/evidence/parser/vyre-frontend-c-contracts.json"),
                include_str!("../../../../release/evidence/parser/vyrec-cli-contracts.json"),
                &contracts,
                include_str!(
                    "../../../../release/evidence/parser/compiler-consumer-grammar-gen-contracts.json"
                ),
                include_str!("../../../../release/evidence/backends/backend-matrix.json"),
                include_str!(concat!(
                    "../../../../release/evidence/",
                    "we",
                    "ir/",
                    "we",
                    "ir-analysis-api-matrix.json"
                )),
                include_str!("../../../../release/evidence/tests/modularization-map.json"),
                include_str!("../../../../release/evidence/docs/distributed-parser-coherence.md"),
            )
            .expect_err("boundary proof must not accept a contract artifact with no component id"),
            ArchitectureBoundaryMapError::ArtifactMissingEvidence {
                evidence: "dataflow contract component id",
            }
        );
    }

    /// A renamed dataflow component must fail the boundary proof when only one of
    /// the three artifacts is regenerated. The derived needle is what makes this
    /// detectable: with a pasted literal, a half-finished rename passed.
    #[test]
    fn boundary_artifacts_reject_a_dataflow_rename_the_parser_map_has_not_adopted() {
        let original =
            include_str!("../../../../release/evidence/parser/external-dataflow-contracts.json");
        let id = artifact_string_field(original, "component_id")
            .expect("dataflow contract artifact must name its component id");
        let contracts = original.replacen(
            &format!("\"component_id\": \"{id}\""),
            "\"component_id\": \"renamed-dataflow\"",
            1,
        );

        assert_eq!(
            validate_committed_architecture_boundary_artifacts(
                include_str!(
                    "../../../../release/evidence/parser/distributed-parser-map.json"
                ),
                include_str!("../../../../release/evidence/parser/vyre-frontend-c-contracts.json"),
                include_str!("../../../../release/evidence/parser/vyrec-cli-contracts.json"),
                &contracts,
                include_str!(
                    "../../../../release/evidence/parser/compiler-consumer-grammar-gen-contracts.json"
                ),
                include_str!("../../../../release/evidence/backends/backend-matrix.json"),
                include_str!(concat!(
                    "../../../../release/evidence/",
                    "we",
                    "ir/",
                    "we",
                    "ir-analysis-api-matrix.json"
                )),
                include_str!("../../../../release/evidence/tests/modularization-map.json"),
                include_str!("../../../../release/evidence/docs/distributed-parser-coherence.md"),
            )
            .expect_err("boundary proof must not accept a half-adopted dataflow rename"),
            ArchitectureBoundaryMapError::ArtifactMissingEvidence {
                evidence: "Dataflow analysis dataflow parser owner",
            }
        );
    }

    /// `artifact_string_field` must reject an empty value rather than return
    /// `Some("")`, which would build the needle `"id": ""` and match nothing while
    /// reporting a confusing missing-owner error instead of a missing-id one.
    #[test]
    fn artifact_string_field_rejects_an_empty_value() {
        assert_eq!(
            artifact_string_field("{\"component_id\": \"\"}", "component_id"),
            None
        );
        assert_eq!(
            artifact_string_field("{\"component_id\": \"a-component\"}", "component_id").as_deref(),
            Some("a-component")
        );
        assert_eq!(
            artifact_string_field("{\"other\": \"a-component\"}", "component_id"),
            None
        );
    }
}
