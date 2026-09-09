//! Payload generation provenance validation for BACKLOG row 47.
//!
//! BACKLOG row 47 requires:
//! "The measured payload must be generated from generic IR and schedule search,
//! not an imported kernel, wrapper, source template, or model-name dispatch."
//! A comparison whose payload came from any non-generic source does not count,
//! and the harness must refuse it rather than record it.

use serde::{Deserialize, Serialize};

/// Provenance of the measured compiler payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PayloadProvenance {
    /// Permitted: Emitted artifact is synthesized from generic IR graph and megakernel schedule search.
    GenericIrAndScheduleSearch {
        /// Intermediate representation graph digest.
        ir_graph_digest: String,
        /// Number of schedule candidates evaluated during search.
        candidates_searched: u32,
    },
    /// Refused: An imported third-party kernel binary or precompiled module.
    ImportedKernel {
        /// Imported kernel identifier or symbol name.
        kernel_name: String,
        /// Origin library or external source.
        origin: String,
    },
    /// Refused: A host wrapper calling external runtime routines directly.
    Wrapper {
        /// Wrapper or shim struct identifier.
        wrapper_name: String,
    },
    /// Refused: A hardcoded backend source template or macro string interpolation.
    SourceTemplate {
        /// Name or path of the hardcoded template.
        template_name: String,
    },
    /// Refused: A dispatch branch selected by matching model names or workload keywords.
    ModelNameDispatch {
        /// Matched model or benchmark name.
        model_name: String,
    },
}

impl PayloadProvenance {
    /// String classification of the provenance kind.
    #[must_use]
    pub const fn kind_str(&self) -> &'static str {
        match self {
            Self::GenericIrAndScheduleSearch { .. } => "generic_ir_and_schedule_search",
            Self::ImportedKernel { .. } => "imported_kernel",
            Self::Wrapper { .. } => "wrapper",
            Self::SourceTemplate { .. } => "source_template",
            Self::ModelNameDispatch { .. } => "model_name_dispatch",
        }
    }

    /// Whether this payload provenance is valid for release floor qualification.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        matches!(self, Self::GenericIrAndScheduleSearch { .. })
    }
}

/// Refusal generated when a measured payload fails provenance validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceRefusal {
    /// Identifier of the benchmark case or workload.
    pub case_id: String,
    /// The refused provenance kind.
    pub provenance: PayloadProvenance,
    /// Detailed diagnostic explaining the refusal and required remediation.
    pub reason: String,
}

impl std::fmt::Display for ProvenanceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "refused: benchmark case `{}` payload provenance `{}` is not permitted: {}",
            self.case_id,
            self.provenance.kind_str(),
            self.reason
        )
    }
}

impl std::error::Error for ProvenanceRefusal {}

/// Validate that a case payload originated strictly from generic IR and schedule search.
///
/// Refuses imported kernels, wrappers, source templates, and model-name dispatches by name.
pub fn validate_payload_provenance(
    case_id: &str,
    provenance: &PayloadProvenance,
) -> Result<(), ProvenanceRefusal> {
    match provenance {
        PayloadProvenance::GenericIrAndScheduleSearch {
            candidates_searched,
            ..
        } => {
            if *candidates_searched == 0 {
                return Err(ProvenanceRefusal {
                    case_id: case_id.to_string(),
                    provenance: provenance.clone(),
                    reason: "schedule search evaluated 0 candidates; search must explore candidate space".to_string(),
                });
            }
            Ok(())
        }
        PayloadProvenance::ImportedKernel { kernel_name, origin } => Err(ProvenanceRefusal {
            case_id: case_id.to_string(),
            provenance: provenance.clone(),
            reason: format!(
                "Fix: imported kernel `{kernel_name}` from `{origin}` is not generated compiler code"
            ),
        }),
        PayloadProvenance::Wrapper { wrapper_name } => Err(ProvenanceRefusal {
            case_id: case_id.to_string(),
            provenance: provenance.clone(),
            reason: format!(
                "Fix: wrapper `{wrapper_name}` delegates execution rather than compiling IR"
            ),
        }),
        PayloadProvenance::SourceTemplate { template_name } => Err(ProvenanceRefusal {
            case_id: case_id.to_string(),
            provenance: provenance.clone(),
            reason: format!(
                "Fix: source template `{template_name}` substitutes text instead of lowering generic IR"
            ),
        }),
        PayloadProvenance::ModelNameDispatch { model_name } => Err(ProvenanceRefusal {
            case_id: case_id.to_string(),
            provenance: provenance.clone(),
            reason: format!(
                "Fix: model name dispatch `{model_name}` selects a special-case branch instead of autonomous schedule search"
            ),
        }),
    }
}
