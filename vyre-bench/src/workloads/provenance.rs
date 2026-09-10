//! Payload, host, and clock provenance for every measured record in this crate.
//!
//! Two facts have one owner here. The payload provenance states that a measured
//! artifact came from generic IR and schedule search rather than an imported
//! kernel, a wrapper, a source template, or a name-matched dispatch branch, and
//! a payload from any other source is refused rather than recorded.
//! [`MeasurementProvenance`] states the host the measurement ran on and the
//! wall-clock instant it ran at. Both are read from the process and the acquired
//! device, so a record cannot carry a host or a timestamp that no run produced.

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

/// Host and clock facts one measured record was produced under.
///
/// A record states these instead of a constant because the constant survives the
/// host it was written on. Construction reads the running process and the
/// acquired device, so the only way to obtain the shape is to have measured
/// something.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementProvenance {
    /// Host architecture, operating system, backend, and device the run used.
    pub host_environment: String,
    /// RFC 3339 UTC instant the run was recorded at.
    pub recorded_at_utc: String,
}

impl MeasurementProvenance {
    /// Read the host, the acquired device, and the clock for one measured run.
    ///
    /// `backend_id` and `device_id` come from the materialized artifact's device
    /// identity, so a caller that has not acquired a device cannot call this.
    ///
    /// # Errors
    ///
    /// Returns an error when the system clock is before the Unix epoch or the
    /// device identity is empty.
    pub fn capture(backend_id: &str, device_id: &str) -> Result<Self, String> {
        if backend_id.trim().is_empty() || device_id.trim().is_empty() {
            return Err(format!(
                "measurement provenance requires an acquired device: backend `{backend_id}`, device `{device_id}`. Fix: acquire a dispatch backend before recording a measurement."
            ));
        }
        Ok(Self {
            host_environment: format!(
                "{}-{} / {backend_id} / {device_id}",
                std::env::consts::ARCH,
                std::env::consts::OS
            ),
            recorded_at_utc: utc_timestamp(std::time::SystemTime::now())?,
        })
    }
}

/// Format one system instant as an RFC 3339 UTC timestamp with second precision.
///
/// # Errors
///
/// Returns an error when `instant` is before the Unix epoch, which no measured
/// run can be.
pub fn utc_timestamp(instant: std::time::SystemTime) -> Result<String, String> {
    let seconds = instant
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| {
            format!(
                "system clock reports an instant before the Unix epoch: {error}. Fix: set the host clock before recording a measurement."
            )
        })?
        .as_secs();
    let days = i64::try_from(seconds / 86_400).map_err(|_| {
        "system clock reports a date outside the representable range. Fix: set the host clock before recording a measurement.".to_string()
    })?;
    let (year, month, day) = civil_from_unix_days(days);
    let second_of_day = seconds % 86_400;
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        second_of_day / 3_600,
        (second_of_day % 3_600) / 60,
        second_of_day % 60
    ))
}

/// Convert days since 1970-01-01 to a proleptic Gregorian civil date.
///
/// The shift places the epoch inside a 400-year era that starts on a leap-year
/// boundary, which removes the leap-day special cases from the division.
fn civil_from_unix_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = u32::try_from(day_of_year - (153 * shifted_month + 2) / 5 + 1)
        .expect("a day of month is within 1..=31");
    let month = u32::try_from(if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    })
    .expect("a month is within 1..=12");
    (if month <= 2 { year + 1 } else { year }, month, day)
}
