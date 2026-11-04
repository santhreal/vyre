use naga::Module;
use std::fs::File;
use std::io::{BufRead, BufReader};
use vyre_emit_naga::BindResultEntry;

/// Error type for [`load_bind_result_log`] failures.
#[derive(Debug)]
pub enum BindResultLogError {
    /// The log file could not be opened (missing, permission denied, etc.).
    Open(std::io::Error),
    /// A line could not be read from the file.
    Read(std::io::Error),
    /// A line in the log was not valid JSON for [`BindResultEntry`].
    /// Contains the line number (1-based) and the raw parse error.
    Parse(usize, serde_json::Error),
}

/// Human-readable context for a Naga validation or emission failure.
pub struct FailureTrace {
    /// Rendered failure context.
    pub text: String,
}

/// Build a validation failure trace for a Naga module.
pub fn failure_trace(module: &Module, error: &naga::valid::ValidationError) -> FailureTrace {
    let text = format!(
        "FAILURE: {:#?}\nentry_points={}\nfunctions={}\nglobals={}",
        error,
        module.entry_points.len(),
        module.functions.len(),
        module.global_variables.len()
    );
    FailureTrace { text }
}

/// Build a WGSL writer failure trace for a validated Naga module.
pub fn failure_trace_wgsl(
    module: &Module,
    info: &naga::valid::ModuleInfo,
    err: &naga::back::wgsl::Error,
) -> FailureTrace {
    let text = format!(
        "FAILURE: {:#?}\nentry_points={}\nfunctions={}\nglobals={}\nmodule_info={:#?}",
        err,
        module.entry_points.len(),
        module.functions.len(),
        module.global_variables.len(),
        info
    );
    FailureTrace { text }
}

/// Load a bind-result log file produced by vyre-emit-naga.
///
/// Returns `Err` on any I/O or parse failure so the caller can surface the
/// problem. Never silently returns a partial or empty result, the complete
/// log is required for accurate trace data.
pub fn load_bind_result_log(path: &str) -> Result<Vec<BindResultEntry>, BindResultLogError> {
    let file = File::open(path).map_err(BindResultLogError::Open)?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();
    for (line_no, raw) in reader.lines().enumerate() {
        let line = raw.map_err(BindResultLogError::Read)?;
        let entry: BindResultEntry =
            serde_json::from_str(&line).map_err(|e| BindResultLogError::Parse(line_no + 1, e))?;
        entries.push(entry);
    }
    Ok(entries)
}
