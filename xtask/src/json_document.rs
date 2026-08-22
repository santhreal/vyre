//! A JSON document a gate reads or writes.

use std::fs;
use std::path::Path;

use serde::Serialize;

/// Write `value` as pretty JSON with a trailing newline.
pub fn write(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create `{}`: {error}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize `{}`: {error}", path.display()))?;
    fs::write(path, format!("{json}\n"))
        .map_err(|error| format!("failed to write `{}`: {error}", path.display()))
}

/// Read one JSON artifact under a byte bound.
///
/// The error names whether the file could not be read or could not be parsed,
/// so a caller pushes one message and does not need two arms of its own: a
/// release gate and the package-readiness report each held the same pair.
pub fn read(path: &Path, max_bytes: u64) -> Result<serde_json::Value, String> {
    let text = crate::output_arg::read_text_bounded(path, max_bytes, "")
        .map_err(|error| format!("cannot read `{}`: {error}", path.display()))?;
    serde_json::from_str::<serde_json::Value>(&text)
        .map_err(|error| format!("`{}` is invalid JSON: {error}", path.display()))
}
