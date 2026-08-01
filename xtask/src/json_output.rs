use std::fs;
use std::path::Path;

use serde::Serialize;

pub(crate) fn write_pretty_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create `{}`: {error}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize `{}`: {error}", path.display()))?;
    fs::write(path, format!("{json}\n"))
        .map_err(|error| format!("failed to write `{}`: {error}", path.display()))
}
