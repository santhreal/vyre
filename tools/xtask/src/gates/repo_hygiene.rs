//! Verify repository artifact hygiene and top-level metadata files.

use std::path::Path;
use walkdir::WalkDir;

/// Check repository hygiene rules.
pub fn check(root: &Path) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let changelog = root.join("CHANGELOG.md");
    match crate::workspace_manifest::read_source_bounded(&changelog) {
        Ok(text) => {
            if !text.starts_with("# Changelog") {
                errors.push("CHANGELOG.md must begin with `# Changelog`".to_owned());
            }
        }
        Err(_) => errors.push("missing CHANGELOG.md".to_owned()),
    }

    for cfg_file in ["Cargo.toml", ".cargo/config.toml"] {
        let path = root.join(cfg_file);
        if let Ok(text) = crate::workspace_manifest::read_source_bounded(&path) {
            for (idx, line) in text.lines().enumerate() {
                let trimmed = line.trim();
                if !trimmed.starts_with('#') && trimmed.contains("/tmp/") {
                    errors.push(format!("{cfg_file}:{}: references `/tmp/`", idx + 1));
                }
            }
        }
    }

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !crate::source_scan::is_pruned(e))
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
            if matches!(ext, "orig" | "rej" | "profraw") {
                errors.push(format!(
                    "stale artifact in workspace tree: {}",
                    entry
                        .path()
                        .strip_prefix(root)
                        .unwrap_or(entry.path())
                        .display()
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
