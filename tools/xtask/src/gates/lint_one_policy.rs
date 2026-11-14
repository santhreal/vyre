//! Verify workspace lint inheritance and absence of unimplemented stubs in production code.

use std::path::Path;

/// Check `[lints] workspace = true` and production stub absence.
pub fn check(root: &Path) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for member in crate::workspace_manifest::workspace_members(root) {
        let manifest_path = root.join(&member).join("Cargo.toml");
        let Ok(text) = crate::workspace_manifest::read_source_bounded(&manifest_path) else {
            continue;
        };
        let Ok(doc) = toml::from_str::<toml::Value>(&text) else {
            continue;
        };
        let inherits = doc
            .get("lints")
            .and_then(|l| l.get("workspace"))
            .and_then(toml::Value::as_bool)
            == Some(true);
        if !inherits {
            errors.push(format!(
                "{member}/Cargo.toml: missing `[lints] workspace = true`"
            ));
        }
    }

    for rel in crate::workspace_manifest::source_file_roster(root) {
        if !rel.contains("/src/") || rel.starts_with("tools/xtask/") || rel.ends_with("/tests.rs") {
            continue;
        }
        let Ok(text) = crate::workspace_manifest::read_source_bounded(&root.join(&rel)) else {
            continue;
        };
        let stripped = crate::cfg_test::strip_cfg_test_items(&text);
        for offset in crate::source_scan::code_offsets(&stripped) {
            let tail = &stripped.as_bytes()[offset..];
            if tail.starts_with(b"todo!(") || tail.starts_with(b"unimplemented!(") {
                let prev_ident = offset > 0 && {
                    let b = stripped.as_bytes()[offset - 1];
                    b.is_ascii_alphanumeric() || b == b'_'
                };
                if !prev_ident {
                    errors.push(format!(
                        "{rel}: contains todo!() or unimplemented!() in production source"
                    ));
                    break;
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
