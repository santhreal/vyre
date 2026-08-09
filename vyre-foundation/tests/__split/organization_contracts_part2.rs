use super::*;

#[test]
fn workspace_wildcard_pub_reexports_are_baselined() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent().unwrap();

    let crates = [
        workspace_root.join("vyre-foundation/src"),
        workspace_root.join("vyre-libs/src"),
        workspace_root.join("vyre-primitives/src"),
        workspace_root.join("vyre-runtime/src"),
        workspace_root.join("vyre/src"),
        workspace_root.join("vyre-spec/src"),
        workspace_root.join("vyre-frontend-c/src"),
        workspace_root.join("conform/vyre-conform/src"),
    ];

    // ROADMAP HM3: vyre's `lower` shim re-exports `vyre-lower`
    // wholesale so external consumers can keep importing through
    // `vyre::lower::*`. The wildcard IS the contract.
    let known: HashSet<String> = [
        "vyre/src/lib.rs pub use vyre_lower::*;",
        "vyre-libs/src/matching/mod.rs pub use crate::scan::*;",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let mut new_violations = Vec::new();

    for src in &crates {
        if !src.is_dir() {
            continue;
        }
        let mut stack = vec![src.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                    let content = std::fs::read_to_string(&path).unwrap();
                    for (line_no, line) in content.lines().enumerate() {
                        let t = line.trim();
                        if t.starts_with("pub use") && t.ends_with("::*;") {
                            let rel = path.strip_prefix(workspace_root).unwrap_or(&path);
                            let key = format!("{} {}", rel.display(), t);
                            if !known.contains(&key) {
                                new_violations.push(format!(
                                    "{}:{} {}",
                                    rel.display(),
                                    line_no + 1,
                                    t
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    assert!(
        new_violations.is_empty(),
        "new wildcard pub re-exports are forbidden. Violations:\n{}",
        new_violations.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 9. Scheduling policy has a single source of truth
// ---------------------------------------------------------------------------

/// Organization contract: `SchedulingPolicy` must be defined in exactly one
/// location. Duplicate definitions create drift risk and violate the
/// substrate-neutrality contract.
#[test]
fn scheduling_policy_has_single_source_of_truth() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent().unwrap();

    let mut definitions = Vec::new();
    let src_dirs = [
        workspace_root.join("vyre-foundation/src"),
        workspace_root.join("vyre-driver/src"),
        workspace_root.join("vyre-runtime/src"),
        workspace_root.join("vyre-libs/src"),
        workspace_root.join("vyre-primitives/src"),
        workspace_root.join("vyre/src"),
    ];

    for src in &src_dirs {
        if !src.is_dir() {
            continue;
        }
        let mut stack = vec![src.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                    let content = std::fs::read_to_string(&path).unwrap();
                    for (line_no, line) in content.lines().enumerate() {
                        let t = line.trim();
                        if t.starts_with("pub struct SchedulingPolicy")
                            || t.starts_with("struct SchedulingPolicy")
                        {
                            let rel = path.strip_prefix(workspace_root).unwrap_or(&path);
                            definitions.push(format!("{}:{}", rel.display(), line_no + 1));
                        }
                    }
                }
            }
        }
    }

    assert_eq!(
        definitions.len(),
        1,
        "SchedulingPolicy must be defined in exactly one location. Found:\n{}",
        definitions.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 11. Agent/skills artifacts stay out of production crate dirs
// ---------------------------------------------------------------------------

// Organization contract: AGENTS.md, SKILL.md, and .kimi/ directories must not
// appear in production source directories (src/ or crate roots). Existing
// violations are baselined; new ones are forbidden. (`//` rather than `///`
// because this is the trailing comment of an `include!()`-d chunk.)
