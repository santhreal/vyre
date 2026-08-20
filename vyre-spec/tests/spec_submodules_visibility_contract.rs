//! Contract tests verifying all specification submodules in `vyre-spec` are public to maintain SemVer compatibility.

const LIB_RS: &str = include_str!("../src/lib.rs");

/// Known internal-only non-public modules in vyre-spec.
/// Any module declaration not in this list MUST be `pub mod <name>;`.
const ALLOWED_PRIVATE_MODULES: &[&str] = &[
    "op_wire",        // internal macro helper module
    "catalog_slices", // internal static slice definitions
    "tests",          // internal unit test harness
];

#[derive(Debug, PartialEq, Eq)]
enum ModuleVisibility {
    Public,
    Private(String),
}

#[derive(Debug)]
struct ModuleDecl {
    name: String,
    visibility: ModuleVisibility,
    raw_line: String,
}

fn parse_module_declarations(source: &str) -> Vec<ModuleDecl> {
    let mut decls = Vec::new();
    let mut in_block_comment = false;

    for line in source.lines() {
        let trimmed = line.trim();
        if in_block_comment {
            if let Some(pos) = trimmed.find("*/") {
                in_block_comment = false;
                let remainder = trimmed[pos + 2..].trim();
                if remainder.is_empty() {
                    continue;
                }
            } else {
                continue;
            }
        }
        if trimmed.starts_with("/*") {
            if !trimmed.contains("*/") {
                in_block_comment = true;
                continue;
            }
        }
        if trimmed.starts_with("//") || trimmed.starts_with("#[") {
            continue;
        }

        // Look for module declarations ending in semicolon: `mod <name>;`, `pub mod <name>;`, `pub(...) mod <name>;`
        if let Some(mod_pos) = trimmed.find("mod ") {
            let before = trimmed[..mod_pos].trim();
            let after = trimmed[mod_pos + 4..].trim();
            if let Some(semicolon_pos) = after.find(';') {
                let mod_name = after[..semicolon_pos].trim().to_string();
                if !mod_name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    continue;
                }

                let visibility = if before.is_empty() {
                    ModuleVisibility::Private("mod (private)".to_string())
                } else if before == "pub" {
                    ModuleVisibility::Public
                } else if before.starts_with("pub(") {
                    ModuleVisibility::Private(before.to_string())
                } else {
                    ModuleVisibility::Private(before.to_string())
                };

                decls.push(ModuleDecl {
                    name: mod_name,
                    visibility,
                    raw_line: line.to_string(),
                });
            }
        }
    }
    decls
}

#[test]
fn all_spec_submodules_are_public() {
    let decls = parse_module_declarations(LIB_RS);

    assert!(
        decls.len() >= 35,
        "Expected at least 35 module declarations in vyre-spec/src/lib.rs, found {}",
        decls.len()
    );

    let mut violations = Vec::new();
    let mut public_count = 0;

    for decl in &decls {
        match &decl.visibility {
            ModuleVisibility::Public => {
                public_count += 1;
            }
            ModuleVisibility::Private(vis) => {
                if !ALLOWED_PRIVATE_MODULES.contains(&decl.name.as_str()) {
                    violations.push(format!(
                        "Module `{}` is declared as non-public ({}) breaking SemVer: `{}`. Specification submodules must be `pub mod {}`.",
                        decl.name, vis, decl.raw_line.trim(), decl.name
                    ));
                }
            }
        }
    }

    assert!(
        public_count >= 30,
        "Expected at least 30 public specification modules, found {}",
        public_count
    );

    if !violations.is_empty() {
        panic!(
            "Found {} SemVer visibility violation(s) in vyre-spec/src/lib.rs:\n{}",
            violations.len(),
            violations.join("\n")
        );
    }
}

#[test]
fn canary_spec_types_are_accessible() {
    let _cat: Option<vyre_spec::category::Category> = None;
    let _contract: Option<vyre_spec::op_contract::OperationContract> = None;
    let _inv: Option<vyre_spec::invariant::Invariant> = None;
    let _golden: Option<vyre_spec::golden_sample::GoldenSample> = None;
}
