use std::fs;
use std::path::Path;

use super::collectors::{compact_line, rel_path, skip_path};
use super::model::ScriptPolicyFinding;

pub(super) fn collect_script_policy_findings(root: &Path) -> Vec<ScriptPolicyFinding> {
    let mut findings = Vec::new();
    collect_script_policy_findings_in(root, &root.join("scripts"), &mut findings);
    findings
}

fn collect_script_policy_findings_in(
    root: &Path,
    path: &Path,
    findings: &mut Vec<ScriptPolicyFinding>,
) {
    let rel = rel_path(root, path);
    if skip_path(&rel) {
        return;
    }
    let Ok(metadata) = fs::metadata(path) else {
        return;
    };
    if metadata.is_dir() {
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                collect_script_policy_findings_in(root, &entry.path(), findings);
            }
        }
        return;
    }
    if path.extension().and_then(|ext| ext.to_str()) != Some("sh") {
        return;
    }
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    if duplicate_shell_toml_parser_body(&rel, &text) {
        let line = text
            .lines()
            .position(|line| line.contains("tomllib.load"))
            .map_or(1, |index| index + 1);
        findings.push(ScriptPolicyFinding {
            path: rel.clone(),
            line,
            text: compact_line("duplicate shell TOML parser body"),
            policy: "use scripts/lib/toml_reader.sh as the only shell TOML parser body".to_string(),
        });
    }
    let mut quote_state = ShellQuoteState::default();
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let has_unquoted_variable = quote_state.has_unquoted_shell_variable(line);
        if trimmed.contains("eval ") || trimmed.contains("bash -c") || trimmed.contains("sh -c") {
            findings.push(ScriptPolicyFinding {
                path: rel.clone(),
                line: line_index + 1,
                text: compact_line(line),
                policy: "no dynamic shell command construction in release scripts".to_string(),
            });
        }
        if sensitive_release_command(trimmed) && has_unquoted_variable {
            findings.push(ScriptPolicyFinding {
                path: rel.clone(),
                line: line_index + 1,
                text: compact_line(line),
                policy: "quote variables in release-script commands that handle repository targets, branches, tags, or evidence paths".to_string(),
            });
        }
    }
}

fn duplicate_shell_toml_parser_body(rel: &str, text: &str) -> bool {
    rel.starts_with("scripts/lib/")
        && rel != "scripts/lib/toml_reader.sh"
        && text.contains("python3 -")
        && text.contains("import tomllib")
        && text.contains("tomllib.load")
}

fn sensitive_release_command(line: &str) -> bool {
    line.starts_with("git ")
        || line.starts_with("gh ")
        || line.starts_with("cp ")
        || line.starts_with("mkdir ")
        || line.starts_with("jq ")
        || line.contains(" git ")
        || line.contains(" gh ")
}

#[derive(Default)]
struct ShellQuoteState {
    contexts: Vec<ShellQuoteContext>,
}

#[derive(Default)]
struct ShellQuoteContext {
    in_single: bool,
    in_double: bool,
    parenthesis_depth: usize,
}

impl ShellQuoteState {
    fn has_unquoted_shell_variable(&mut self, line: &str) -> bool {
        if self.contexts.is_empty() {
            self.contexts.push(ShellQuoteContext::default());
        }
        let chars = line.chars().collect::<Vec<_>>();
        let mut found_unquoted = false;
        let mut index = 0;
        while index < chars.len() {
            let nested_context = self.contexts.len() > 1;
            let mut push_command_substitution = false;
            let mut pop_command_substitution = false;
            {
                let context = self
                    .contexts
                    .last_mut()
                    .expect("shell quote state always has a root context");
                match chars[index] {
                    '\\' if !context.in_single => {
                        index = index.saturating_add(2);
                        continue;
                    }
                    '\'' if !context.in_double => context.in_single = !context.in_single,
                    '"' if !context.in_single => context.in_double = !context.in_double,
                    '$' if !context.in_single => {
                        let next = chars.get(index + 1).copied();
                        if next == Some('(') {
                            push_command_substitution = true;
                        } else if !context.in_double
                            && next.is_some_and(|ch| {
                                ch == '{' || ch == '_' || ch.is_ascii_alphabetic()
                            })
                        {
                            found_unquoted = true;
                        }
                    }
                    '(' if nested_context && !context.in_single && !context.in_double => {
                        context.parenthesis_depth += 1;
                    }
                    ')' if nested_context && !context.in_single && !context.in_double => {
                        context.parenthesis_depth = context.parenthesis_depth.saturating_sub(1);
                        pop_command_substitution = context.parenthesis_depth == 0;
                    }
                    _ => {}
                }
            }
            if push_command_substitution {
                self.contexts.push(ShellQuoteContext {
                    parenthesis_depth: 1,
                    ..ShellQuoteContext::default()
                });
                index += 2;
                continue;
            }
            if pop_command_substitution {
                self.contexts.pop();
            }
            index += 1;
        }
        found_unquoted
    }
}

#[cfg(test)]
mod tests {
    use super::{collect_script_policy_findings, ShellQuoteState};

    /// A copied TOML parser must remain visible so shell helpers cannot fork the canonical data-loading contract.
    #[test]
    fn duplicate_shell_toml_parser_body_is_a_script_policy_finding() {
        let dir = tempfile::tempdir().expect("Fix: create script-policy fixture directory.");
        let lib = dir.path().join("scripts/lib");
        std::fs::create_dir_all(&lib).expect("Fix: create scripts/lib fixture directory.");
        std::fs::write(
            lib.join("release_train.sh"),
            r#"#!/usr/bin/env bash
python3 - "$manifest" <<'PY'
import tomllib
tomllib.load(handle)
PY
"#,
        )
        .expect("Fix: write duplicate shell TOML parser fixture.");
        std::fs::write(
            lib.join("toml_reader.sh"),
            r#"#!/usr/bin/env bash
python3 - "$manifest" <<'PY'
import tomllib
tomllib.load(handle)
PY
"#,
        )
        .expect("Fix: write canonical shell TOML parser fixture.");

        let findings = collect_script_policy_findings(dir.path());

        assert!(
            findings.iter().any(|finding| {
                finding.path == "scripts/lib/release_train.sh"
                    && finding
                        .policy
                        .contains("scripts/lib/toml_reader.sh as the only shell TOML parser")
            }),
            "Fix: release script helpers must not copy the shell TOML parser body; findings={findings:?}"
        );
        assert!(
            findings
                .iter()
                .all(|finding| finding.path != "scripts/lib/toml_reader.sh"),
            "Fix: the canonical TOML reader must be allowed to own the parser body; findings={findings:?}"
        );
    }

    /// A variable quoted inside a command substitution must not be reported as an unquoted release path.
    #[test]
    fn nested_command_substitution_preserves_variable_quotes() {
        let mut state = ShellQuoteState::default();

        assert!(!state.has_unquoted_shell_variable(r#"mkdir -p "$(dirname "$BASELINE_FILE")""#));
    }

    /// Dollar-prefixed jq bindings inside a multiline single-quoted program are jq variables, not shell expansions.
    #[test]
    fn multiline_single_quoted_program_does_not_create_shell_findings() {
        let mut state = ShellQuoteState::default();
        let program = [
            "jq -n '",
            "  {evidence: (\"git push origin \" + $vyre_tag)}",
            "'",
        ];

        assert_eq!(
            program
                .iter()
                .map(|line| state.has_unquoted_shell_variable(line))
                .collect::<Vec<_>>(),
            vec![false, false, false]
        );
    }

    /// An actual unquoted release target must still fail after nested quote parsing is enabled.
    #[test]
    fn unquoted_release_variable_remains_detectable() {
        let mut state = ShellQuoteState::default();

        assert!(state.has_unquoted_shell_variable("git push $remote $release_tag"));
    }
}
