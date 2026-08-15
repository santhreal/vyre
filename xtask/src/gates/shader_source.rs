//! Shader source is emitted through the backend AST, never built from strings.
//!
//! A shader assembled by string concatenation fails at dispatch time, after
//! every upstream correctness check has already passed. The rule has three
//! parts: shader syntax tokens may not appear in a string-building source file
//! outside the owning driver crate, a file inside that crate may not carry
//! string-append calls beside shader tokens, and the number of files that parse
//! a shader from text rather than emitting a module is pinned.
//!
//! The shell original held all three at a cap of zero and then compared the
//! measured count against zero with `-lt` twice, to report progress. Neither
//! branch can fire, because no count is below zero. Both are gone: the runner
//! reports a count below the pin.

use std::collections::BTreeSet;
use std::path::Path;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// Shader syntax tokens that give away shader construction.
const TOKENS: &[&str] = &[
    "@compute",
    "@workgroup_size",
    "@group(",
    "@binding(",
    "var<storage",
    "var<uniform",
    "var<workgroup",
    "-> @location",
];

/// Calls that build text rather than a module.
const STRING_BUILDERS: &[&str] = &[
    "push_str",
    "format_args",
    "format!",
    "write!",
    "writeln!",
    "r#\"",
];

/// Files permitted to hold shader syntax because they are shader sources, test
/// or benchmark inputs, or the driver's own structural emitter.
const PERMITTED_PREFIXES: &[&str] = &[
    "vyre-driver-wgpu/src/",
    "vyre-driver-wgpu/shaders/",
    "benches/shaders/",
    ".internals/",
    "vyre-foundation/src/transform/compiler/",
];

/// Where a shader may be parsed from text instead of emitted structurally.
const PARSE_SCOPE: &[&str] = &["vyre-driver-wgpu/src", "vyre-foundation/src"];

pub struct ShaderSource;

impl Gate for ShaderSource {
    fn name(&self) -> &'static str {
        "shader-source"
    }

    fn help(&self) -> &'static str {
        "shader text built from strings, and files parsing shader text"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }

        let sources = tree.all_rust();
        report.note(format!("scanned {} Rust source file(s)", sources.len()));

        for file in &sources {
            let text = tree.read(file)?;
            let builds_strings = scan::contains_any(&text, STRING_BUILDERS);
            let owned_by_driver = is_permitted(file);

            if builds_strings && !owned_by_driver {
                for (number, line) in scan::numbered(&text) {
                    if scan::is_comment(line) {
                        continue;
                    }
                    if let Some(token) = scan::first_of(line, TOKENS) {
                        report.find(Finding::at(
                            file.clone(),
                            number,
                            format!("shader token `{token}` in a file that builds strings"),
                            "emit the shader through the driver's module AST inside \
                             vyre-driver-wgpu, and delete the text assembly",
                        ));
                    }
                }
            }

            if file.starts_with("vyre-driver-wgpu/src") {
                let appends = text.matches("push_str").count();
                if appends > 0
                    && scan::contains_any(
                        &text,
                        &["@compute", "var<storage", "@workgroup_size"],
                    )
                {
                    report.find(Finding::in_file(
                        file.clone(),
                        format!(
                            "{appends} string-append call(s) beside shader tokens inside the driver crate"
                        ),
                        "build the module structurally and let the backend writer produce the \
                         text once, at the end",
                    ));
                }
            }
        }

        let parse_scope = tree.rust(PARSE_SCOPE)?;
        let mut parsing: BTreeSet<&Path> = BTreeSet::new();
        for file in &parse_scope {
            if tree.read(file)?.contains("naga::front::wgsl::parse_str") {
                parsing.insert(file.as_path());
            }
        }
        for file in parsing {
            report.find(Finding::in_file(
                file,
                "shader parsed from text rather than emitted as a module",
                "emit the module AST directly; every lowering owes a module, not a source string",
            ));
        }

        Ok(report)
    }
}

/// Whether a file may hold shader syntax.
fn is_permitted(path: &Path) -> bool {
    let text = path.to_string_lossy();
    if text.ends_with(".wgsl") {
        return true;
    }
    if scan::is_test_tree(path) {
        return true;
    }
    PERMITTED_PREFIXES
        .iter()
        .any(|prefix| text.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the permitted set is the whole strength of this rule. The shell
    /// original listed `docs/`, `ARCHITECTURE.md` and `THESIS.md` in a path
    /// allowance that only ever saw `*.rs` files, so three quarters of the
    /// allowance described files the scan could not reach.
    #[test]
    fn only_shader_owning_paths_may_hold_shader_syntax() {
        assert!(is_permitted(Path::new("vyre-driver-wgpu/src/emit.rs")));
        assert!(is_permitted(Path::new("vyre-driver-wgpu/shaders/add.wgsl")));
        assert!(is_permitted(Path::new("vyre-libs/tests/shader_text.rs")));
        assert!(is_permitted(Path::new("anything/x.wgsl")));
        assert!(!is_permitted(Path::new("vyre-lower/src/lowering.rs")));
        assert!(!is_permitted(Path::new("vyre-driver/src/pipeline.rs")));
    }
}
