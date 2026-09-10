//! A test that needs a device says so by failing.
//!
//! A GPU test that returns early when a probe fails is a smoke alarm wired to
//! nothing. This gate finds the silent-skip shapes and allows one only when a
//! loud abort sits near it, either an acquisition that panics or an assertion
//! that names the corrective action.

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// Lines above and below a skip site that may carry its loud abort.
const WINDOW_BEFORE: usize = 10;
const WINDOW_AFTER: usize = 20;

/// Evidence that a nearby path aborts loudly instead of skipping.
const LOUD: &[&str] = &[
    "acquire_or_panic",
    "panic!(\"no adapter",
    "panic!(\"adapter probe",
    "panic!(\"GPU required",
    "panic!(\"gpu required",
    "panic!(\"headless backend",
];

/// Silent-skip sites in GPU tests.
pub struct GpuLoudness;

impl crate::gate::GateBehavior for GpuLoudness {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        report.cover_complete("gpu source files", tree.all_rust().len());
        for path in tree.all_rust() {
            let text = tree.read(&path)?;
            let lines: Vec<&str> = text.lines().collect();
            let masked = scan::mask_literals(&text);
            let masked_lines: Vec<&str> = masked.lines().collect();

            // Try AST parse first to discover structured skip nodes
            if let Ok(syntax) = syn::parse_file(&text) {
                let mut ast_skips = Vec::new();
                find_ast_silent_skips(&syntax, &mut ast_skips);
                for (line, reason) in ast_skips {
                    let index = (line as usize).saturating_sub(1);
                    if !loud_within_window(&lines, index) {
                        report.find(Finding::at(
                            path.clone(),
                            line,
                            format!("{reason} skips the test when no device is present"),
                            "acquire the backend through the panicking constructor, or pair the \
                             skip with a test that exercises the same path and aborts loudly; a \
                             probe failure is a configuration failure and must be reported",
                        ));
                    }
                }
                // Also check comment-based skips and macro patterns that AST comments elide
                for (index, line) in lines.iter().enumerate() {
                    let blanked = masked_lines.get(index).copied().unwrap_or(line);
                    if let Some((code, comment)) = blanked.split_once("//") {
                        if comment.contains("no GPU") {
                            if code.contains("return Ok(());") || code.contains("return;") {
                                if !loud_within_window(&lines, index) {
                                    report.find(Finding::at(
                                        path.clone(),
                                        (index + 1) as u32,
                                        "a device-conditional early return skips the test when no device is present".to_string(),
                                        "acquire the backend through the panicking constructor, or pair the \
                                         skip with a test that exercises the same path and aborts loudly; a \
                                         probe failure is a configuration failure and must be reported",
                                    ));
                                }
                            }
                        }
                    }
                }
                continue;
            }

            // Fallback for unparseable fragments
            for (index, line) in lines.iter().enumerate() {
                let blanked = masked_lines.get(index).copied().unwrap_or(line);
                for skip in silent_skips(line, blanked) {
                    if loud_within_window(&lines, index) {
                        continue;
                    }
                    report.find(Finding::at(
                        path.clone(),
                        (index + 1) as u32,
                        format!("{skip} skips the test when no device is present"),
                        "acquire the backend through the panicking constructor, or pair the \
                         skip with a test that exercises the same path and aborts loudly; a \
                         probe failure is a configuration failure and must be reported",
                    ));
                }
            }
        }
        Ok(report)
    }
}

/// Collect silent skips from a parsed syn::File AST.
fn find_ast_silent_skips(file: &syn::File, sink: &mut Vec<(u32, &'static str)>) {
    for attr in &file.attrs {
        check_attr_for_silent_skip(attr, sink);
    }
    for item in &file.items {
        check_item_for_silent_skips(item, sink);
    }
}

fn check_attr_for_silent_skip(attr: &syn::Attribute, sink: &mut Vec<(u32, &'static str)>) {
    let s = quote::quote!(#attr).to_string();
    if s.contains("cfg (not (") && s.contains("gpu") {
        sink.push((
            attr.pound_token.span.start().line as u32,
            "a cfg that compiles the test out without a device",
        ));
    } else if s.contains("cfg_attr") && s.contains("gpu") && s.contains("ignore") {
        sink.push((
            attr.pound_token.span.start().line as u32,
            "a cfg_attr that ignores the test without the gpu feature",
        ));
    }
}

fn check_item_for_silent_skips(item: &syn::Item, sink: &mut Vec<(u32, &'static str)>) {
    match item {
        syn::Item::Fn(item_fn) => {
            for attr in &item_fn.attrs {
                check_attr_for_silent_skip(attr, sink);
            }
            for stmt in &item_fn.block.stmts {
                check_stmt_for_silent_skips(stmt, sink);
            }
        }
        syn::Item::Mod(item_mod) => {
            for attr in &item_mod.attrs {
                check_attr_for_silent_skip(attr, sink);
            }
            if let Some((_, items)) = &item_mod.content {
                for inner in items {
                    check_item_for_silent_skips(inner, sink);
                }
            }
        }
        _ => {}
    }
}

fn check_stmt_for_silent_skips(stmt: &syn::Stmt, sink: &mut Vec<(u32, &'static str)>) {
    match stmt {
        syn::Stmt::Expr(expr, _) => check_expr_for_silent_skips(expr, sink),
        syn::Stmt::Local(local) => {
            if let Some(init) = &local.init {
                check_expr_for_silent_skips(&init.expr, sink);
            }
        }
        _ => {}
    }
}

/// Walk every statement of `block`.
fn check_block_for_silent_skips(block: &syn::Block, sink: &mut Vec<(u32, &'static str)>) {
    for stmt in &block.stmts {
        check_stmt_for_silent_skips(stmt, sink);
    }
}

fn check_expr_for_silent_skips(expr: &syn::Expr, sink: &mut Vec<(u32, &'static str)>) {
    match expr {
        syn::Expr::If(expr_if) => {
            judge_guard(expr_if, sink);
            check_block_for_silent_skips(&expr_if.then_branch, sink);
            if let Some((_, otherwise)) = &expr_if.else_branch {
                check_expr_for_silent_skips(otherwise, sink);
            }
        }
        syn::Expr::Block(block) => check_block_for_silent_skips(&block.block, sink),
        syn::Expr::Loop(body) => check_block_for_silent_skips(&body.body, sink),
        syn::Expr::While(body) => check_block_for_silent_skips(&body.body, sink),
        syn::Expr::ForLoop(body) => check_block_for_silent_skips(&body.body, sink),
        syn::Expr::Match(expr_match) => {
            for arm in &expr_match.arms {
                check_expr_for_silent_skips(&arm.body, sink);
            }
        }
        syn::Expr::Macro(expr_macro) => {
            let mac_str = quote::quote!(#expr_macro).to_string();
            if (mac_str.contains("println !") || mac_str.contains("eprintln !"))
                && (mac_str.contains("skipped")
                    || mac_str.contains("no GPU")
                    || mac_str.contains("GPU unavailable"))
            {
                sink.push((
                    expr_macro.mac.path.segments[0].ident.span().start().line as u32,
                    "a printed excuse for not running",
                ));
            }
        }
        _ => {}
    }
}

/// Report `expr_if` when its own condition probes a failure and its own body
/// leaves successfully anyway.
///
/// The condition and the body are read separately. Stringifying the whole `if`
/// convicted any branch whose body happened to contain a guard and a `return`
/// that had nothing to do with each other, which is the shape of every CLI
/// entry point in the tree: `if subcommand == "prove" { if let Err(error) =
/// prove(args) { eprintln!("{error}"); exit(1) } return; }` printed the error
/// and exited nonzero, which is the loudest report available, and read as a
/// silent skip.
fn judge_guard(expr_if: &syn::ExprIf, sink: &mut Vec<(u32, &'static str)>) {
    let then_branch = &expr_if.then_branch;
    let body = quote::quote!(#then_branch).to_string();
    if !returns_successfully(&body) || aborts_loudly(&body) {
        return;
    }
    let condition = &expr_if.cond;
    let condition = quote::quote!(#condition).to_string();
    let line = expr_if.if_token.span.start().line as u32;
    if condition.contains("is_err ()") {
        sink.push((line, "an is_err guard returning early"));
    } else if condition.contains("let Err") {
        sink.push((line, "an if-let-Err guard returning early"));
    }
}

/// Whether a guard body leaves without carrying the failure it just observed.
///
/// This is the whole shape the gate exists to find. A body that returns the
/// error, or propagates it with `?`, has reported it to its caller and is
/// ordinary error handling; reading any `return` as a skip convicted every
/// recovery path in the drivers, including one that re-queues a flush and
/// returns the error it caught.
fn returns_successfully(body: &str) -> bool {
    body.contains("return ;") || body.contains("return Ok (") || body.trim_end().ends_with("return }")
}

/// Whether a guard body ends the process or reports the correction instead of
/// continuing.
///
/// A hardcoded list of six panic message prefixes decided this before, so a
/// body that aborted with any other wording, with a nonzero process exit, or
/// through an assertion was read as a skip. The question is what the body does,
/// not how its message opens. `exit (0)` is excluded: leaving successfully is
/// the skip this gate exists to find.
///
/// A body that emits a diagnostic naming the correction has reported the
/// failure, which is the same standard every error in this workspace is held
/// to. That is not the printed excuse the gate also looks for: an excuse says
/// the run was skipped and names nothing to do about it.
fn aborts_loudly(body: &str) -> bool {
    if body.contains("exit (") && !body.contains("exit (0)") {
        return true;
    }
    if body.contains("Fix:") {
        return true;
    }
    ["panic !", "unreachable !", "todo !", "unimplemented !", "abort ()"]
        .iter()
        .any(|needle| body.contains(needle))
        || body.contains("assert !")
        || body.contains("assert_eq !")
}

/// Which silent-skip shapes a line carries.
///
/// Three of these were unreachable in the shell original because their pattern
/// was malformed and the error was discarded, so no tree ever matched them. They
/// are live here, which is why the pin covers occurrences the shell never saw.
///
/// Two inputs, because the discriminator sits in a different place per shape. A
/// guard is code, and a detector that builds the same guard out of string pieces
/// must not read as one, so those shapes are judged on the masked line. So is a
/// skip that explains itself in a trailing comment: masking blanks a quoted
/// example of the shape and leaves a real comment untouched. An attribute
/// carries its feature name inside a literal, which masking blanks, so those
/// shapes are judged on the raw line and anchored on the attribute opener: a
/// pattern table row does not begin with `#[`. A printed excuse is a literal
/// too, and a table row spelling one escapes its own quotes, so the raw line
/// tells the two apart.
fn silent_skips(raw: &str, masked: &str) -> Vec<&'static str> {
    let mut found = Vec::new();
    if masked.contains("if ") && masked.contains("is_err()") && masked.contains('{') {
        if masked.contains("return Ok(());") {
            found.push("an is_err guard returning Ok");
        }
        if masked.contains("return;") {
            found.push("an is_err guard returning early");
        }
    }
    if masked.contains("if let Err(")
        && masked.contains('=')
        && masked.contains('{')
        && masked.contains("return")
    {
        found.push("an if-let-Err guard returning early");
    }
    for macro_name in ["println!(\"", "eprintln!(\""] {
        for excuse in ["skipped", "no GPU", "GPU unavailable"] {
            if raw.contains(&format!("{macro_name}{excuse}")) {
                found.push("a printed excuse for not running");
            }
        }
    }
    if raw.trim_start().starts_with("#[") {
        if raw.contains("#[cfg(not(") && raw.contains("gpu") {
            found.push("a cfg that compiles the test out without a device");
        }
        if raw.contains("#[cfg_attr(not(feature = \"gpu\")") && raw.contains("ignore") {
            found.push("a cfg_attr that ignores the test without the gpu feature");
        }
        if raw.contains("#[cfg_attr(not(any(") && raw.contains("gpu") && raw.contains("ignore") {
            found.push("a cfg_attr that ignores the test without any gpu feature");
        }
    }
    if let Some((code, comment)) = masked.split_once("//") {
        if comment.contains("no GPU") {
            if code.contains("return Ok(());") {
                found.push("a device-conditional early Ok");
            } else if code.contains("return;") {
                found.push("a device-conditional early return");
            }
        }
    }
    found
}

/// Whether a loud abort sits in the window around a skip site.
fn loud_within_window(lines: &[&str], index: usize) -> bool {
    let start = index.saturating_sub(WINDOW_BEFORE);
    let end = (index + WINDOW_AFTER + 1).min(lines.len());
    lines[start..end].iter().any(|line| {
        LOUD.iter().any(|needle| line.contains(needle))
            || (line.contains("assert!(\"") && line.contains("Fix:"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Judge one line the way the run loop does: raw beside its masked form.
    fn skips(line: &str) -> Vec<&'static str> {
        let masked = scan::mask_literals(line);
        silent_skips(line, &masked)
    }

    /// Judge whole source the way the AST arm does.
    fn ast_skips(source: &str) -> Vec<(u32, &'static str)> {
        let file = syn::parse_file(source).expect("test source must parse");
        let mut sink = Vec::new();
        find_ast_silent_skips(&file, &mut sink);
        sink
    }

    /// WHY: the arm has to keep convicting the shape it exists for, at every
    /// depth. The walk used to read only the top level of a function body, so a
    /// skip one block in was invisible; a gate that cannot fail on a nested
    /// instance of its own subject certifies what it never checked. The two
    /// conditions and the two successful exits are all four combinations.
    #[test]
    fn a_guard_that_leaves_successfully_is_convicted_at_any_depth() {
        for guard in [
            "if probe().is_err() { return; }",
            "if probe().is_err() { return Ok(()); }",
            "if let Err(error) = probe() { return; }",
            "if let Err(error) = probe() { return Ok(()); }",
        ] {
            for (shape, nesting) in [
                ("top level", format!("fn t() {{ {guard} }}")),
                ("a nested block", format!("fn t() {{ {{ {guard} }} }}")),
                ("a loop", format!("fn t() {{ loop {{ {guard} }} }}")),
                ("a for body", format!("fn t() {{ for _ in 0..1 {{ {guard} }} }}")),
                (
                    "an else branch",
                    format!("fn t() {{ if a {{ }} else {{ {guard} }} }}"),
                ),
                (
                    "a match arm",
                    format!("fn t() {{ match a {{ _ => {{ {guard} }} }} }}"),
                ),
                (
                    "a let initializer",
                    format!("fn t() {{ let _v = {{ {guard} 1 }}; }}"),
                ),
            ] {
                assert_eq!(
                    ast_skips(&nesting).len(),
                    1,
                    "{guard:?} in {shape} was not convicted"
                );
            }
        }
    }

    /// WHY: propagating the failure is the opposite of skipping it, and reading
    /// any `return` as a skip convicted thirty ordinary recovery paths in the
    /// drivers. Each of these observed an error and handed it to its caller.
    #[test]
    fn carrying_the_failure_onward_is_not_a_skip() {
        for body in [
            "fn t() -> R { if let Err(error) = probe() { return Err(error); } Ok(()) }",
            "fn t() -> R { if probe().is_err() { return Err(Failed); } Ok(()) }",
            "fn t() -> R { if let Err(e) = probe() { log(&e); return Err(e.into()); } Ok(()) }",
        ] {
            assert!(
                ast_skips(body).is_empty(),
                "propagation read as a skip: {body:?}"
            );
        }
    }

    /// WHY: stringifying the whole `if` joined a condition to a `return` that
    /// belonged to a different statement, which convicted every CLI entry point
    /// in the tree. This one prints the error and exits nonzero, the loudest
    /// report available, and the trailing `return` ends the dispatch arm.
    #[test]
    fn a_dispatch_arm_around_a_loud_guard_is_not_a_skip() {
        let source = "fn main() { if sub == \"prove\" { \
            if let Err(error) = prove(args) { eprintln!(\"{error}\"); std::process::exit(1); } \
            return; } }";
        assert!(
            ast_skips(source).is_empty(),
            "a nonzero exit inside a dispatch arm is a report, not a skip"
        );
        assert_eq!(
            ast_skips(
                "fn main() { if sub == \"prove\" { \
                 if let Err(_e) = prove(args) { return; } return; } }"
            )
            .len(),
            1,
            "the same arm with a swallowing guard must still be convicted"
        );
    }

    /// WHY: this workspace holds every error to naming its correction, so a body
    /// that emits one has reported the failure. Without this the gate demanded a
    /// panic from three diagnostic paths that already say what to do, and the
    /// only edit that clears such a finding is deleting the diagnostic. The
    /// negative case is the printed excuse the gate does look for: it names no
    /// correction.
    #[test]
    fn a_diagnostic_naming_the_correction_is_a_report() {
        assert!(
            ast_skips(
                "fn t() { if let Err(error) = write(p) { \
                 tracing::error!(\"could not write: {error}. Fix: free space on that volume.\"); \
                 return; } }"
            )
            .is_empty()
        );
        assert_eq!(
            ast_skips(
                "fn t() { if let Err(_e) = probe() { \
                 println!(\"skipping: no GPU\"); return; } }"
            )
            .len(),
            1,
            "an excuse that names nothing to do is still a skip"
        );
    }

    /// WHY: `exit(0)` leaves successfully, which is the skip, and every other
    /// exit code carries the failure out. Excluding the whole `exit(` family
    /// would let a test opt out of the gate by exiting clean.
    #[test]
    fn only_a_nonzero_exit_counts_as_carrying_the_failure() {
        assert!(
            ast_skips("fn t() { if probe().is_err() { std::process::exit(1); return; } }")
                .is_empty()
        );
        assert_eq!(
            ast_skips("fn t() { if probe().is_err() { std::process::exit(0); return; } }").len(),
            1,
            "exiting clean on a failed probe is the skip this gate exists to find"
        );
    }

    /// WHY: the shell original carried ten patterns and matched seven, because
    /// three were malformed and grep's exit of 2 read as no match. Enumerating
    /// every shape with the line it must catch is what keeps a shape from going
    /// quiet again: a shape that stops matching turns this red rather than
    /// lowering a count nobody reads.
    #[test]
    fn every_shape_matches_the_line_it_names() {
        let injections = [
            "        if backend.is_err() { return Ok(()); }",
            "        if backend.is_err() { return; }",
            "        if let Err(error) = probe() { return Ok(()); }",
            "        println!(\"skipped: no adapter\");",
            "        println!(\"no GPU on this host\");",
            "        eprintln!(\"GPU unavailable\");",
            "#[cfg(not(feature = \"gpu\"))]",
            "#[cfg_attr(not(feature = \"gpu\"), ignore)]",
            "#[cfg_attr(not(any(feature = \"gpu\", feature = \"cuda\")), ignore)]",
            "        return; // no GPU here",
            "        return Ok(()); // no GPU here",
        ];
        for line in injections {
            assert!(
                !skips(line).is_empty(),
                "no shape matched the injected line {line:?}"
            );
        }
    }

    /// WHY: the allowance is the only thing standing between a probe helper and a
    /// finding, so it has to apply to every shape rather than the one it was
    /// written against.
    #[test]
    fn a_loud_abort_covers_any_shape() {
        for line in [
            "        if backend.is_err() { return Ok(()); }",
            "#[cfg(not(feature = \"gpu\"))]",
            "        return; // no GPU here",
        ] {
            let lines = vec![line, "        let backend = Backend::acquire_or_panic();"];
            assert!(
                loud_within_window(&lines, 0),
                "the allowance missed {line:?}"
            );
        }
    }

    /// WHY: the three cfg shapes are the ones the shell original could never
    /// match. If they stop matching here the gate silently returns to asserting
    /// nothing, which is the defect this port exists to fix. They also prove the
    /// attribute shapes are read raw: masking blanks the feature name they key on.
    #[test]
    fn the_cfg_shapes_are_reachable() {
        assert_eq!(
            skips("#[cfg(not(feature = \"gpu\"))]").len(),
            1,
            "a cfg that compiles a test out without a device is a skip"
        );
        assert_eq!(
            skips("#[cfg_attr(not(feature = \"gpu\"), ignore)]").len(),
            1
        );
        assert_eq!(
            skips("#[cfg_attr(not(any(feature = \"gpu\", feature = \"cuda\")), ignore)]").len(),
            1
        );
    }

    /// WHY: a detector's own pattern table is source that contains every shape it
    /// looks for. A guard written as code that builds a pattern must not read as
    /// a guard, which is what the mask buys, and an attribute row in a table does
    /// not start with the attribute opener, which is what the anchor buys. A
    /// quoted example of a commented skip is a row too: the mask blanks the
    /// comment inside the literal and leaves a real trailing comment standing,
    /// which is how the two are told apart.
    #[test]
    fn a_pattern_table_is_not_a_skip_site() {
        assert!(
            skips("if line.contains(\"if let Err(\") && line.contains(\"return\") { hit(); }")
                .is_empty()
        );
        assert!(skips("if line.contains(\"#[cfg(not(\") && line.contains(\"gpu\") {").is_empty());
        assert!(
            skips("            \"        return; // no GPU here\",").is_empty(),
            "a quoted example of the commented shape is a table row, not a skip site"
        );
        let source = "let x = 1; // no GPU here\n";
        let masked = scan::mask_literals(source);
        assert_eq!(masked.trim_end(), "let x = 1; // no GPU here");
        assert_eq!(masked.len(), source.len());
    }

    /// WHY: the allowance is the whole reason a legitimate probe helper does not
    /// read as a violation, and it is bounded. A loud abort thirty lines below a
    /// skip does not cover it.
    #[test]
    fn the_allowance_window_is_bounded() {
        let mut lines = vec!["if probe().is_err() { return; }"];
        lines.resize(26, "    // filler");
        lines.push("    let backend = Backend::acquire_or_panic();");
        assert!(!loud_within_window(&lines, 0));
        let near = vec![
            "let backend = Backend::acquire_or_panic();",
            "if probe().is_err() { return; }",
        ];
        assert!(loud_within_window(&near, 1));
    }

    /// WHY: the comment forms only count when the excuse is in the comment. A
    /// return next to an unrelated comment is ordinary control flow.
    #[test]
    fn a_commented_return_counts_only_on_the_device_excuse() {
        assert_eq!(skips("        return; // no GPU here").len(), 1);
        assert!(skips("        return; // caller owns the retry").is_empty());
    }
}
