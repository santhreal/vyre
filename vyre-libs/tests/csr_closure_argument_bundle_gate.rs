//! Gate for the CSR closure argument-transposition class.
//!
//! Two defects hid inside the seven-and-nine-slot positional parameter lists
//! that every CSR closure entry point used to carry:
//!
//! 1. Four `&[u32]` slices in a row (`edge_offsets`, `edge_targets`,
//!    `edge_kind_mask`, `seed`). Any two of them swapped at a call site is a
//!    type-correct program that computes a different closure, and no compiler
//!    diagnostic exists for it.
//! 2. `csr_backward_or_changed::cpu_ref` named the per-edge kind array `masks`
//!    and the scalar edge-kind filter `edge_kind_mask`, inverting the roles the
//!    rest of the tree gives those two names. A reader repointing a call from a
//!    sibling module fed the scalar where the array belonged.
//!
//! Both members are discovered from the tree at run time: the file set is the
//! directory walk of `vyre-primitives/src/graph`, and the function set is every
//! `fn` those files declare. Nothing here is a checked-in list of names, so a
//! CSR entry point added tomorrow is gated on the day it lands, not on the day
//! someone remembers to extend a table.
//!
//! What this does not catch: a bundle whose *fields* are filled from the wrong
//! locals (`edge_offsets: &targets`). Field names make that visible at the call
//! site, which is the point of the bundle, but visible is not proven.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use vyre_test_support::collect_rust_files;
use vyre_test_support::monorepo::vyre_crate_directory;

/// A parameter as declared: name with any `mut` binding mode stripped, and the
/// type with interior whitespace collapsed so `& [u32]` and `&[u32]` compare
/// equal.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Param {
    name: String,
    ty: String,
}

#[derive(Debug, Clone)]
struct Function {
    file: String,
    name: String,
    params: Vec<Param>,
}

impl Function {
    /// Longest run of consecutive parameters that share one type. A run of
    /// three is the transposition hazard: two adjacent swaps are undetectable.
    fn longest_same_type_run(&self) -> usize {
        let mut longest = 1;
        let mut run = 1;
        for pair in self.params.windows(2) {
            run = if pair[0].ty == pair[1].ty { run + 1 } else { 1 };
            longest = longest.max(run);
        }
        if self.params.is_empty() {
            0
        } else {
            longest
        }
    }

    fn count_of_type(&self, ty: &str) -> usize {
        self.params.iter().filter(|p| p.ty == ty).count()
    }

    fn has_param_named(&self, name: &str) -> bool {
        self.params.iter().any(|p| p.name == name)
    }

    fn takes_csr_bundle(&self) -> bool {
        self.params
            .iter()
            .any(|p| p.ty.contains("CsrClosureInputs") || p.ty.contains("CsrGraphView"))
    }

    fn location(&self) -> String {
        format!("{}::{}", self.file, self.name)
    }
}

/// Every `.rs` file under `vyre-primitives/src/graph`, minus test subtrees.
/// Production signatures are the contract; a test may spell a fixture helper
/// however it likes.
fn graph_source_files() -> Vec<PathBuf> {
    let root = vyre_crate_directory("vyre-primitives").join("src/graph");
    assert!(
        root.is_dir(),
        "Fix: CSR closure gate could not find {} - the graph module moved, so point this gate at its new home.",
        root.display()
    );
    let mut files = Vec::new();
    collect_rust_files(&root, &mut files);
    files.retain(|path| !holds_test_code(path));
    files.sort();
    assert!(
        !files.is_empty(),
        "Fix: CSR closure gate walked {} and found no Rust sources, so it would pass vacuously.",
        root.display()
    );
    files
}

/// Whether a walked path holds test code rather than a production signature.
///
/// A `tests` directory and a `tests.rs` file are the two shapes this tree uses
/// for a module's tests, and neither declares a contract a call site must match.
fn holds_test_code(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == "tests")
        || path.file_stem().is_some_and(|stem| stem == "tests")
}

/// Blank out line comments, block comments and string literals so a `fn` inside
/// prose or a doc example is not mistaken for a declaration.
fn strip_noncode(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < bytes.len() {
        let rest = &src[i..];
        if rest.starts_with("//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
        } else if rest.starts_with("/*") {
            let mut depth = 0usize;
            while i < bytes.len() {
                if src[i..].starts_with("/*") {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if src[i..].starts_with("*/") {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    out.push(if bytes[i] == b'\n' { '\n' } else { ' ' });
                    i += 1;
                }
            }
        } else if bytes[i] == b'"' {
            out.push(' ');
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    out.push_str("  ");
                    i += 2;
                    continue;
                }
                let done = bytes[i] == b'"';
                out.push(if bytes[i] == b'\n' { '\n' } else { ' ' });
                i += 1;
                if done {
                    break;
                }
            }
        } else {
            let ch = src[i..].chars().next().expect("index is a char boundary");
            out.push_str(&" ".repeat(ch.len_utf8() - 1));
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Byte index of the closing delimiter that matches the opener at `open`.
fn matching_close(src: &str, open: usize) -> Option<usize> {
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    for (offset, byte) in bytes[open..].iter().enumerate() {
        match byte {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + offset);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split a parameter list on commas that sit at nesting depth zero, so
/// `Result<T, E>` and `&[(u32, u32)]` stay whole.
fn split_params(list: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for ch in list.chars() {
        match ch {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth -= 1,
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut current));
                continue;
            }
            _ => {}
        }
        current.push(ch);
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

fn parse_functions(file: &Path, display: &str) -> Vec<Function> {
    let raw = std::fs::read_to_string(file)
        .unwrap_or_else(|err| panic!("Fix: CSR closure gate cannot read {display}: {err}"));
    let src = strip_noncode(&raw);
    let mut out = Vec::new();
    let mut search = 0usize;
    while let Some(found) = src[search..].find("fn ") {
        let at = search + found;
        search = at + 3;
        // `fn` must be a token, not the tail of `asfn` or `r#fn`.
        if at > 0 {
            let prev = src.as_bytes()[at - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' || prev == b'#' {
                continue;
            }
        }
        let after = &src[at + 3..];
        let name: String = after
            .chars()
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let name_end = at + 3 + after.find(&name).unwrap_or(0) + name.len();
        let Some(paren_offset) = src[name_end..].find('(') else {
            continue;
        };
        // Only generics may sit between the name and the parameter list.
        let between = &src[name_end..name_end + paren_offset];
        if !between
            .chars()
            .all(|c| c.is_whitespace() || "<>,:'&[]_+".contains(c) || c.is_alphanumeric())
        {
            continue;
        }
        let open = name_end + paren_offset;
        let Some(close) = matching_close(&src, open) else {
            continue;
        };
        let mut params = Vec::new();
        for raw_param in split_params(&src[open + 1..close]) {
            let Some((lhs, rhs)) = raw_param.split_once(':') else {
                continue;
            };
            let name = lhs.trim().trim_start_matches("mut ").trim().to_string();
            let ty = rhs.split_whitespace().collect::<Vec<_>>().join("");
            if name.is_empty() || ty.is_empty() {
                continue;
            }
            params.push(Param { name, ty });
        }
        out.push(Function {
            file: display.to_string(),
            name,
            params,
        });
        search = close;
    }
    out
}

/// Every production function declared under `vyre-primitives/src/graph`.
fn graph_functions() -> Vec<Function> {
    let crate_dir = vyre_crate_directory("vyre-primitives");
    graph_source_files()
        .iter()
        .flat_map(|path| {
            let display = path
                .strip_prefix(&crate_dir)
                .unwrap_or(path)
                .display()
                .to_string();
            parse_functions(path, &display)
        })
        .collect()
}

/// The CSR-slice family: anything that receives three or more `&[u32]` buffers,
/// which is exactly the set where an adjacent swap is type-correct.
fn csr_slice_family(functions: &[Function]) -> Vec<&Function> {
    functions
        .iter()
        .filter(|f| f.count_of_type("&[u32]") >= 3 || f.takes_csr_bundle())
        .collect()
}

/// The closure entry points: CSR-slice-family members that iterate to a
/// fixpoint, marked by the `max_iters` budget they all carry, plus everything
/// already converted to the bundle.
fn csr_closure_entry_points(functions: &[Function]) -> Vec<&Function> {
    csr_slice_family(functions)
        .into_iter()
        .filter(|f| f.has_param_named("max_iters") || f.takes_csr_bundle())
        .collect()
}

#[test]
fn csr_closure_entry_points_take_the_graph_as_a_named_bundle() {
    let functions = graph_functions();
    let members = csr_closure_entry_points(&functions);
    assert!(
        members.len() >= 10,
        "Fix: the CSR closure gate discovered only {} entry points under src/graph, \
         so its discovery predicate no longer matches the family it guards. \
         Repair the predicate rather than accepting a vacuous pass.",
        members.len()
    );

    let unbundled: Vec<String> = members
        .iter()
        .filter(|f| !f.takes_csr_bundle() || f.longest_same_type_run() >= 3)
        .map(|f| {
            format!(
                "{} (longest same-type run {}, bundle {})",
                f.location(),
                f.longest_same_type_run(),
                if f.takes_csr_bundle() {
                    "present"
                } else {
                    "absent"
                }
            )
        })
        .collect();

    assert!(
        unbundled.is_empty(),
        "Fix: every CSR closure entry point must receive the graph as a \
         `CsrClosureInputs` / `CsrGraphView` bundle and must not declare three or more \
         consecutive parameters of one type, because adjacent same-typed arguments \
         transpose silently at call sites. Offenders: {}",
        unbundled.join(", ")
    );
}

#[test]
fn csr_slice_family_keeps_one_name_per_role() {
    let functions = graph_functions();
    let family = csr_slice_family(&functions);
    assert!(
        family.len() >= 40,
        "Fix: the CSR closure gate discovered only {} slice-taking functions under \
         src/graph, so its discovery predicate no longer matches the family it guards.",
        family.len()
    );

    // `edge_offsets`, `edge_targets` and `edge_kind_mask` are per-edge or
    // per-node arrays everywhere in the tree; `allow_mask` is the scalar
    // edge-kind filter everywhere in the tree. A signature that gives one of
    // these names the other role is the `csr_backward_or_changed` defect.
    let array_roles: BTreeSet<&str> = ["edge_offsets", "edge_targets", "edge_kind_mask"]
        .into_iter()
        .collect();
    let scalar_roles: BTreeSet<&str> = ["allow_mask"].into_iter().collect();

    let mut inverted = Vec::new();
    for function in &family {
        for param in &function.params {
            let is_slice = param.ty.ends_with("[u32]");
            if is_slice && scalar_roles.contains(param.name.as_str()) {
                inverted.push(format!(
                    "{}: `{}` is the scalar edge-kind filter everywhere else, declared here as `{}`",
                    function.location(),
                    param.name,
                    param.ty
                ));
            }
            if param.ty == "u32" && array_roles.contains(param.name.as_str()) {
                inverted.push(format!(
                    "{}: `{}` is a buffer everywhere else, declared here as `{}`",
                    function.location(),
                    param.name,
                    param.ty
                ));
            }
        }
    }

    assert!(
        inverted.is_empty(),
        "Fix: the CSR slice family must give each role one name across the tree, so a \
         call repointed from a sibling module cannot feed the scalar filter where the \
         per-edge array belongs. Inverted roles: {}",
        inverted.join("; ")
    );
}
