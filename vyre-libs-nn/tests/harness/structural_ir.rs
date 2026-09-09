//! Structural IR summary of a `Program`, and the golden it is pinned to.
//!
//! # Why this is not `Program::fingerprint`
//!
//! A clone-family guard asks one question: does the surviving owner still emit
//! what every former copy emitted. `Program::fingerprint` answers a different
//! one. It is BLAKE3 over `canonical_wire_bytes`, and those bytes open with
//! `WIRE_FORMAT_VERSION`, so incrementing the serialization revision moves
//! every fingerprint in the workspace while no program's meaning moves. A table
//! of wire digests therefore goes red on a relabelling, reports the difference
//! as 32 opaque bytes, and is answered by copying the measured numbers back in.
//!
//! # What a section covers
//!
//! Everything here is a function of the IR model, never of the wire format:
//!
//! - `digest` is BLAKE3 over [`render_structural_ir`], the derived `Debug` of
//!   the canonicalized buffer roster and entry node tree. It carries node
//!   kinds, field names, operand expressions, literal values, identifier text,
//!   region generators and nesting, in source order. A changed operand, a
//!   dropped node or a reordered data dependence moves it.
//! - The buffer roster is written out in full, because it is the program's ABI
//!   and it is small.
//! - The node and expression histograms, the nesting depth, and the per-region
//!   breakdown say which class of change happened. A moved digest with an
//!   unchanged histogram is an operand or an order; a changed node histogram is
//!   a node added or dropped; a changed roster is the ABI; a change confined to
//!   one region identity localizes to that owner.
//!
//! The full rendering is not checked in: two direct-path fixtures unroll to
//! about 53000 lines each, and the corpus would be ten megabytes. Regenerate it
//! locally from [`render_structural_ir`] when a digest moves and the histograms
//! do not say enough.
//!
//! Using derived `Debug` and the `visit` decomposition rather than a
//! hand-written match buys the same closure the workspace relies on elsewhere:
//! `Node` and `Expr` are declared by one macro invocation and are
//! `#[non_exhaustive]`, `child_bodies`, `node_operands` and `expr_children` are
//! the exhaustive owners of nesting and operand positions, so a new variant or
//! a new field reaches this summary with no edit here.
//!
//! Canonicalization runs first, so authoring-order noise is normalized away:
//! buffer declarations sort by their stable key and commutative operands take
//! their canonical order. Neither is a semantic difference, and a guard that
//! fails on one fails on a non-bug.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use vyre_foundation::hashing::domain_digest;
use vyre_foundation::ir::{expr_variant_name, node_variant_name, Expr, Node, Program};
use vyre_foundation::visit::{child_bodies, expr_children, node_operands, node_variadic_operands};

/// Line that opens each entry point's section in a golden.
const SECTION_MARKER: &str = "===== ";

/// Domain separator for the structural digest.
const DIGEST_DOMAIN: &[u8] = b"vyre-libs-nn/nn-attention-clone-family-structural-ir/v1";

/// Header written above a golden, naming what it holds and how to regenerate it.
const HEADER: &str = "\
# Structural IR golden for the nn/attention clone families.
#
# One section per clone-family entry point, in roster order. `digest` is BLAKE3
# over the canonicalized buffer roster and node tree of the program that entry
# point builds. The remaining lines say which class of change a moved digest is:
# the roster is the ABI, the histograms and depth say whether a node was added
# or dropped, and the per-region breakdown localizes a change to one owner.
#
# Nothing here reads the wire format, so a serialization revision does not move
# a section and a changed operand, dropped node or reordered data dependence
# does.
#
# Regenerate with the bless test in the file that reads this golden, then read
# the diff: a change here is a change in what the compiler emits.
";

/// Full structural rendering of `program`, canonicalized.
///
/// The digest is taken over this. It is not checked in; see the module docs.
pub(crate) fn render_structural_ir(program: &Program) -> String {
    let canonical = program.canonicalized();
    format!(
        "buffers:\n{:#?}\n\nentry:\n{:#?}\n",
        canonical.buffers(),
        canonical.entry()
    )
}

/// Golden section body for `program`.
pub(crate) fn render_section(program: &Program) -> String {
    let canonical = program.canonicalized();
    let digest = domain_digest(DIGEST_DOMAIN, render_structural_ir(program).as_bytes());

    let mut out = String::new();
    let _ = writeln!(out, "digest {}", hex(&digest));

    let mut whole = Counts::default();
    let mut regions: BTreeMap<&str, Counts> = BTreeMap::new();
    for node in canonical.entry() {
        walk(node, 1, &mut whole, None, &mut regions);
    }
    let _ = writeln!(out, "nodes {} exprs {}", whole.nodes, whole.exprs);
    let _ = writeln!(out, "depth {}", whole.depth);
    let _ = writeln!(out, "node_kinds {}", render_histogram(&whole.node_kinds));
    let _ = writeln!(out, "expr_kinds {}", render_histogram(&whole.expr_kinds));

    out.push_str("buffers:\n");
    for buffer in canonical.buffers() {
        let _ = writeln!(out, "  {buffer:?}");
    }

    out.push_str("regions:\n");
    for (identity, counts) in &regions {
        let _ = writeln!(
            out,
            "  {identity} nodes={} exprs={} node_kinds={} expr_kinds={}",
            counts.nodes,
            counts.exprs,
            render_histogram(&counts.node_kinds),
            render_histogram(&counts.expr_kinds),
        );
    }
    out
}

/// Node and expression tallies for one scope.
#[derive(Default)]
struct Counts {
    nodes: usize,
    exprs: usize,
    depth: usize,
    node_kinds: BTreeMap<&'static str, usize>,
    expr_kinds: BTreeMap<&'static str, usize>,
}

impl Counts {
    fn record_node(&mut self, node: &Node, depth: usize) {
        self.nodes += 1;
        self.depth = self.depth.max(depth);
        *self.node_kinds.entry(node_variant_name(node)).or_default() += 1;
    }

    fn record_expr(&mut self, expr: &Expr) {
        self.exprs += 1;
        *self.expr_kinds.entry(expr_variant_name(expr)).or_default() += 1;
    }
}

/// Tally `node` into the whole-program counts and into its innermost enclosing
/// region, then descend.
///
/// A nested region's nodes are tallied under that region rather than its
/// parent, so a change inside a shared owner localizes to the owner instead of
/// showing up in every entry point that embeds it.
fn walk<'a>(
    node: &'a Node,
    depth: usize,
    whole: &mut Counts,
    enclosing: Option<&'a str>,
    regions: &mut BTreeMap<&'a str, Counts>,
) {
    whole.record_node(node, depth);
    if let Some(identity) = enclosing {
        regions.entry(identity).or_default().record_node(node, depth);
    }
    for operand in node_operands(node).into_iter().flatten() {
        walk_expr(operand, whole, enclosing, regions);
    }
    for operand in node_variadic_operands(node) {
        walk_expr(operand, whole, enclosing, regions);
    }
    let inner = match node {
        Node::Region { generator, .. } => Some(generator.as_str()),
        _ => enclosing,
    };
    if let Node::Region { generator, .. } = node {
        regions.entry(generator.as_str()).or_default();
    }
    for body in child_bodies(node) {
        for child in body {
            walk(child, depth + 1, whole, inner, regions);
        }
    }
}

fn walk_expr<'a>(
    expr: &Expr,
    whole: &mut Counts,
    enclosing: Option<&'a str>,
    regions: &mut BTreeMap<&'a str, Counts>,
) {
    whole.record_expr(expr);
    if let Some(identity) = enclosing {
        regions.entry(identity).or_default().record_expr(expr);
    }
    for child in expr_children(expr).iter() {
        walk_expr(child, whole, enclosing, regions);
    }
}

fn render_histogram(counts: &BTreeMap<&'static str, usize>) -> String {
    if counts.is_empty() {
        return "none".to_string();
    }
    let mut out = String::new();
    for (index, (name, count)) in counts.iter().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        let _ = write!(out, "{name}={count}");
    }
    out
}

fn hex(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Render `(entry point id, section)` pairs into a golden corpus.
pub(crate) fn render_golden<'a>(sections: impl IntoIterator<Item = (&'a str, String)>) -> String {
    let mut out = String::from(HEADER);
    for (id, rendered) in sections {
        let _ = writeln!(out, "{SECTION_MARKER}{id}");
        out.push_str(&rendered);
        if !rendered.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

/// Whether `corpus` carries a section for `id`.
///
/// A golden that stopped naming an entry point silently stopped covering it.
pub(crate) fn golden_contains(corpus: &str, id: &str) -> bool {
    corpus.contains(&format!("{SECTION_MARKER}{id}\n"))
}

/// Compare `actual` against the golden at `path`.
///
/// # Panics
///
/// Panics when the golden is unreadable, or when the structural IR differs. The
/// report names the entry point, the line, and both sides of the first
/// difference, because the decision a reader has to make is whether the IR
/// moved or only a name did, and an opaque digest cannot answer that.
pub(crate) fn assert_matches_golden(path: &Path, actual: &str) {
    let expected = vyre_test_support::read_source_file_bounded(path).unwrap_or_else(|error| {
        panic!(
            "Fix: cannot read the structural IR golden {}: {error}. Run the bless test in this file to create it.",
            path.display()
        )
    });
    if expected == actual {
        return;
    }
    panic!(
        "Fix: the structural IR of a clone-family entry point changed.\n{}\n\
         Restore the IR, or, if the new IR is correct, run the bless test in \
         this file and state in the commit body which entry points changed and why.",
        first_difference(&expected, actual),
    );
}

/// Write `actual` to `path`, creating parent directories.
///
/// # Panics
///
/// Panics when the golden cannot be written.
pub(crate) fn write_golden(path: &Path, actual: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("Fix: cannot create {}: {error}", parent.display()));
    }
    fs::write(path, actual)
        .unwrap_or_else(|error| panic!("Fix: cannot write {}: {error}", path.display()));
}

/// The entry point, line number and both sides of the first differing line.
fn first_difference(expected: &str, actual: &str) -> String {
    let mut section = "<header>".to_string();
    let mut expected_lines = expected.lines();
    let mut actual_lines = actual.lines();
    let mut line = 0usize;
    loop {
        line += 1;
        match (expected_lines.next(), actual_lines.next()) {
            (None, None) => {
                return "The two corpora agree line by line and differ only in trailing bytes."
                    .to_string()
            }
            (left, right) if left == right => {
                if let Some(text) = left.and_then(|text| text.strip_prefix(SECTION_MARKER)) {
                    section = text.to_string();
                }
            }
            (left, right) => {
                return format!(
                    "Entry point `{section}`, golden line {line}:\n  golden: {}\n  actual: {}",
                    left.unwrap_or("<end of golden>"),
                    right.unwrap_or("<end of actual>"),
                );
            }
        }
    }
}
