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
//! The full rendering is not checked in: a direct-path fixture can unroll to
//! tens of thousands of lines, and a corpus of them would be megabytes.
//! Regenerate it locally from [`render_structural_ir`] when a digest moves and
//! the histograms do not say enough.
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
//!
//! # One harness, several consumers
//!
//! [`StructuralIrGolden`] carries the digest domain separator and the golden
//! path, so each crate pins its own corpus at its own location and two crates'
//! goldens cannot collide through a shared digest.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use vyre_foundation::hashing::domain_digest;
use vyre_foundation::ir::{expr_variant_name, node_variant_name, Expr, Node, Program};
use vyre_foundation::visit::{child_bodies, expr_children, node_operands, node_variadic_operands};

/// Line that opens each entry point's section in a golden.
const SECTION_MARKER: &str = "===== ";

/// Full structural rendering of `program`, canonicalized.
///
/// The digest is taken over this. It is not checked in; see the module docs.
#[must_use]
pub fn render_structural_ir(program: &Program) -> String {
    let canonical = program.canonicalized();
    format!(
        "buffers:\n{:#?}\n\nentry:\n{:#?}\n",
        canonical.buffers(),
        canonical.entry()
    )
}

/// Whether `corpus` carries a section for `id`.
///
/// A golden that stopped naming an entry point silently stopped covering it.
#[must_use]
pub fn golden_contains(corpus: &str, id: &str) -> bool {
    corpus.contains(&format!("{SECTION_MARKER}{id}\n"))
}

/// Write `actual` to `path`, creating parent directories.
///
/// # Panics
///
/// Panics when the golden cannot be written.
pub fn write_golden(path: &Path, actual: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("Fix: cannot create {}: {error}", parent.display()));
    }
    fs::write(path, actual)
        .unwrap_or_else(|error| panic!("Fix: cannot write {}: {error}", path.display()));
}

/// One crate's structural IR golden: a digest domain and a corpus path.
pub struct StructuralIrGolden {
    domain: &'static str,
    subject: &'static str,
    path: PathBuf,
}

impl StructuralIrGolden {
    /// Pin a corpus at `path`, digesting under `domain`.
    ///
    /// `domain` separates one crate's digests from another's, so the same
    /// program rendered by two consumers produces two values and a section
    /// cannot be copied between corpora. `subject` names the families the
    /// corpus covers and appears in the golden header.
    #[must_use]
    pub fn new(domain: &'static str, subject: &'static str, path: PathBuf) -> Self {
        Self {
            domain,
            subject,
            path,
        }
    }

    /// Where the corpus is pinned.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Render `(entry point id, program)` pairs into a golden corpus.
    #[must_use]
    pub fn render<'a>(
        &self,
        entry_points: impl IntoIterator<Item = (&'a str, Program)>,
    ) -> String {
        let mut out = self.header();
        for (id, program) in entry_points {
            let _ = writeln!(out, "{SECTION_MARKER}{id}");
            let rendered = self.render_section(&program);
            out.push_str(&rendered);
            if !rendered.ends_with('\n') {
                out.push('\n');
            }
        }
        out
    }

    /// Golden section body for `program`.
    #[must_use]
    pub fn render_section(&self, program: &Program) -> String {
        let canonical = program.canonicalized();
        let digest = domain_digest(
            self.domain.as_bytes(),
            render_structural_ir(program).as_bytes(),
        );

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

    /// Compare `actual` against the pinned corpus.
    ///
    /// # Panics
    ///
    /// Panics when the golden is unreadable, or when the structural IR
    /// differs. The report names each entry point that moved, which class of
    /// change moved it, and both sides of every field that differs, because
    /// the decision a reader has to make is whether the IR moved or only a
    /// name did, and an opaque digest cannot answer that.
    pub fn assert_matches(&self, actual: &str) {
        let expected = crate::read_source_file_bounded(&self.path).unwrap_or_else(|error| {
            panic!(
                "Fix: cannot read the structural IR golden {}: {error}. Run the bless test in this file to create it.",
                self.path.display()
            )
        });
        if expected == actual {
            return;
        }
        panic!(
            "Fix: the structural IR of a clone-family entry point changed.\n{}\n\
             Restore the IR, or, if the new IR is correct, run the bless test in \
             this file and state in the commit body which entry points changed and why.",
            report_differences(&expected, actual),
        );
    }

    /// Rewrite the pinned corpus with `actual`.
    ///
    /// # Panics
    ///
    /// Panics when the golden cannot be written.
    pub fn bless(&self, actual: &str) {
        write_golden(&self.path, actual);
    }

    /// Header written above a golden, naming what it holds and how to
    /// regenerate it.
    fn header(&self) -> String {
        format!(
            "\
# Structural IR golden for {subject}.
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
",
            subject = self.subject
        )
    }
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
        regions
            .entry(identity)
            .or_default()
            .record_node(node, depth);
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

/// Field groups a section is compared by, in the order a section writes them.
const GROUPS: [&str; 7] = [
    "digest",
    "counts",
    "depth",
    "node_kinds",
    "expr_kinds",
    "buffers",
    "regions",
];

/// Differing lines per group beyond this many are summarized as a count.
const MAX_LINES_PER_GROUP: usize = 4;

/// Every entry point that moved, the class of change that moved it, and both
/// sides of each field that differs.
///
/// Comparison is per section and per field group rather than line by line, so
/// a section that gained or lost a line does not shift every later section out
/// of alignment and turn one change into a whole-corpus diff.
fn report_differences(expected: &str, actual: &str) -> String {
    let (expected_header, expected_sections) = split_sections(expected);
    let (actual_header, actual_sections) = split_sections(actual);

    let mut out = String::new();
    if expected_header != actual_header {
        out.push_str("The golden header differs. Re-bless it.\n");
    }

    for (id, expected_lines) in &expected_sections {
        let Some((_, actual_lines)) = actual_sections.iter().find(|(other, _)| other == id) else {
            let _ = writeln!(out, "Entry point `{id}` is no longer rendered at all.");
            continue;
        };
        report_section(&mut out, id, expected_lines, actual_lines);
    }
    for (id, _) in &actual_sections {
        if !expected_sections.iter().any(|(other, _)| other == id) {
            let _ = writeln!(
                out,
                "Entry point `{id}` is rendered but the golden does not carry it."
            );
        }
    }

    if out.is_empty() {
        out.push_str("The two corpora agree field by field and differ only in trailing bytes.\n");
    }
    out
}

/// One section's report, or nothing when the section is unchanged.
fn report_section(out: &mut String, id: &str, expected: &[&str], actual: &[&str]) {
    let expected_groups = group_lines(expected);
    let actual_groups = group_lines(actual);
    let changed: Vec<&'static str> = GROUPS
        .iter()
        .copied()
        .filter(|group| expected_groups.get(*group) != actual_groups.get(*group))
        .collect();
    if changed.is_empty() {
        return;
    }

    let _ = writeln!(
        out,
        "Entry point `{id}`: {}.{}",
        class_of(&changed),
        localization(&expected_groups, &actual_groups),
    );
    for group in changed {
        let _ = writeln!(out, "  {group}");
        let empty: Vec<&str> = Vec::new();
        let left = expected_groups.get(group).unwrap_or(&empty);
        let right = actual_groups.get(group).unwrap_or(&empty);
        let mut shown = 0usize;
        let mut suppressed = 0usize;
        for index in 0..left.len().max(right.len()) {
            let golden = left.get(index).copied();
            let measured = right.get(index).copied();
            if golden == measured {
                continue;
            }
            if shown == MAX_LINES_PER_GROUP {
                suppressed += 1;
                continue;
            }
            shown += 1;
            let _ = writeln!(out, "    golden: {}", golden.unwrap_or("<absent>"));
            let _ = writeln!(out, "    actual: {}", measured.unwrap_or("<absent>"));
        }
        if suppressed > 0 {
            let _ = writeln!(out, "    and {suppressed} further differing line(s)");
        }
    }
}

/// Split a corpus into its header lines and its `(entry point id, lines)`
/// sections.
fn split_sections(corpus: &str) -> (Vec<&str>, Vec<(&str, Vec<&str>)>) {
    let mut header = Vec::new();
    let mut sections: Vec<(&str, Vec<&str>)> = Vec::new();
    for line in corpus.lines() {
        if let Some(id) = line.strip_prefix(SECTION_MARKER) {
            sections.push((id, Vec::new()));
        } else if let Some((_, lines)) = sections.last_mut() {
            lines.push(line);
        } else {
            header.push(line);
        }
    }
    (header, sections)
}

/// Bucket a section's lines by the field each one belongs to.
fn group_lines<'a>(lines: &[&'a str]) -> BTreeMap<&'static str, Vec<&'a str>> {
    let mut out: BTreeMap<&'static str, Vec<&'a str>> = BTreeMap::new();
    let mut list: Option<&'static str> = None;
    for line in lines {
        let group = if line.starts_with("digest ") {
            list = None;
            "digest"
        } else if line.starts_with("nodes ") {
            list = None;
            "counts"
        } else if line.starts_with("depth ") {
            list = None;
            "depth"
        } else if line.starts_with("node_kinds ") {
            list = None;
            "node_kinds"
        } else if line.starts_with("expr_kinds ") {
            list = None;
            "expr_kinds"
        } else if *line == "buffers:" {
            list = Some("buffers");
            "buffers"
        } else if *line == "regions:" {
            list = Some("regions");
            "regions"
        } else {
            list.unwrap_or("digest")
        };
        out.entry(group).or_default().push(line);
    }
    out
}

/// Which class of change the set of moved fields is.
///
/// The order is by how much a reader has to do about it: an ABI move first, a
/// node population move next, and a digest that moved with every count held
/// last, because that one is an operand or an order and nothing else can
/// produce it.
fn class_of(changed: &[&'static str]) -> &'static str {
    if changed.contains(&"buffers") {
        "the buffer roster changed, which is the program's ABI"
    } else if changed.contains(&"counts") || changed.contains(&"node_kinds") {
        "a node was added or dropped"
    } else if changed.contains(&"expr_kinds") {
        "an operand expression was added or dropped"
    } else if changed.contains(&"depth") {
        "the nesting depth changed"
    } else if changed.contains(&"regions") {
        "a region's node population changed while the whole-program counts held"
    } else {
        "an operand or a data dependence order changed; every count, histogram and buffer held"
    }
}

/// Which region identities the change reaches, when it reaches some.
fn localization(
    expected: &BTreeMap<&'static str, Vec<&str>>,
    actual: &BTreeMap<&'static str, Vec<&str>>,
) -> String {
    let empty: Vec<&str> = Vec::new();
    let left = expected.get("regions").unwrap_or(&empty);
    let right = actual.get("regions").unwrap_or(&empty);
    if left == right {
        return String::new();
    }
    let mut identities: BTreeSet<&str> = BTreeSet::new();
    for index in 0..left.len().max(right.len()) {
        let golden = left.get(index).copied();
        let measured = right.get(index).copied();
        if golden == measured {
            continue;
        }
        for line in [golden, measured].into_iter().flatten() {
            if let Some(identity) = line.trim_start().split_whitespace().next() {
                identities.insert(identity);
            }
        }
    }
    if identities.is_empty() {
        return String::new();
    }
    format!(
        " The change reaches region(s) {}.",
        identities
            .into_iter()
            .map(|identity| format!("`{identity}`"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}
