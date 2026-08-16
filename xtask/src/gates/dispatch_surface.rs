//! What a dispatch trait requires of a backend, and what it hands back.
//!
//! A dispatch trait that declares both an owned-row and a borrowed-row form of
//! the same call has one correct arrangement: the borrowed form is the required
//! method and the owned form is a default that borrows what it was handed. The
//! other arrangement compiles just as well and forces every implementor that
//! binds caller memory to receive rows it must first own, so the default body
//! copies every input byte on a path whose whole purpose is not to.
//!
//! The return side has the same shape of defect. A dispatch that hands back
//! `Vec<Vec<u8>>` allocates one vector per output row on every call, so a trait
//! that returns rows must also offer a form that fills slots the caller keeps,
//! and that form must fill them rather than assign fresh rows through them.
//!
//! Judging call sites cannot see either. A caller holding reusable owned rows
//! and passing `&rows[..n]` copies nothing, and telling it to build a borrowed
//! spine would add an allocation per call; the shape that decides is the
//! trait's, so the trait is what this reads.

use std::path::{Path, PathBuf};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// The row shape a dispatch method takes.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Rows {
    /// `inputs: &[Vec<u8>]`, one owned allocation per row.
    Owned,
    /// `inputs: &[&[u8]]`, caller memory borrowed as a spine of slices.
    Borrowed,
}

/// One method declared inside a trait block.
struct Declaration {
    name: String,
    /// One-based line the `fn` is declared on.
    line: u32,
    /// One-based line the default body starts on, meaningful when it has one.
    body_line: u32,
    /// The input row shape, for a method that takes one.
    rows: Option<Rows>,
    /// Whether the return type carries nested byte rows.
    returns_rows: bool,
    /// Names of the caller-owned byte slots the signature fills.
    slots: Vec<String>,
    /// Lines of the default body, empty when the method is required.
    body: Vec<String>,
}

/// The byte copies a borrowing default has no reason to perform.
const COPIES_BYTES: &[&str] = &["to_vec()", "clone()", "extend_from_slice"];

/// The nested byte rows a dispatch allocates one vector per row to return.
const ROWS: &str = "Vec<Vec<u8>>";

/// The parameter form a caller-owned slot arrives in.
const MUTABLE: &str = "&mut ";

/// The types a caller allocates once and expects a dispatch to refill.
const SLOT_TYPES: &[&str] = &["OutputBuffers", "Vec<u8>"];

/// Owned-row dispatch declared as the method a backend must implement.
pub struct OwnedDispatch;

impl Gate for OwnedDispatch {
    fn name(&self) -> &'static str {
        "hot-path-owned-dispatch"
    }

    fn help(&self) -> &'static str {
        "owned-row dispatch required of backends instead of defaulted over the borrowed form"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut findings = Vec::new();
        let mut traits = 0usize;
        for path in tree.all_rust() {
            if scan::is_test_tree(&path) {
                continue;
            }
            let text = tree.read(&path)?;
            if !text.contains("dispatch_borrowed") {
                continue;
            }
            for declarations in trait_declarations(&text) {
                let dispatch: Vec<&Declaration> = declarations
                    .iter()
                    .filter(|declaration| {
                        declaration.rows.is_some() && declaration.name.starts_with("dispatch")
                    })
                    .collect();
                if dispatch.is_empty() {
                    continue;
                }
                traits += 1;
                findings.extend(pairing_findings(&path, &dispatch));
            }
        }
        let mut report = Report::with_findings(findings);
        report
            .notes
            .push(format!("checked {traits} dispatch trait(s)"));
        Ok(report)
    }
}

/// Nested byte rows returned by a trait that offers no slots to fill instead.
pub struct NestedRows;

impl Gate for NestedRows {
    fn name(&self) -> &'static str {
        "hot-path-nested-rows"
    }

    fn help(&self) -> &'static str {
        "nested byte rows returned by a dispatch trait with no caller-slot form"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut findings = Vec::new();
        let mut traits = 0usize;
        for path in tree.all_rust() {
            if scan::is_test_tree(&path) {
                continue;
            }
            let text = tree.read(&path)?;
            if !text.contains(ROWS) {
                continue;
            }
            for declarations in trait_declarations(&text) {
                if !declarations
                    .iter()
                    .any(|declaration| declaration.returns_rows)
                {
                    continue;
                }
                traits += 1;
                findings.extend(slot_findings(&path, &declarations));
            }
        }
        let mut report = Report::with_findings(findings);
        report
            .notes
            .push(format!("checked {traits} row-returning trait(s)"));
        Ok(report)
    }
}

/// Every production trait block's method declarations, one entry per trait.
///
/// A trait declared inside a `#[cfg(test)]` module is a fixture, including this
/// gate's own examples of the arrangement it forbids, and a rule that reports
/// its own examples can only be quietened by deleting them.
fn trait_declarations(text: &str) -> Vec<Vec<Declaration>> {
    let lines: Vec<&str> = text.lines().collect();
    let test_only = scan::cfg_test_lines(&lines);
    let mut traits = Vec::new();
    let mut index = 0usize;
    while index < lines.len() {
        let Some(open) = trait_block_start(lines[index]) else {
            index += 1;
            continue;
        };
        let end = block_end(&lines, index, open);
        if !test_only.get(index).copied().unwrap_or(false) {
            let declarations = trait_method_declarations(&lines, index + 1, end);
            if !declarations.is_empty() {
                traits.push(declarations);
            }
        }
        index = end + 1;
    }
    traits
}

/// The brace depth a trait declaration opens on its own line, when it is one.
fn trait_block_start(line: &str) -> Option<i32> {
    let code = scan::scan_code(line);
    let trimmed = code.code.trim_start();
    let declaration = trimmed
        .strip_prefix("pub ")
        .unwrap_or(trimmed)
        .strip_prefix("trait ")?;
    declaration.chars().next()?.is_alphabetic().then_some(())?;
    (code.brace_delta > 0).then_some(code.brace_delta)
}

/// The index of the line where a block opened at `start` closes.
fn block_end(lines: &[&str], start: usize, open: i32) -> usize {
    let mut depth = open;
    for (offset, line) in lines.iter().enumerate().skip(start + 1) {
        depth += scan::scan_code(line).brace_delta;
        if depth <= 0 {
            return offset;
        }
    }
    lines.len() - 1
}

/// Every method a trait block declares between `start` and `end`.
fn trait_method_declarations(lines: &[&str], start: usize, end: usize) -> Vec<Declaration> {
    let mut declarations = Vec::new();
    let mut index = start;
    while index < end {
        let Some(name) = declared_method_name(lines[index]) else {
            index += 1;
            continue;
        };
        let declared_at = index;
        let (signature, terminator) = signature_through_terminator(lines, index, end);
        index = terminator + 1;
        let rows = if signature.contains(": &[&[u8]]") {
            Some(Rows::Borrowed)
        } else if signature.contains(": &[Vec<u8>]") {
            Some(Rows::Owned)
        } else {
            None
        };
        let returns_rows = signature
            .rsplit_once("->")
            .is_some_and(|(_, returned)| returned.contains(ROWS));
        let mut body = Vec::new();
        if lines[terminator].trim_end().ends_with('{') {
            let close = block_end(lines, terminator, 1);
            body = lines[terminator + 1..close.min(end)]
                .iter()
                .map(|line| (*line).to_owned())
                .collect();
            index = close + 1;
        }
        declarations.push(Declaration {
            name,
            line: one_based(declared_at),
            body_line: one_based(terminator + 1),
            rows,
            returns_rows,
            slots: slot_parameters(&signature),
            body,
        });
    }
    declarations
}

/// A one-based line number for a zero-based index.
fn one_based(index: usize) -> u32 {
    u32::try_from(index + 1).unwrap_or(u32::MAX)
}

/// The method a line declares, when it declares one.
fn declared_method_name(line: &str) -> Option<String> {
    let code = scan::scan_code(line);
    let mut declaration = code.code.trim_start();
    for prefix in ["pub ", "async ", "unsafe ", "const "] {
        declaration = declaration.strip_prefix(prefix).unwrap_or(declaration);
    }
    let name = declaration
        .strip_prefix("fn ")?
        .split(['(', '<'])
        .next()?
        .trim();
    (!name.is_empty()).then(|| name.to_owned())
}

/// The caller-owned byte slots a signature fills.
///
/// A slot is a `&mut` parameter the caller allocated once and reuses: the buffer
/// set or row vector a dispatch writes into instead of returning fresh rows. A
/// `&mut` timing record or configuration is not a slot.
fn slot_parameters(signature: &str) -> Vec<String> {
    let mut slots = Vec::new();
    let mut rest = signature;
    while let Some(position) = rest.find(MUTABLE) {
        let (declared, tail) = rest.split_at(position);
        rest = &tail[MUTABLE.len()..];
        let declared_type = rest.split([',', ')']).next().unwrap_or_default();
        if !SLOT_TYPES.iter().any(|slot| declared_type.contains(slot)) {
            continue;
        }
        let head = declared.trim_end();
        let head = head.strip_suffix(':').unwrap_or(head).trim_end();
        let name = head
            .rsplit(|character: char| !character.is_alphanumeric() && character != '_')
            .next()
            .unwrap_or_default();
        if !name.is_empty() {
            slots.push(name.to_owned());
        }
    }
    slots
}

/// The whole signature text plus the index of the line that terminates it.
///
/// A signature ends where its body opens or where a required method's semicolon
/// stands. Reading to that point rather than to the return arrow keeps a `where`
/// clause and a return type spelled across several lines inside the signature.
fn signature_through_terminator(lines: &[&str], start: usize, end: usize) -> (String, usize) {
    let mut signature = String::new();
    for (offset, line) in lines.iter().enumerate().take(end).skip(start) {
        let code = scan::scan_code(line);
        signature.push(' ');
        signature.push_str(&code.code);
        let trimmed = code.code.trim_end();
        if trimmed.ends_with(';') || trimmed.ends_with('}') || code.brace_delta > 0 {
            return (signature, offset);
        }
    }
    (signature, start)
}

/// What is wrong with how a trait pairs its owned and borrowed dispatch forms.
fn pairing_findings(path: &Path, declarations: &[&Declaration]) -> Vec<Finding> {
    let mut findings = Vec::new();
    for borrowed in declarations
        .iter()
        .filter(|declaration| declaration.rows == Some(Rows::Borrowed))
    {
        let Some(suffix) = borrowed.name.strip_prefix("dispatch_borrowed") else {
            continue;
        };
        let owned_name = format!("dispatch{suffix}");
        let Some(owned) = declarations.iter().find(|declaration| {
            declaration.name == owned_name && declaration.rows == Some(Rows::Owned)
        }) else {
            continue;
        };
        if owned.body.is_empty() {
            findings.push(required_owned_finding(path, borrowed, owned));
        }
        findings.extend(copy_findings(path, owned));
        findings.extend(copy_findings(path, borrowed));
    }
    findings
}

/// The owned form is what a backend must implement, so the borrowed form is
/// reachable only by staging rows the caller already holds.
fn required_owned_finding(path: &Path, borrowed: &Declaration, owned: &Declaration) -> Finding {
    Finding::at(
        PathBuf::from(path),
        owned.line,
        format!(
            "`{}` is required of every implementor, so `{}` is reached only by owning rows first",
            owned.name, borrowed.name
        ),
        format!(
            "require `{}` instead and give `{}` a default body that borrows its rows",
            borrowed.name, owned.name
        ),
    )
}

/// Findings about a default body that copies the rows it was handed.
///
/// A default exists so an implementor can skip one shape of the call. Copying
/// the rows to reach the other shape is what makes the skip cost the whole
/// input on every dispatch, whichever direction the default points.
fn copy_findings(path: &Path, declaration: &Declaration) -> Vec<Finding> {
    declaration
        .body
        .iter()
        .enumerate()
        .filter(|(_, line)| {
            let code = scan::scan_code(line);
            code.code.contains("inputs") && COPIES_BYTES.iter().any(|copy| code.code.contains(copy))
        })
        .map(|(offset, _)| {
            Finding::at(
                PathBuf::from(path),
                declaration.body_line + u32::try_from(offset).unwrap_or(0),
                format!(
                    "the `{}` default copies the rows it was handed",
                    declaration.name
                ),
                "delegate through a spine of slices so the default costs one pointer per row, \
                 not one copy of every input byte",
            )
        })
        .collect()
}

/// What is wrong with the slot form a row-returning trait offers.
fn slot_findings(path: &Path, declarations: &[Declaration]) -> Vec<Finding> {
    let mut findings = Vec::new();
    let fills_slots = declarations
        .iter()
        .any(|declaration| !declaration.slots.is_empty());
    if !fills_slots {
        if let Some(returning) = declarations
            .iter()
            .find(|declaration| declaration.returns_rows)
        {
            findings.push(missing_slot_finding(path, returning));
        }
    }
    for declaration in declarations {
        findings.extend(overwrite_findings(path, declaration));
    }
    findings
}

/// Returning rows is the only shape this trait offers, so every caller on it
/// allocates one vector per output row on every call.
fn missing_slot_finding(path: &Path, returning: &Declaration) -> Finding {
    Finding::at(
        PathBuf::from(path),
        returning.line,
        format!(
            "`{}` returns freshly allocated byte rows and this trait offers no form that fills slots the caller keeps",
            returning.name
        ),
        format!(
            "declare `{}_into` taking the caller's output buffers and default the returning form over it",
            returning.name
        ),
    )
}

/// Findings about a default body that replaces the slot it was given.
///
/// A slot parameter exists so the rows a caller already allocated are refilled.
/// Assigning through it frees them and hands back fresh ones, which costs the
/// allocation the parameter was added to avoid while compiling and while every
/// assertion about the values still passes.
fn overwrite_findings(path: &Path, declaration: &Declaration) -> Vec<Finding> {
    let mut findings = Vec::new();
    for (offset, line) in declaration.body.iter().enumerate() {
        let code = scan::scan_code(line);
        for slot in &declaration.slots {
            if !assigns_through(&code.code, slot) {
                continue;
            }
            findings.push(Finding::at(
                PathBuf::from(path),
                declaration.body_line + u32::try_from(offset).unwrap_or(0),
                format!(
                    "the `{}` default replaces the caller's `{slot}` instead of filling it",
                    declaration.name
                ),
                "write into the slot the caller passed, so its rows keep the capacity they \
                 arrived with",
            ));
        }
    }
    findings
}

/// Whether `code` assigns a whole new value through `slot`.
fn assigns_through(code: &str, slot: &str) -> bool {
    let mut rest = code;
    while let Some(position) = rest.find('*') {
        rest = &rest[position + 1..];
        let Some(after) = rest.trim_start().strip_prefix(slot) else {
            continue;
        };
        if after.starts_with(|character: char| character.is_alphanumeric() || character == '_') {
            continue;
        }
        if let Some(assigned) = after.trim_start().strip_prefix('=') {
            if !assigned.starts_with('=') {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::fixture_checkout;

    /// The gate's report for a tree holding one driver source file.
    fn run(text: &str) -> Report {
        let (_temporary, root) = fixture_checkout::checkout(&[("vyre-driver/src/backend.rs", text)]);
        OwnedDispatch
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate reads a fixture tree")
    }

    /// The row-returning gate's report for a tree holding one driver source file.
    fn rows(text: &str) -> Report {
        let (_temporary, root) = fixture_checkout::checkout(&[("vyre-driver/src/backend.rs", text)]);
        NestedRows
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate reads a fixture tree")
    }

    #[test]
    fn a_row_returning_trait_that_offers_slots_is_clean() {
        let report = rows(
            r"
pub trait Backend {
    fn dispatch_borrowed(
        &self,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, Error>;

    fn dispatch_borrowed_into(
        &self,
        inputs: &[&[u8]],
        outputs: &mut OutputBuffers,
    ) -> Result<(), Error> {
        let result = self.dispatch_borrowed(inputs)?;
        replace_output_buffers_preserving_slots(result, outputs);
        Ok(())
    }
}
",
        );

        assert!(
            report.findings.is_empty(),
            "a trait that offers a slot form has nothing to report: {:?}",
            report.findings
        );
        assert_eq!(
            report.notes,
            vec!["checked 1 row-returning trait(s)".to_owned()]
        );
    }

    /// WHY: returning rows as the only shape charges every caller one vector per
    /// output row per call, and no call site can opt out of it.
    #[test]
    fn a_row_returning_trait_with_no_slot_form_is_reported() {
        let report = rows(
            r"
pub trait Backend {
    fn dispatch_borrowed(
        &self,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, Error>;
}
",
        );

        assert_eq!(
            report.findings.len(),
            1,
            "one trait without slots is one finding: {:?}",
            report.findings
        );
        assert!(
            report.findings[0].message.contains("`dispatch_borrowed`")
                && report.findings[0].message.contains("no form that fills slots"),
            "the returning method must be named: {:?}",
            report.findings[0]
        );
        assert!(
            report.findings[0].fix.contains("dispatch_borrowed_into"),
            "the fix must name the slot form to declare: {:?}",
            report.findings[0]
        );
    }

    /// WHY: a slot form that assigns through the parameter frees the rows the
    /// caller reused and returns fresh ones, which is the allocation the
    /// parameter exists to avoid. Every value assertion still passes.
    #[test]
    fn a_slot_default_that_replaces_the_caller_rows_is_reported() {
        let report = rows(
            r"
pub trait Backend {
    fn dispatch_borrowed(
        &self,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, Error>;

    fn dispatch_borrowed_into(
        &self,
        inputs: &[&[u8]],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), Error> {
        *outputs = self.dispatch_borrowed(inputs)?;
        Ok(())
    }
}
",
        );

        assert_eq!(
            report.findings.len(),
            1,
            "one replacement is one finding: {:?}",
            report.findings
        );
        assert!(
            report.findings[0]
                .message
                .contains("replaces the caller's `outputs` instead of filling it"),
            "the replaced slot must be named: {:?}",
            report.findings[0]
        );
    }

    /// WHY: a comparison against the slot reads it and replaces nothing, so a
    /// rule that searched for the slot name beside an equals sign would report
    /// the assertion a default makes about what it wrote.
    #[test]
    fn a_default_that_compares_the_slot_is_clean() {
        let report = rows(
            r"
pub trait Backend {
    fn dispatch_borrowed(
        &self,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, Error>;

    fn dispatch_borrowed_into(
        &self,
        inputs: &[&[u8]],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), Error> {
        let result = self.dispatch_borrowed(inputs)?;
        replace_output_buffers_preserving_slots(result, outputs);
        debug_assert!(*outputs == expected(inputs), 'rows survive the refill');
        Ok(())
    }
}
",
        );

        assert!(
            report.findings.is_empty(),
            "reading the slot is not replacing it: {:?}",
            report.findings
        );
    }

    /// WHY: a trait that only ever fills slots names the row type in a parameter,
    /// so reading the whole signature for the type rather than the return would
    /// report the shape that is already correct.
    #[test]
    fn a_slot_only_trait_is_not_read_as_returning_rows() {
        let report = rows(
            r"
pub trait Backend {
    fn dispatch_borrowed_into(
        &self,
        inputs: &[&[u8]],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), Error>;
}
",
        );

        assert!(
            report.findings.is_empty(),
            "a slot-only trait returns no rows: {:?}",
            report.findings
        );
        assert_eq!(
            report.notes,
            vec!["checked 0 row-returning trait(s)".to_owned()]
        );
    }

    /// WHY: this gate's own fixtures are traits with no slot form, so a rule that
    /// read them could only be quietened by deleting the cases that prove it
    /// fails.
    #[test]
    fn a_trait_declared_under_cfg_test_is_a_fixture() {
        let report = rows(
            r"
#[cfg(test)]
mod tests {
    pub trait Backend {
        fn dispatch_borrowed(
            &self,
            inputs: &[&[u8]],
        ) -> Result<Vec<Vec<u8>>, Error>;
    }
}
",
        );

        assert!(
            report.findings.is_empty(),
            "a fixture trait states no production contract: {:?}",
            report.findings
        );
        assert_eq!(
            report.notes,
            vec!["checked 0 row-returning trait(s)".to_owned()]
        );
    }

    const BORROWED_IS_REQUIRED: &str = r"
pub trait Backend {
    fn dispatch(
        &self,
        inputs: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, Error> {
        let borrowed = borrow(inputs)?;
        self.dispatch_borrowed(&borrowed)
    }

    fn dispatch_borrowed(
        &self,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, Error>;
}
";

    #[test]
    fn a_trait_that_requires_the_borrowed_form_is_clean() {
        let report = run(BORROWED_IS_REQUIRED);

        assert!(
            report.findings.is_empty(),
            "the arrangement the rule prescribes must report nothing: {:?}",
            report.findings
        );
        assert_eq!(report.notes, vec!["checked 1 dispatch trait(s)".to_owned()]);
    }

    #[test]
    fn a_trait_that_requires_the_owned_form_reports_the_requirement_and_the_copy() {
        let report = run(
            r"
pub trait Backend {
    fn dispatch(
        &self,
        inputs: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, Error>;

    fn dispatch_borrowed(
        &self,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, Error> {
        let owned: Vec<Vec<u8>> = inputs.iter().map(|row| row.to_vec()).collect();
        self.dispatch(&owned)
    }
}
",
        );

        let messages: Vec<&str> = report
            .findings
            .iter()
            .map(|finding| finding.message.as_str())
            .collect();
        assert!(
            messages
                .iter()
                .any(|message| message.contains("`dispatch` is required of every implementor")),
            "the required owned form must be named: {messages:?}"
        );
        assert!(
            messages.iter().any(|message| message
                .contains("the `dispatch_borrowed` default copies the rows it was handed")),
            "the copy the requirement forces must be named: {messages:?}"
        );
    }

    /// WHY: an asynchronous dispatch is an optional capability, so both forms of
    /// it default and neither is required. A rule that demanded the borrowed one
    /// be required would report every optional pair, and the only way to satisfy
    /// it would be to force every implementor to write an async path.
    #[test]
    fn an_optional_pair_where_both_forms_default_is_clean() {
        let report = run(
            r"
pub trait Backend {
    fn dispatch_async(
        &self,
        inputs: &[Vec<u8>],
    ) -> Result<Pending, Error> {
        ready(self.dispatch(inputs)?)
    }

    fn dispatch_borrowed_async(
        &self,
        inputs: &[&[u8]],
    ) -> Result<Pending, Error> {
        ready(self.dispatch_borrowed(inputs)?)
    }
}
",
        );

        assert!(
            report.findings.is_empty(),
            "an optional capability declares no requirement to invert: {:?}",
            report.findings
        );
    }

    #[test]
    fn a_borrowing_default_that_copies_rows_is_reported_at_the_copy() {
        let report = run(
            r"
pub trait Backend {
    fn dispatch(
        &self,
        inputs: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, Error> {
        let staged: Vec<Vec<u8>> = inputs.iter().map(|row| row.to_vec()).collect();
        self.dispatch_borrowed(&borrow(&staged))
    }

    fn dispatch_borrowed(
        &self,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, Error>;
}
",
        );

        assert_eq!(
            report.findings.len(),
            1,
            "one copy is one finding: {:?}",
            report.findings
        );
        assert!(
            report.findings[0]
                .message
                .contains("copies the rows it was handed"),
            "the copy must be named: {:?}",
            report.findings[0]
        );
    }

    #[test]
    fn an_owned_only_dispatch_trait_is_left_alone() {
        let report = run(
            r"
pub trait ProgramDispatcher {
    fn dispatch(
        &self,
        inputs: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, Error>;
}

fn borrow(rows: &[Vec<u8>]) -> Vec<&[u8]> {
    rows.iter().map(Vec::as_slice).collect()
}
",
        );

        assert!(
            report.findings.is_empty(),
            "a trait with no borrowed form has no pair to arrange: {:?}",
            report.findings
        );
        assert_eq!(
            report.notes,
            vec!["checked 0 dispatch trait(s)".to_owned()],
            "a file that never spells the borrowed form is not read for traits"
        );
    }

    #[test]
    fn an_impl_block_is_not_read_as_a_trait_declaration() {
        let report = run(
            r"
impl Backend for Cuda {
    fn dispatch(
        &self,
        inputs: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, Error> {
        self.dispatch_borrowed(&borrow(inputs))
    }
}
",
        );

        assert!(
            report.findings.is_empty(),
            "an implementation states no requirement: {:?}",
            report.findings
        );
    }
}
