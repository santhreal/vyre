//! Contracts that may not drift without someone deciding they should.
//!
//! Four rules live here. Adding a backend stays one crate plus link-time
//! registration. Ordinary outputs stage through the readback ring. Every field of
//! the program type is either on the wire or explicitly transient. And the seven
//! frozen declarations match their snapshots byte for byte, because a change to
//! one of them is a major version event.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// Concrete backend crates, each owning its own implementation.
const BACKENDS: &[&str] = &[
    "vyre-driver-cuda",
    "vyre-driver-wgpu",
    "vyre-driver-metal",
    "vyre-driver-spirv",
    "vyre-driver-reference",
];

/// Backend identifiers the core registry must not name.
const BACKEND_IDS: &[&str] = &["\"cuda\"", "\"wgpu\"", "\"spirv\"", "\"metal\"", "\"dxil\""];

/// Adding a backend is one crate plus inventory submissions.
pub struct BackendExtension;

impl Gate for BackendExtension {
    fn name(&self) -> &'static str {
        "backend-extension"
    }

    fn help(&self) -> &'static str {
        "the core registry stays generic and each backend registers itself"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        const INVENTORY: &str = "vyre-driver/src/backend/registry/inventory_streams.rs";
        const ACQUIRE: &str = "vyre-driver/src/backend/registry/acquire.rs";

        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();

        let inventory = tree.read(INVENTORY)?;
        for (needle, message) in [
            (
                "inventory::collect!(BackendRegistration);",
                "BackendRegistration is not an inventory collection",
            ),
            (
                "inventory::collect!(BackendPrecedence);",
                "BackendPrecedence is not an inventory collection",
            ),
            (
                "inventory::collect!(BackendCapability);",
                "BackendCapability is not an inventory collection",
            ),
            (
                "LazyLock<Result<BackendRegistry, BackendError>>",
                "the registry is not frozen through one fallible process-wide value",
            ),
            (
                "inventory::iter::<BackendRegistration>",
                "the registry is not populated from the registration inventory",
            ),
        ] {
            if !inventory.contains(needle) {
                report.find(Finding::in_file(
                    INVENTORY,
                    message,
                    format!("restore `{needle}`; a backend registers itself at link time"),
                ));
            }
        }

        let acquire = tree.read(ACQUIRE)?;
        for (needle, message) in [
            (
                "registered_backends_by_precedence_slice",
                "backend acquisition does not route through the precedence-sorted frozen slice",
            ),
            (
                "backend_dispatches",
                "preferred backend acquisition does not consult dispatch metadata",
            ),
        ] {
            if !acquire.contains(needle) {
                report.find(Finding::in_file(
                    ACQUIRE,
                    message,
                    format!("restore `{needle}` so acquisition reads the frozen registry"),
                ));
            }
        }

        let registry = tree.rust(&["vyre-driver/src/backend/registry"])?;
        for hit in tree.hits(&registry, |line| scan::contains_any(line, BACKEND_IDS))? {
            report.find(Finding::at(
                hit.file,
                hit.line,
                format!("core registry names a concrete backend: {}", hit.text),
                "derive the identifier from the registration the backend submits; adding a \
                 backend must not require editing the core registry",
            ));
        }

        for backend in BACKENDS {
            let manifest = format!("{backend}/Cargo.toml");
            if !tree.exists(&manifest) {
                report.find(Finding::in_file(
                    manifest,
                    "backend crate manifest is missing",
                    "restore the crate, or delete it from the backend list in this gate",
                ));
                continue;
            }
            let text = tree.read(&manifest)?;
            if !text.contains("vyre-driver") {
                report.find(Finding::in_file(
                    manifest.clone(),
                    "backend crate does not depend on vyre-driver",
                    "depend on vyre-driver instead of editing core registry code",
                ));
            }
            if !(text.contains("inventory.workspace") || text.contains("inventory =")) {
                report.find(Finding::in_file(
                    manifest.clone(),
                    "backend crate does not depend on inventory",
                    "depend on inventory so the backend registers itself at link time",
                ));
            }

            let sources = tree.rust(&[&format!("{backend}/src")])?;
            for (message, matcher) in backend_source_requirements() {
                let found = !tree
                    .hits(&sources, |line| matcher(line))?
                    .is_empty();
                if !found {
                    report.find(Finding::in_file(
                        format!("{backend}/src"),
                        format!("{backend} {message}"),
                        "keep a backend one crate that implements the backend trait and \
                         submits its own registration, precedence and capability records",
                    ));
                }
            }
        }

        Ok(report)
    }
}

/// One thing a backend crate's own sources must contain: the sentence a missing
/// one reads as, and the line predicate that finds it.
type SourceRequirement = (&'static str, fn(&str) -> bool);

/// What every backend crate's own sources must contain.
fn backend_source_requirements() -> Vec<SourceRequirement> {
    vec![
        (
            "does not implement the backend trait in its own crate",
            |line: &str| line.contains("impl ") && line.contains("VyreBackend for"),
        ),
        (
            "does not submit backend metadata through inventory",
            |line: &str| followed_by(line, "inventory::submit!", '{'),
        ),
        ("does not submit BackendRegistration", |line: &str| {
            followed_by(line, "BackendRegistration", '{')
        }),
        ("does not submit BackendPrecedence", |line: &str| {
            followed_by(line, "BackendPrecedence", '{')
        }),
        ("does not submit BackendCapability", |line: &str| {
            followed_by(line, "BackendCapability", '{')
        }),
        ("does not advertise supported_ops", |line: &str| {
            followed_by(line, "supported_ops", ':')
        }),
    ]
}

/// Whether a needle is followed, after optional whitespace, by one character.
fn followed_by(line: &str, needle: &str, terminator: char) -> bool {
    let mut from = 0;
    while let Some(at) = line[from..].find(needle) {
        let after = from + at + needle.len();
        if line[after..].trim_start().starts_with(terminator) {
            return true;
        }
        from += at + 1;
        if from >= line.len() {
            break;
        }
    }
    false
}

/// Ordinary outputs stage through the size-classed readback ring.
pub struct ReadbackRing;

impl Gate for ReadbackRing {
    fn name(&self) -> &'static str {
        "readback-ring"
    }

    fn help(&self) -> &'static str {
        "direct dispatch stages ordinary outputs through readback ring slots"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        const RECORD: &str = "vyre-driver-wgpu/src/engine/record_and_readback/mod.rs";
        const RECORD_MODULES: &str = "vyre-driver-wgpu/src/engine/record_and_readback";
        const ARENA: &str = "vyre-driver-wgpu/src/lib.rs";
        /// Each of these is load-bearing for routing an output through a ring
        /// slot. Absence means the routing was removed or renamed.
        const REQUIRED: &[&str] = &[
            "readback_rings:",
            "SubmittedReadback::Ring",
            ".record_copy(",
            ".arm_ticket(",
            ".with_mapped_ticket(",
        ];

        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let sources = tree.rust(&[RECORD, RECORD_MODULES])?;
        for needle in REQUIRED {
            if tree.hits(&sources, |line| line.contains(needle))?.is_empty() {
                report.find(Finding::in_file(
                    RECORD,
                    format!("the record and readback modules no longer contain `{needle}`"),
                    "route ordinary output readbacks through ring slots before falling back \
                     to pooled staging",
                ));
            }
        }
        if !tree.read(ARENA)?.contains("ReadbackRingSet::new()") {
            report.find(Finding::in_file(
                ARENA,
                "the dispatch arena does not own a readback ring set",
                "keep the rings in the backend dispatch arena so hot dispatches reuse slots",
            ));
        }
        // A third rule used to live here as a pattern spanning a loop body, which
        // a line-based search can never match, so it fired on no tree. It is not
        // replaced: the pooled per-output loop in the staging module is the
        // legitimate fallback taken when no ring set is supplied, and which branch
        // the code sits in is not a line property. Two GPU tests in
        // vyre-driver-wgpu/src/pipeline/tests/readback_ring_contracts.rs own it by
        // exercising both branches.
        report.note(
            "the branch-selection invariant is owned by the two readback ring GPU tests, \
             not by this scan",
        );
        Ok(report)
    }
}

/// Every field of the program type is on the wire or explicitly transient.
pub struct ProgramWireFields;

impl Gate for ProgramWireFields {
    fn name(&self) -> &'static str {
        "program-wire-fields"
    }

    fn help(&self) -> &'static str {
        "program fields that are neither serialized nor declared transient"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        const ENCODE: &str = "vyre-foundation/src/serial/wire/encode/to_wire/mod.rs";
        const DECODE: &str = "vyre-foundation/src/serial/wire/decode/from_wire/mod.rs";
        const SERIALIZED: &[&str] = &[
            "entry_op_id",
            "buffers",
            "workgroup_size",
            "entry",
            "non_composable_with_self",
        ];
        /// Cache and provenance state, reset on invalidation and read by no
        /// encoder. Serializing any of it would put mutable local state into the
        /// wire identity.
        const TRANSIENT: &[&str] = &[
            "buffer_index",
            "hash",
            "validation_set",
            "structural_validated",
            "structural_validation_fingerprint",
            "mutation_provenance",
            "fingerprint",
            "normalized_cache_digest",
            "output_buffer_index",
            "has_indirect_dispatch",
            "stats",
        ];

        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();

        // The declaration is located rather than named. This gate once named a
        // path that no longer existed, failed on it, and could not be wired.
        let candidates = tree.rust(&["vyre-foundation/src"])?;
        let declaration = tree
            .hits(
                &candidates
                    .into_iter()
                    .filter(|path| !path.to_string_lossy().contains("/tests/"))
                    .collect::<Vec<PathBuf>>(),
                |line| followed_by(line, "pub struct Program", '{'),
            )?
            .into_iter()
            .next();
        let Some(declaration) = declaration else {
            return Err(GateError::new(
                "no file declares `pub struct Program`",
                "the program type was renamed or removed; repoint this gate at it",
            ));
        };

        let core = tree.read(&declaration.file)?;
        let encode = tree.read(ENCODE)?;
        let decode = tree.read(DECODE)?;

        for field in SERIALIZED {
            if !core.contains(field) {
                report.find(Finding::in_file(
                    declaration.file.clone(),
                    format!("serialized field `{field}` is gone from the program type"),
                    "restore the field, or move it out of the serialized list in this gate \
                     in the same change that removes it from the wire",
                ));
            }
            if !encode.contains(field) && !decode.contains(field) {
                report.find(Finding::in_file(
                    ENCODE,
                    format!("serialized field `{field}` is named by neither encode nor decode"),
                    "wire the field through encode and decode, or declare it transient",
                ));
            }
        }

        let known: BTreeSet<&str> = SERIALIZED.iter().chain(TRANSIENT).copied().collect();
        for (number, field) in program_fields(&core, declaration.line) {
            if !known.contains(field.as_str()) {
                report.find(Finding::at(
                    declaration.file.clone(),
                    number,
                    format!("program field `{field}` is neither serialized nor declared transient"),
                    "wire the field through encode and decode in the same change, or add it \
                     to the transient list in this gate with a cache-only rationale",
                ));
            }
        }

        Ok(report)
    }
}

/// The public fields declared in the program struct, with their line numbers.
fn program_fields(text: &str, declaration_line: u32) -> Vec<(u32, String)> {
    let mut fields = Vec::new();
    let mut inside = false;
    for (number, line) in scan::numbered(text) {
        if !inside {
            if number == declaration_line {
                inside = true;
            }
            continue;
        }
        if line.starts_with('}') {
            break;
        }
        let trimmed = line.trim_start();
        let Some(rest) = trimmed
            .strip_prefix("pub(crate) ")
            .or_else(|| trimmed.strip_prefix("pub "))
        else {
            continue;
        };
        let end = rest
            .find(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
            .unwrap_or(rest.len());
        let name = &rest[..end];
        if name.is_empty() || !rest[end..].trim_start().starts_with(':') {
            continue;
        }
        fields.push((number, name.to_string()));
    }
    fields
}

/// The seven frozen declarations match their snapshots.
pub struct FrozenContracts;

impl Gate for FrozenContracts {
    fn name(&self) -> &'static str {
        "frozen-contracts"
    }

    fn help(&self) -> &'static str {
        "frozen declarations against their committed snapshots"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        /// Name, source file, and the declaration keyword to extract.
        const CONTRACTS: &[(&str, &str, &str)] = &[
            (
                "VyreBackend",
                "vyre-driver/src/backend/vyre_backend.rs",
                "pub trait VyreBackend",
            ),
            (
                "ExprVisitor",
                "vyre-foundation/src/visit/expr_visitor/mod.rs",
                "pub trait ExprVisitor",
            ),
            (
                "Lowerable",
                "vyre-driver/src/backend/lowering.rs",
                "pub trait LowerableOp",
            ),
            (
                "AlgebraicLaw",
                "vyre-spec/src/algebraic_law.rs",
                "pub enum AlgebraicLaw",
            ),
            (
                "EnforceGate",
                "vyre-driver/src/registry/enforce.rs",
                "pub trait EnforceGate",
            ),
            (
                "MutationClass",
                "vyre-driver/src/registry/mutation.rs",
                "pub enum MutationClass",
            ),
            (
                "PassBoundaryClass",
                "vyre-foundation/src/optimizer/mod.rs",
                "pub enum PassBoundaryClass",
            ),
        ];
        const SNAPSHOTS: &str = "docs/frozen-traits";

        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        for (name, file, keyword) in CONTRACTS {
            let snapshot = format!("{SNAPSHOTS}/{name}.txt");
            if !tree.exists(file) {
                report.find(Finding::in_file(
                    *file,
                    format!("frozen contract `{name}` has no source file"),
                    "restore the declaration, or record the removal as a major version event",
                ));
                continue;
            }
            let Some(current) = extract_declaration(&tree.read(file)?, keyword) else {
                report.find(Finding::in_file(
                    *file,
                    format!("frozen contract `{name}` is not declared as `{keyword}`"),
                    "restore the declaration under its frozen name, or record the rename as \
                     a major version event",
                ));
                continue;
            };
            if ctx.write {
                let path = tree.absolute(&snapshot);
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).map_err(|error| {
                        GateError::new(
                            format!("cannot create `{}`: {error}", parent.display()),
                            "check the checkout is writable",
                        )
                    })?;
                }
                fs::write(&path, &current).map_err(|error| {
                    GateError::new(
                        format!("cannot write `{snapshot}`: {error}"),
                        "check the checkout is writable",
                    )
                })?;
                report.note(format!("refreshed {snapshot}"));
                continue;
            }
            if !tree.exists(&snapshot) {
                report.find(Finding::in_file(
                    snapshot.clone(),
                    format!("frozen contract `{name}` has no snapshot"),
                    "run the gate with --write and review the snapshot before committing it",
                ));
                continue;
            }
            if tree.read(&snapshot)? != current {
                report.find(Finding::in_file(
                    *file,
                    format!("frozen contract `{name}` no longer matches its snapshot"),
                    "if the change is intended, refresh the snapshot with --write and bump \
                     the major version, because a frozen contract change is a major event",
                ));
            }
        }
        for orphan in orphan_snapshots(&tree.absolute(SNAPSHOTS), SNAPSHOTS, CONTRACTS)? {
            report.find(Finding::in_file(
                orphan.clone(),
                format!("`{orphan}` snapshots a declaration this gate does not freeze"),
                "add the declaration to the frozen set, or delete the snapshot, because a \
                 snapshot nothing compares against freezes nothing",
            ));
        }
        Ok(report)
    }
}

/// Snapshot files under `directory` that no frozen contract claims.
///
/// A snapshot is the only durable record of a frozen declaration, so one left
/// behind by a deleted row reads as coverage that no longer exists.
fn orphan_snapshots(
    directory_path: &Path,
    directory: &str,
    contracts: &[(&str, &str, &str)],
) -> Result<Vec<String>, GateError> {
    let claimed: BTreeSet<String> = contracts
        .iter()
        .map(|(name, _, _)| format!("{directory}/{name}.txt"))
        .collect();
    let mut orphans = Vec::new();
    let entries = fs::read_dir(directory_path).map_err(|error| {
        GateError::new(
            format!("cannot list `{directory}`: {error}"),
            "restore the snapshot directory, because the frozen set is stored there",
        )
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            GateError::new(
                format!("cannot read an entry of `{directory}`: {error}"),
                "check the checkout is readable",
            )
        })?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".txt") {
            continue;
        }
        let path = format!("{directory}/{name}");
        if !claimed.contains(&path) {
            orphans.push(path);
        }
    }
    orphans.sort();
    Ok(orphans)
}

/// The declaration of `keyword`: the item line, its signatures, and its closer.
///
/// The block ends where its braces balance. Three things are left out, because
/// none of them is the frozen thing and each one moves on ordinary work: a
/// default method body, a doc or code comment, and blank space. A snapshot that
/// moved when a default body was refactored or a doc link was repointed would
/// report a version event on every such edit, and a reader who has refreshed it
/// twice for nothing stops reading it at all.
///
/// Braces are counted on the code alone. A comment is prose and a string is
/// data, so neither nests the declaration.
fn extract_declaration(text: &str, keyword: &str) -> Option<String> {
    let mut collected = String::new();
    let mut depth = 0_i32;
    let mut inside = false;
    let mut open = false;
    for line in text.lines() {
        if !inside {
            if !line.contains(keyword) {
                continue;
            }
            inside = true;
        }
        let trimmed = line.trim_start();
        let prose = trimmed.is_empty() || scan::is_comment(trimmed);
        let code = if prose {
            String::new()
        } else {
            scan::mask_literals(trimmed)
        };
        let opened = i32::try_from(code.matches('{').count()).unwrap_or(i32::MAX);
        let closed = i32::try_from(code.matches('}').count()).unwrap_or(i32::MAX);
        if !prose && depth <= 1 {
            if depth == 1 && opened > closed {
                collected.push_str(&signature_of(trimmed));
            } else {
                collected.push_str(trimmed);
            }
            collected.push('\n');
        }
        depth += opened - closed;
        if depth > 0 {
            open = true;
        } else if open {
            return Some(collected);
        }
    }
    if inside && !collected.is_empty() {
        Some(collected)
    } else {
        None
    }
}

/// A signature line with its body opener replaced by a terminator.
///
/// `fn probe(&self) -> bool {` reads as `fn probe(&self) -> bool;`, which is the
/// part of it that callers depend on.
fn signature_of(line: &str) -> String {
    match line.rfind('{') {
        Some(at) => format!("{};", line[..at].trim_end()),
        None => line.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the snapshot is a byte comparison, so the extraction has to end at
    /// the closing brace of the declaration. Reading to the end of the file would
    /// make every unrelated edit below a frozen contract read as drift.
    #[test]
    fn a_declaration_ends_where_its_braces_balance() {
        let text = "\
use std::fmt;

pub trait Example {
    fn one(&self);
    fn two(&self) {
        let _ = 1;
    }
}

pub trait Other {}
";
        let block = extract_declaration(text, "pub trait Example")
            .expect("the declaration is found");
        assert_eq!(
            block,
            "pub trait Example {\nfn one(&self);\nfn two(&self);\n}\n"
        );
    }

    /// WHY: a default body and a doc comment both move on ordinary work. If
    /// either were part of the snapshot, refactoring a default body or
    /// repointing a doc link would report a major version event, and a reader
    /// who has refreshed the snapshot twice for nothing stops reading it. Only
    /// the signatures may move the bytes.
    #[test]
    fn neither_a_default_body_nor_a_comment_is_part_of_the_contract() {
        let before = "\
pub trait Example {
    /// Probe the device.
    ///
    /// ```no_run
    /// use vyre_foundation::Program;
    /// # fn example(program: &Program) -> bool {
    /// true
    /// # }
    /// ```
    fn probe(&self) -> bool {
        let mut count = 0;
        for _ in 0..2 {
            count += 1;
        }
        count > 0
    }
}
";
        let after = "\
pub trait Example {
    /// Probe the device, which is what the caller waits on.
    ///
    /// ```no_run
    /// use vyre_foundation::ir::Program;
    /// # fn example(program: &Program) -> bool {
    /// true
    /// # }
    /// ```
    fn probe(&self) -> bool {
        helper(self)
    }
}
";
        let renamed = "\
pub trait Example {
    fn probe(&self, timeout: u64) -> bool {
        helper(self)
    }
}
";
        let extract = |text| extract_declaration(text, "pub trait Example");
        assert_eq!(
            extract(before),
            Some("pub trait Example {\nfn probe(&self) -> bool;\n}\n".to_string())
        );
        assert_eq!(extract(before), extract(after));
        assert_ne!(extract(before), extract(renamed));
    }

    /// WHY: a brace inside a string literal is data, not nesting. Counting one
    /// would end the declaration early and freeze half of it.
    #[test]
    fn a_brace_in_a_string_literal_does_not_close_the_declaration() {
        let text = "\
pub enum E {
    A,
}
";
        let braced = "\
pub enum E {
    A,
}
const FORMAT: &str = \"}\";
";
        assert_eq!(
            extract_declaration(braced, "pub enum E"),
            extract_declaration(text, "pub enum E")
        );
    }

    /// WHY: indentation is stripped so a rustfmt width change does not read as a
    /// contract change, which is the one thing this snapshot must not do.
    #[test]
    fn indentation_is_not_part_of_the_contract() {
        let one = extract_declaration("pub enum E {\n    A,\n}\n", "pub enum E");
        let two = extract_declaration("pub enum E {\n        A,\n}\n", "pub enum E");
        assert_eq!(one, two);
    }

    /// WHY: the field walk decides what has to be on the wire. It must read only
    /// the program struct's own fields, stopping at its closing brace, or a field
    /// of the next type would be demanded on the wire.
    #[test]
    fn only_the_program_struct_fields_are_read() {
        let text = "\
pub struct Program {
    pub entry: u32,
    pub(crate) buffers: Vec<u8>,
    method_like: (),
}

pub struct Other {
    pub unrelated: u32,
}
";
        let fields = program_fields(text, 1);
        assert_eq!(
            fields.iter().map(|(_, name)| name.as_str()).collect::<Vec<_>>(),
            vec!["entry", "buffers"]
        );
    }

    /// WHY: `inventory::submit!` also appears in prose and in a use statement.
    /// Only the macro invocation opening a block is a submission.
    #[test]
    fn a_needle_counts_only_when_its_terminator_follows() {
        assert!(followed_by("inventory::submit! {", "inventory::submit!", '{'));
        assert!(followed_by("    supported_ops: &[],", "supported_ops", ':'));
        assert!(!followed_by(
            "use inventory::submit;",
            "inventory::submit!",
            '{'
        ));
        assert!(!followed_by("// supported_ops is advertised", "supported_ops", ':'));
    }

    /// WHY: the frozen set is a table in this file and the snapshots are files on
    /// disk. Deleting a row leaves its snapshot behind, where it reads as a
    /// frozen declaration nothing compares against. The gate has to name it.
    #[test]
    fn a_snapshot_no_contract_claims_is_reported() {
        let root = std::env::temp_dir().join(format!("vyre-frozen-orphan-{}", std::process::id()));
        let snapshots = root.join("docs/frozen-traits");
        fs::create_dir_all(&snapshots).expect("the fixture directory is created");
        fs::write(snapshots.join("Claimed.txt"), "pub trait Claimed {}\n")
            .expect("the claimed snapshot is written");
        fs::write(snapshots.join("Retired.txt"), "pub trait Retired {}\n")
            .expect("the retired snapshot is written");
        fs::write(snapshots.join("notes.md"), "prose\n").expect("the note is written");

        let contracts: &[(&str, &str, &str)] =
            &[("Claimed", "src/claimed.rs", "pub trait Claimed")];
        let orphans = orphan_snapshots(&snapshots, "docs/frozen-traits", contracts)
            .expect("the snapshot directory is listed");

        fs::remove_dir_all(&root).expect("the fixture is removed");
        assert_eq!(orphans, vec!["docs/frozen-traits/Retired.txt".to_string()]);
    }
}
