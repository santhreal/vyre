//! Structural gate against triple-nested byte-row types on backend dispatch
//! surfaces.
//!
//! A batched dispatch that produces one output row per unit of work used to
//! spell its result `Vec<Vec<Vec<u8>>>`: a vector of dispatches, each a vector
//! of output slots, each a vector of bytes. The middle level held exactly one
//! slot at every call site, so an `n`-dispatch batch allocated `2n + 1` vectors
//! to carry `n` byte rows and copied each row out of the mapped staging range
//! into a vector of its own. Batched rows now live in one buffer behind
//! `vyre_driver::BatchOutputs`.
//!
//! The `hot-path-nested-rows` gate reads the two-level `Vec<Vec<u8>>` on the
//! trait that returns it: rows returned by a dispatch trait are legitimate as
//! long as the trait also offers a form that fills slots the caller keeps. That
//! rule says nothing about a third level appearing inside a backend, so this
//! contract covers the depth.
//!
//! Scope is every crate that implements the backend trait, decided by parsing
//! each workspace member's own sources rather than from a list typed here, so a
//! backend crate added later is covered the day it lands. Fixture registries
//! outside those crates keep their cases-by-buffers-by-bytes shape, which is a
//! genuinely three-level subject rather than a dispatch result.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use proc_macro2::LineColumn;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{GenericArgument, ItemImpl, PathArguments, Type, TypePath};

const BACKEND_TRAIT: &str = "VyreBackend";
const BYTE: &str = "u8";

#[derive(Default)]
struct NestedByteRowVisitor {
    locations: Vec<LineColumn>,
}

impl<'ast> Visit<'ast> for NestedByteRowVisitor {
    fn visit_type_path(&mut self, path: &'ast TypePath) {
        if is_triple_nested_byte_rows(path) {
            self.locations.push(path.span().start());
        }
        visit::visit_type_path(self, path);
    }
}

/// Whether `path` is a three-deep vector whose innermost element is `u8`.
///
/// Matched on the parsed type rather than the source text, so a line break, a
/// fully qualified `alloc::vec::Vec`, or extra whitespace between the angle
/// brackets cannot hide an occurrence, and a doc comment naming the type cannot
/// fabricate one.
fn is_triple_nested_byte_rows(path: &TypePath) -> bool {
    let Some(rows) = vec_element(path) else {
        return false;
    };
    let Some(row) = as_vec_element(rows) else {
        return false;
    };
    let Some(byte) = as_vec_element(row) else {
        return false;
    };
    is_u8(byte)
}

/// The single generic argument of a `Vec<..>` type path, or `None` for anything
/// else.
fn vec_element(path: &TypePath) -> Option<&Type> {
    let segment = path.path.segments.last()?;
    if segment.ident != "Vec" {
        return None;
    }
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    if arguments.args.len() != 1 {
        return None;
    }
    match arguments.args.first()? {
        GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

fn as_vec_element(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Path(path) => vec_element(path),
        _ => None,
    }
}

fn is_u8(ty: &Type) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    path.qself.is_none()
        && path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == BYTE && segment.arguments.is_none())
}

fn nested_byte_row_locations(source: &str) -> Result<Vec<LineColumn>, syn::Error> {
    let file = syn::parse_file(source)?;
    let mut visitor = NestedByteRowVisitor::default();
    visitor.visit_file(&file);
    Ok(visitor.locations)
}

#[derive(Default)]
struct BackendImplVisitor {
    found: bool,
}

impl<'ast> Visit<'ast> for BackendImplVisitor {
    fn visit_item_impl(&mut self, item: &'ast ItemImpl) {
        if let Some((_, path, _)) = &item.trait_ {
            if path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == BACKEND_TRAIT)
            {
                self.found = true;
            }
        }
        visit::visit_item_impl(self, item);
    }
}

/// Whether `source` implements the backend trait for any type.
///
/// Parsed rather than grepped because the driver crates name the trait both bare
/// and fully qualified, and a substring rule would also match the name inside a
/// doc comment or a string literal.
fn implements_backend_trait(source: &str) -> Result<bool, syn::Error> {
    let file = syn::parse_file(source)?;
    let mut visitor = BackendImplVisitor::default();
    visitor.visit_file(&file);
    Ok(visitor.found)
}

fn read_source(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| {
        panic!(
            "Fix: nested byte-row gate cannot read {}: {error}",
            path.display()
        )
    })
}

fn parse_or_panic<T>(path: &Path, parsed: Result<T, syn::Error>) -> T {
    parsed.unwrap_or_else(|error| {
        panic!(
            "Fix: nested byte-row gate cannot parse {}: {error}",
            path.display()
        )
    })
}

/// Whether the member rooted at `src_dir` implements the backend trait anywhere
/// in its production sources.
fn member_implements_backend_trait(src_dir: &Path) -> bool {
    super::workspace_sources::rust_sources_under(src_dir)
        .iter()
        .any(|path| parse_or_panic(path, implements_backend_trait(&read_source(path))))
}

/// Every workspace member `src` directory that implements the backend trait.
fn backend_crate_src_dirs(root: &Path) -> Vec<PathBuf> {
    super::workspace_sources::workspace_member_src_dirs(root)
        .into_iter()
        .filter(|src_dir| member_implements_backend_trait(src_dir))
        .collect()
}

/// No backend crate may name a three-deep byte-row vector in production source.
#[test]
fn backend_sources_reject_triple_nested_byte_row_types() {
    let root = super::workspace_sources::workspace_root();
    let mut violations = Vec::new();

    for src_dir in backend_crate_src_dirs(&root) {
        for path in super::workspace_sources::rust_sources_under(&src_dir) {
            let locations = parse_or_panic(&path, nested_byte_row_locations(&read_source(&path)));
            violations.extend(locations.into_iter().map(|location| {
                super::workspace_sources::violation_location(&root, &path, location)
            }));
        }
    }

    assert_eq!(
        violations,
        Vec::<String>::new(),
        "replace three-deep byte-row vectors with one contiguous buffer plus row offsets, such as `vyre_driver::BatchOutputs`"
    );
}

/// Membership in the scanned set must agree with the trait scan for EVERY
/// workspace member, and the set must span more than one crate.
///
/// A scope typed into this file stops covering a backend crate added after it
/// was written, and a filter that silently resolves to one crate or to nothing
/// reports success over almost nothing. Both read identically to a clean tree,
/// so the set construction is asserted against the predicate it is built from
/// rather than trusted.
#[test]
fn the_scanned_set_is_exactly_the_crates_that_implement_the_backend_trait() {
    let root = super::workspace_sources::workspace_root();
    let scanned: BTreeSet<PathBuf> = backend_crate_src_dirs(&root).into_iter().collect();

    let mut disagreements = Vec::new();
    for src_dir in super::workspace_sources::workspace_member_src_dirs(&root) {
        let implements = member_implements_backend_trait(&src_dir);
        if implements != scanned.contains(&src_dir) {
            disagreements.push(format!(
                "{} implements={implements} scanned={}",
                src_dir.strip_prefix(&root).unwrap_or(&src_dir).display(),
                scanned.contains(&src_dir)
            ));
        }
    }

    assert_eq!(
        disagreements,
        Vec::<String>::new(),
        "the nested byte-row scan must cover exactly the crates that implement the backend trait"
    );
    assert!(
        scanned.len() > 1,
        "Fix: the nested byte-row scan resolved {} backend crate(s); the defect class spans every backend, so a one-crate or empty scope is a broken derivation",
        scanned.len()
    );
}

/// The detector must recognise the exact shape the batched readback path used.
///
/// A gate that cannot fail on the defect it names is a clean-tree assertion.
#[test]
fn detector_rejects_a_triple_nested_byte_row_return() {
    let source = "fn dispatch() -> Result<Vec<Vec<Vec<u8>>>, Error> { unreachable!() }";
    assert_eq!(nested_byte_row_locations(source).unwrap().len(), 1);
}

/// A fully qualified spelling and a line-broken one are the same type.
#[test]
fn detector_rejects_qualified_and_line_broken_spellings() {
    let source = "type Rows = alloc::vec::Vec<\n    Vec<\n        Vec<u8>,\n    >,\n>;";
    assert_eq!(nested_byte_row_locations(source).unwrap().len(), 1);
}

/// Two levels is the per-slot output of one dispatch and must stay legal, and a
/// three-deep vector of something other than bytes is a different subject.
#[test]
fn detector_accepts_two_levels_and_non_byte_elements() {
    let source = "fn slots() -> Vec<Vec<u8>> { Vec::new() }\nfn words() -> Vec<Vec<Vec<u32>>> { Vec::new() }";
    assert_eq!(nested_byte_row_locations(source).unwrap(), Vec::new());
}

/// A doc comment naming the type must not be read as an occurrence, or the gate
/// would forbid explaining the shape it replaced.
#[test]
fn detector_ignores_the_type_named_in_prose() {
    let source = "/// Replaced the `Vec<Vec<Vec<u8>>>` result.\npub struct Rows;";
    assert_eq!(nested_byte_row_locations(source).unwrap(), Vec::new());
}

/// The backend-crate detector must accept both spellings the driver crates use
/// and reject an unrelated trait impl.
#[test]
fn backend_trait_detector_accepts_bare_and_qualified_impls() {
    assert!(implements_backend_trait("impl VyreBackend for B { }").unwrap());
    assert!(implements_backend_trait("impl vyre_driver::VyreBackend for B { }").unwrap());
    assert!(!implements_backend_trait("impl Default for B { }").unwrap());
    assert!(!implements_backend_trait("/// impl VyreBackend for B\nstruct B;").unwrap());
}
