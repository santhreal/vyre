//! The physical-IR variant space, read from this crate's own source.
//!
//! `KernelOpKind` is the closed enum every `vyre-emit-*` crate has to decide
//! on, and every emitter contract suite asks the same question: which variants
//! exist right now. Each suite had written the same source scanner, resolved
//! the workspace root at run time and read `descriptor/mod.rs` off disk, so a
//! run whose working directory pointed at another checkout enumerated another
//! tree's enum and still passed.
//!
//! The declaration is in this crate, so the enumeration is in this crate.
//! `include_str!` binds the scan to the source that was compiled, which
//! removes the working directory from the answer and makes the file a rebuild
//! trigger.

use std::collections::BTreeSet;

/// This crate's declaration of the physical-IR op enum, bound at compile time.
const DESCRIPTOR_SOURCE: &str = include_str!("descriptor/mod.rs");

/// The header line that opens the physical-IR op enum.
const KERNEL_OP_KIND_HEADER: &str = "pub enum KernelOpKind";

/// Every `KernelOpKind` variant name this crate declares.
///
/// Derived from source rather than from a written list, so a variant added to
/// the enum enters the space with no second edit and a contract stated over
/// the space covers it immediately.
#[must_use]
pub fn kernel_op_kind_variants() -> BTreeSet<String> {
    variant_names(DESCRIPTOR_SOURCE, KERNEL_OP_KIND_HEADER)
}

/// The variant identifiers declared directly inside the enum `header` opens.
///
/// Nested braces belong to a struct variant's fields, not to the enum, so the
/// scan admits an identifier only at brace depth one.
fn variant_names(source: &str, header: &str) -> BTreeSet<String> {
    let mut variants = BTreeSet::new();
    let mut depth = 0_usize;
    let mut inside = false;

    for line in source.lines() {
        let trimmed = line.trim();

        if !inside {
            if trimmed.starts_with(header) {
                inside = true;
                depth = usize::from(trimmed.contains('{'));
            }
            continue;
        }

        if depth == 0 {
            if trimmed.contains('{') {
                depth = 1;
            }
            continue;
        }

        if depth == 1 {
            if trimmed.starts_with('}') {
                break;
            }
            if let Some(ident) = variant_identifier(trimmed) {
                variants.insert(ident.to_string());
            }
        }

        let opens = trimmed.matches('{').count();
        let closes = trimmed.matches('}').count();
        depth = depth.saturating_add(opens).saturating_sub(closes);
        if depth == 0 {
            break;
        }
    }

    variants
}

/// The variant identifier `line` declares, if it declares one.
///
/// Doc comments, attributes and blank lines carry no variant, and a field line
/// inside a struct variant never reaches here because it sits deeper than
/// brace depth one.
fn variant_identifier(line: &str) -> Option<&str> {
    if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
        return None;
    }
    let ident = line.split(['{', '(', ',', ' ']).next()?.trim();
    let first = ident.chars().next()?;
    (first.is_ascii_uppercase()).then_some(ident)
}
