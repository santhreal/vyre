//! The source-derived inventory of mutable shared state owners, closed in both
//! directions against the `StateOwnerRecovery` impl that states each one's
//! failure domain and recovery class.
//!
//! A poisoned lock means a thread panicked while holding it, so the guarded
//! value may be half written. `vyre-foundation::failure_domain` states what
//! happens next as a `FailureDomain` and a `RecoveryClass`, and the rest of
//! the gate rejects answering that question anywhere else. That leaves one
//! hole: an owner nobody ever answered the question for. A field added to a
//! struct in the runtime or the driver introduces a new place a panic can
//! leave half-written state, and no rule about how poison is handled at a call
//! site says which recovery the owner is entitled to.
//!
//! The inventory here is read out of the syntax tree at run time, never from a
//! list in this file and never from a declaration file beside it. A struct
//! holding a `Mutex`, `RwLock`, `DashMap` or `AtomicGuardedState` field must
//! implement `StateOwnerRecovery` in the same file, and a `StateOwnerRecovery`
//! impl for a type that holds no such field is equally a finding. Adding a
//! field turns the gate red until the type states its recovery; deleting the
//! last one turns it red until the impl goes with it. Both domain and class
//! are enum variants written in Rust, so the compiler checks their spelling
//! and this gate only has to check that the decision exists.
//!
//! A static cannot implement a trait, so a static of owner type is reported
//! and the fix is a named type around it. `OnceLock`, `LazyLock` and the
//! atomics are not owners, for the reason the parent module states: no poison
//! flag, no exposed half-written value, and so no recovery decision to record.

use std::collections::BTreeSet;
use std::path::Path;

use quote::ToTokens;

use crate::gate::{Finding, GateError, Report};
use crate::gates::scan::Tree;

/// Roots whose every owner of mutable shared state states a recovery contract.
///
/// These two crates hold the state a device generation is bound to: admitted
/// artifacts, resident pages, residency leases, routing profiles and restart
/// budgets. A panic under any of their locks can strand a device resource, so
/// this is where an unanswered owner costs the most.
pub const OWNER_CONTRACT_ROOTS: &[&str] = &["vyre-runtime/src", "vyre-driver/src"];

/// Roots that own no mutable shared state at all.
///
/// The megakernel is the artifact body a device generation executes. It reads
/// a program and emits a module, and holding a lock there would place mutable
/// state inside the thing every backend is expected to be able to reproduce
/// byte for byte. There is no recovery contract that makes an owner here
/// correct, so the answer is not which `FailureDomain` it reaches but that it
/// must not exist.
pub const PURE_ROOTS: &[&str] = &["vyre-megakernel/src"];

/// The trait a mutable state owner states its recovery contract through.
const CONTRACT_TRAIT: &str = "StateOwnerRecovery";

/// Wrapper types that expose a value a panicking thread may have left half
/// written.
const OWNER_TYPES: &[&str] = &["Mutex", "RwLock", "DashMap", "AtomicGuardedState"];

/// One owner of mutable shared state found in source.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceOwner {
    /// One-based line of the field or static name.
    line: u32,
    /// `Type.field` for a struct field, or the name of a static.
    id: String,
    /// The struct that holds the field, absent for a static.
    holder: Option<String>,
}

/// Whether a token stream names the bare identifier `test`.
///
/// Reading identifiers rather than the printed text is what keeps
/// `feature = "test-support"` out of the answer: there `test` is part of a
/// string literal and gates nothing out of a production build.
fn names_test(tokens: proc_macro2::TokenStream) -> bool {
    tokens.into_iter().any(|tree| match tree {
        proc_macro2::TokenTree::Ident(ident) => ident == "test",
        proc_macro2::TokenTree::Group(group) => names_test(group.stream()),
        _ => false,
    })
}

/// Whether an attribute list gates the item out of a production build.
fn is_test_gated(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .any(|attr| attr.path().is_ident("cfg") && names_test(attr.meta.to_token_stream()))
}

/// Whether an attribute list marks the item as a test.
fn is_test_item(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| attr.path().is_ident("test"))
}

/// The owner wrapper a type resolves to.
///
/// The walk descends through every type constructor that can hold one, so
/// `Arc<Mutex<T>>`, `OnceLock<Mutex<T>>`, `[Mutex<T>; 256]` and
/// `(Mutex<T>, u64)` each resolve to their inner owner. A sharded cache is a
/// fixed-size array of mutexes, and stopping at `Type::Path` left every one of
/// its shards outside the inventory.
fn owner_wrapper(ty: &syn::Type) -> Option<&'static str> {
    match ty {
        syn::Type::Path(path) => path.path.segments.iter().find_map(|segment| {
            if let Some(owner) = OWNER_TYPES.iter().copied().find(|name| segment.ident == *name) {
                return Some(owner);
            }
            match &segment.arguments {
                syn::PathArguments::AngleBracketed(arguments) => {
                    arguments.args.iter().find_map(|argument| match argument {
                        syn::GenericArgument::Type(inner) => owner_wrapper(inner),
                        _ => None,
                    })
                }
                _ => None,
            }
        }),
        syn::Type::Reference(reference) => owner_wrapper(&reference.elem),
        syn::Type::Paren(paren) => owner_wrapper(&paren.elem),
        syn::Type::Group(group) => owner_wrapper(&group.elem),
        syn::Type::Array(array) => owner_wrapper(&array.elem),
        syn::Type::Slice(slice) => owner_wrapper(&slice.elem),
        syn::Type::Ptr(pointer) => owner_wrapper(&pointer.elem),
        syn::Type::Tuple(tuple) => tuple.elems.iter().find_map(owner_wrapper),
        _ => None,
    }
}

/// The bare name of the type an `impl` block is written for.
fn implemented_type(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => path
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string()),
        syn::Type::Reference(reference) => implemented_type(&reference.elem),
        syn::Type::Paren(paren) => implemented_type(&paren.elem),
        syn::Type::Group(group) => implemented_type(&group.elem),
        _ => None,
    }
}

/// What one file states about mutable state ownership.
#[derive(Default)]
struct FileContracts {
    owners: Vec<SourceOwner>,
    /// Types stating a recovery contract, paired with the line of the impl.
    contracts: Vec<(String, u32)>,
}

/// Collects the owners and the recovery contracts in one file, in every scope
/// that can define either.
///
/// A function-body static is as much an owner as a struct field: the trace
/// event ring in the driver was a `Mutex` declared inside the accessor that
/// hands it out. Visiting the whole tree rather than the top-level item list
/// is what reaches it, an impl method, and a trait default body alike.
struct OwnerVisitor<'a> {
    found: &'a mut FileContracts,
}

impl<'ast> syn::visit::Visit<'ast> for OwnerVisitor<'_> {
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if is_test_gated(&node.attrs) {
            return;
        }
        syn::visit::visit_item_mod(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        if is_test_gated(&node.attrs) || is_test_item(&node.attrs) {
            return;
        }
        syn::visit::visit_item_fn(self, node);
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        if is_test_gated(&node.attrs) || is_test_item(&node.attrs) {
            return;
        }
        syn::visit::visit_impl_item_fn(self, node);
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        if is_test_gated(&node.attrs) {
            return;
        }
        let names_contract = node
            .trait_
            .as_ref()
            .and_then(|(_, path, _)| path.segments.last())
            .is_some_and(|segment| segment.ident == CONTRACT_TRAIT);
        if names_contract {
            if let Some(name) = implemented_type(&node.self_ty) {
                self.found
                    .contracts
                    .push((name, node.impl_token.span.start().line as u32));
            }
        }
        syn::visit::visit_item_impl(self, node);
    }

    fn visit_item_struct(&mut self, node: &'ast syn::ItemStruct) {
        if is_test_gated(&node.attrs) {
            return;
        }
        let syn::Fields::Named(fields) = &node.fields else {
            return;
        };
        for field in &fields.named {
            if is_test_gated(&field.attrs) || owner_wrapper(&field.ty).is_none() {
                continue;
            }
            let Some(name) = field.ident.as_ref() else {
                continue;
            };
            self.found.owners.push(SourceOwner {
                line: name.span().start().line as u32,
                id: format!("{}.{name}", node.ident),
                holder: Some(node.ident.to_string()),
            });
        }
    }

    fn visit_item_static(&mut self, node: &'ast syn::ItemStatic) {
        if is_test_gated(&node.attrs) || owner_wrapper(&node.ty).is_none() {
            return;
        }
        self.found.owners.push(SourceOwner {
            line: node.ident.span().start().line as u32,
            id: node.ident.to_string(),
            holder: None,
        });
    }
}

/// Read one file's owners and recovery contracts.
fn contracts_in(text: &str, relative_path: &Path) -> Result<FileContracts, GateError> {
    let file = syn::parse_file(text).map_err(|error| {
        GateError::new(
            format!("cannot parse `{}`: {error}", relative_path.display()),
            "keep the file parseable Rust so its state owners can be inventoried",
        )
    })?;
    let mut found = FileContracts::default();
    syn::visit::Visit::visit_file(&mut OwnerVisitor { found: &mut found }, &file);
    Ok(found)
}

/// Check both directions and report how many owners were inventoried.
pub fn check(tree: &Tree, report: &mut Report) -> Result<usize, GateError> {
    let mut inventoried = 0;
    for relative_path in tree.rust(OWNER_CONTRACT_ROOTS)? {
        let stem = relative_path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if stem == "tests" || stem.ends_with("_tests") {
            continue;
        }
        let found = contracts_in(&tree.read(&relative_path)?, &relative_path)?;
        inventoried += found.owners.len();

        let holders: BTreeSet<&str> = found
            .owners
            .iter()
            .filter_map(|owner| owner.holder.as_deref())
            .collect();
        let stated: BTreeSet<&str> = found
            .contracts
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();

        for owner in &found.owners {
            let Some(holder) = owner.holder.as_deref() else {
                report.find(Finding::at(
                    relative_path.clone(),
                    owner.line,
                    format!(
                        "static `{}` owns mutable shared state and cannot state a recovery class",
                        owner.id
                    ),
                    format!(
                        "wrap it in a named type holding the lock and implement {CONTRACT_TRAIT} \
                         for that type; a static has nothing to implement it on"
                    ),
                ));
                continue;
            };
            if stated.contains(holder) {
                continue;
            }
            report.find(Finding::at(
                relative_path.clone(),
                owner.line,
                format!(
                    "`{}` owns mutable shared state and `{holder}` states no recovery class",
                    owner.id
                ),
                format!(
                    "implement {CONTRACT_TRAIT} for `{holder}` in this file, naming the \
                     FailureDomain a panic under this lock reaches and the RecoveryClass that \
                     remediates it"
                ),
            ));
        }

        for (name, line) in &found.contracts {
            if holders.contains(name.as_str()) {
                continue;
            }
            report.find(Finding::at(
                relative_path.clone(),
                *line,
                format!("`{name}` states a recovery class but owns no mutable shared state"),
                format!(
                    "delete the {CONTRACT_TRAIT} impl, or move it to the type that holds the lock \
                     it describes"
                ),
            ));
        }
    }

    for relative_path in tree.rust(PURE_ROOTS)? {
        let stem = relative_path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if stem == "tests" || stem.ends_with("_tests") {
            continue;
        }
        for owner in contracts_in(&tree.read(&relative_path)?, &relative_path)?.owners {
            report.find(Finding::at(
                relative_path.clone(),
                owner.line,
                format!("`{}` owns mutable shared state in a pure root", owner.id),
                "move the state to the runtime, which states a recovery contract for it, and \
                 leave the megakernel reproducible from its program alone"
                    .to_string(),
            ));
        }
    }

    Ok(inventoried)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read one synthetic file the way the gate reads a tracked one.
    fn parse(source: &str) -> FileContracts {
        contracts_in(source, Path::new("synthetic.rs")).expect("synthetic source parses")
    }

    /// WHY: the wrapper list is the gate's definition of an owner, so a
    /// wrapper added to `OWNER_TYPES` that the visitor never resolves would
    /// widen the definition and inventory nothing. Deriving the cases from the
    /// const turns that into a red test instead of a silent hole.
    #[test]
    fn every_owner_wrapper_is_inventoried() {
        for wrapper in OWNER_TYPES {
            let found = parse(&format!("struct Holder {{ guarded: {wrapper}<u32> }}"));
            assert_eq!(
                found.owners.len(),
                1,
                "a `{wrapper}` field must be inventoried as a mutable state owner"
            );
            assert_eq!(found.owners[0].holder.as_deref(), Some("Holder"));
            assert_eq!(found.owners[0].id, "Holder.guarded");
        }
    }

    /// WHY: a sharded cache holds its locks behind array, tuple and smart
    /// pointer constructors. Stopping at the outermost path left every shard
    /// uninventoried.
    #[test]
    fn an_owner_behind_a_constructor_is_still_an_owner() {
        for ty in [
            "Arc<Mutex<u32>>",
            "[Mutex<u32>; 256]",
            "(Mutex<u32>, u64)",
            "OnceLock<RwLock<u32>>",
        ] {
            let found = parse(&format!("struct Holder {{ guarded: {ty} }}"));
            assert_eq!(found.owners.len(), 1, "`{ty}` holds a mutable state owner");
        }
    }

    /// WHY: a type with no poison flag exposes no half-written value, so
    /// calling it an owner would demand a recovery decision that has nothing
    /// to decide.
    #[test]
    fn a_type_without_a_poison_flag_is_not_an_owner() {
        for ty in ["OnceLock<u32>", "AtomicU64", "Vec<u32>", "RefCell<u32>"] {
            let found = parse(&format!("struct Holder {{ field: {ty} }}"));
            assert!(found.owners.is_empty(), "`{ty}` owns no poisonable state");
        }
    }

    /// WHY: a static has no type of its own to implement the contract on, so
    /// the gate reports it separately and the fix is a named wrapper type.
    #[test]
    fn a_static_owner_records_no_holder() {
        let found = parse("static CACHE: Mutex<u32> = Mutex::new(0);");
        assert_eq!(found.owners.len(), 1);
        assert_eq!(found.owners[0].holder, None);
        assert_eq!(found.owners[0].id, "CACHE");
    }

    /// WHY: a fixture holding a lock ships in no production build, so
    /// demanding a recovery contract from it would report a fault that cannot
    /// reach a device.
    #[test]
    fn test_gated_state_is_outside_the_inventory() {
        let found = parse(
            "#[cfg(test)]\n\
             struct Fixture { guarded: Mutex<u32> }\n\
             struct Holder { #[cfg(test)] guarded: Mutex<u32> }\n",
        );
        assert!(found.owners.is_empty());
    }

    /// WHY: the inventory is closed in both directions, so the impl side has
    /// to be read out of the same file for the two sets to be comparable.
    #[test]
    fn a_recovery_contract_is_read_from_the_impl() {
        let found = parse(
            "struct Holder { guarded: Mutex<u32> }\n\
             impl StateOwnerRecovery for Holder {}\n",
        );
        assert_eq!(found.owners.len(), 1);
        assert_eq!(
            found.contracts.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>(),
            vec!["Holder"]
        );
    }
}
