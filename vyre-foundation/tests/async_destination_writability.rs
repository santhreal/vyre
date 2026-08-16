//! The class closed here: an async transfer that writes into a buffer the
//! program declared unwritable.
//!
//! `AsyncLoad` and `AsyncStore` name endpoints that may sit outside the
//! dispatch's buffer table, so neither endpoint has to resolve. When the
//! destination does resolve, the target compilers lower the transfer to a
//! counted store loop through that buffer's binding
//! (`vyre-emit-naga/src/emitter/async_op.rs`), so a read-only destination is a
//! write to a read-only binding and the backend is the first thing that says
//! so.
//!
//! The rule is also what makes loop-invariant hoisting sound. `LoopLicm` treats
//! a load from a read-only buffer as invariant because nothing in a valid
//! program writes one. A DMA into a read-only destination is exactly such a
//! write, and a load hoisted past it answers a value from before the transfer.
//! The store rule `V063` never covered that case, because a transfer is not a
//! `Node::Store`.
//!
//! The two rules share one predicate. A test per access mode is what keeps the
//! transfer rule from admitting what the store rule refuses, or the reverse.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_foundation::validate::validate;

/// Every access mode the frozen contract declares, read off the type at run
/// time.
///
/// `BufferAccess` is `#[non_exhaustive]` and carries no iterator, so a match or
/// a literal list here would go stale in silence. Deserializing a name no
/// variant carries makes serde name every variant that exists, which is the
/// derive macro answering rather than this file remembering.
fn access_modes() -> Vec<BufferAccess> {
    let refusal = serde_json::from_str::<BufferAccess>("\"not_an_access_mode\"")
        .expect_err("a name no variant carries must be refused")
        .to_string();
    let listed = refusal
        .split_once("expected one of ")
        .unwrap_or_else(|| {
            panic!(
                "serde no longer lists the variants it expected, so the roster cannot be \
                 derived: {refusal}"
            )
        })
        .1;
    let listed = listed.split_once(" at line ").map_or(listed, |(head, _)| head);
    listed
        .split(", ")
        .map(|name| name.trim_matches(['`', ' ', '.']))
        .map(|name| {
            serde_json::from_str::<BufferAccess>(&format!("\"{name}\"")).unwrap_or_else(|error| {
                panic!("serde listed `{name}` and then refused it: {error}")
            })
        })
        .collect()
}

/// Access modes the frozen contract carries.
///
/// Measured from [`access_modes`] at 5 on 2026-08-16. A sixth mode turns this
/// red so that its write licence is a decision recorded here rather than a
/// default.
const ACCESS_MODE_COUNT: usize = 5;

/// Whether a program may write into a buffer of this mode.
///
/// Restated here rather than imported, because the point of the test is to
/// compare the validator's answer against an independent statement of the rule.
fn admits_write(access: &BufferAccess) -> bool {
    matches!(
        access,
        BufferAccess::ReadWrite | BufferAccess::WriteOnly | BufferAccess::Workgroup
    )
}

fn declaration(name: &str, access: &BufferAccess) -> BufferDecl {
    match access {
        BufferAccess::Workgroup => BufferDecl::workgroup(name, 8, DataType::U32),
        other => BufferDecl::storage(name, 0, other.clone(), DataType::U32).with_count(8),
    }
}

fn transfer_into(destination: &str) -> Vec<Node> {
    vec![
        Node::AsyncLoad {
            source: "ssd".into(),
            destination: Ident::from(destination),
            offset: Box::new(Expr::u32(0)),
            size: Box::new(Expr::u32(4)),
            tag: "stage0".into(),
        },
        Node::AsyncWait {
            tag: "stage0".into(),
        },
    ]
}

fn codes(buffers: Vec<BufferDecl>, entry: Vec<Node>) -> Vec<String> {
    validate(&Program::wrapped(buffers, [8, 1, 1], entry))
        .iter()
        .map(|error| error.code().as_str().to_string())
        .collect()
}

fn reports_v134(buffers: Vec<BufferDecl>, entry: Vec<Node>) -> bool {
    codes(buffers, entry).iter().any(|code| code == "V134")
}

#[test]
fn every_access_mode_the_contract_carries_has_a_write_decision() {
    let modes = access_modes();
    assert_eq!(
        modes.len(),
        ACCESS_MODE_COUNT,
        "BufferAccess gained or lost a mode: {modes:?}. Fix: decide whether the new mode \
         admits a write, record it in admits_store and in admits_write here, then move \
         ACCESS_MODE_COUNT."
    );
}

#[test]
fn a_transfer_is_refused_exactly_when_its_destination_admits_no_write() {
    for access in access_modes() {
        let reported = reports_v134(
            vec![declaration("target", &access)],
            transfer_into("target"),
        );
        assert_eq!(
            reported,
            !admits_write(&access),
            "a transfer into a {access:?} destination must be refused only when that mode \
             admits no write. Fix: vyre-foundation/src/validate/bytes_rejection.rs::admits_store."
        );
    }
}

#[test]
fn a_store_is_refused_for_the_same_modes_a_transfer_is() {
    for access in access_modes() {
        let store = codes(
            vec![declaration("target", &access)],
            vec![Node::store("target", Expr::u32(0), Expr::u32(1))],
        );
        let refused = store.iter().any(|code| code == "V063");
        assert_eq!(
            refused,
            !admits_write(&access),
            "the store rule and the transfer rule must admit the same modes; {access:?} \
             diverged. Fix: both read admits_store, so a divergence is a second predicate."
        );
    }
}

#[test]
fn an_async_store_destination_carries_the_same_rule_as_an_async_load() {
    let entry = vec![
        Node::AsyncStore {
            source: "vram".into(),
            destination: "target".into(),
            offset: Box::new(Expr::u32(0)),
            size: Box::new(Expr::u32(4)),
            tag: "drain".into(),
        },
        Node::AsyncWait {
            tag: "drain".into(),
        },
    ];
    assert!(
        reports_v134(
            vec![declaration("target", &BufferAccess::ReadOnly)],
            entry.clone()
        ),
        "a store transfer writes its destination exactly as a load transfer does"
    );
    assert!(
        !reports_v134(
            vec![declaration("target", &BufferAccess::ReadWrite)],
            entry
        ),
        "a writable destination is accepted"
    );
}

#[test]
fn an_unresolved_destination_is_not_a_finding() {
    assert!(
        !reports_v134(
            vec![BufferDecl::read("input", 0, DataType::U32).with_count(8)],
            transfer_into("vram"),
        ),
        "an endpoint the dispatch does not bind names a storage tier, and a tier has no \
         access mode to check"
    );
}

#[test]
fn a_read_only_source_is_not_a_finding() {
    assert!(
        !reports_v134(
            vec![
                BufferDecl::read("cold", 0, DataType::U32).with_count(8),
                BufferDecl::storage("hot", 1, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(8),
            ],
            vec![
                Node::AsyncLoad {
                    source: "cold".into(),
                    destination: "hot".into(),
                    offset: Box::new(Expr::u32(0)),
                    size: Box::new(Expr::u32(4)),
                    tag: "stage0".into(),
                },
                Node::AsyncWait {
                    tag: "stage0".into(),
                },
            ],
        ),
        "a transfer reads its source, so a read-only source is the ordinary case"
    );
}
