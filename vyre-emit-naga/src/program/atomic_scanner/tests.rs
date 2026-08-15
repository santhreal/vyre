//! # Why this suite exists
//!
//! The write half of this scan used to be a hand-rolled recursive descent whose
//! last arm was `_ => {}`. That arm answered "does this statement write a
//! buffer" with "no" for every variant its author had not listed, and the four
//! collective variants were not listed: a buffer written only by an `AllReduce`
//! was reported as never written, auto-downgraded to `BufferAccess::ReadOnly`,
//! and emitted as `var<storage, read>`.
//!
//! The class closed here is therefore two questions, not one:
//!   - descent, which must reach a nested atomic through EVERY body-bearing
//!     variant, and
//!   - per-node classification, which must report every buffer the declaration
//!     site calls a write.
//!
//! Both loops take their variant set from `vyre_test_support::ir_variants`,
//! which is checked against `vyre_foundation::ir::NODE_VARIANT_NAMES` at run
//! time. A new `Node` variant has no fixture, so
//! `assert_covers_every_node_variant` fails naming it, before either question is
//! asked. What this does NOT catch: a buffer an extension payload names, which
//! core cannot enumerate and the Naga extension registry must report instead.

use super::*;
use vyre_foundation::ir::{Expr, Ident, Node};
use vyre_foundation::visit::node_buffer_refs;
use vyre_test_support::ir_variants::{
    assert_covers_every_node_variant, assert_samples_match_declared_shape, node_body_slot_samples,
    node_variant_samples,
};

fn scan(node: &Node) -> BufferTargets {
    let mut out = BufferTargets::default();
    scan_buffer_targets(std::slice::from_ref(node), &mut out)
        .expect("Fix: scan_buffer_targets must succeed for these IR shapes");
    out
}

#[test]
fn a_nested_atomic_is_found_through_every_body_bearing_variant() {
    let marker = Node::let_bind(
        "atomic_probe",
        Expr::atomic_add("nested_atomic_target", Expr::u32(0), Expr::u32(1)),
    );
    let samples = node_body_slot_samples(&marker);
    assert_samples_match_declared_shape(&samples, true);
    assert_covers_every_node_variant(&node_variant_samples());
    assert!(!samples.is_empty(), "the fixture set must not be empty");

    for sample in &samples {
        let targets = scan(&sample.node);
        assert!(
            targets
                .atomic
                .contains(&Ident::from("nested_atomic_target")),
            "{} hides a nested atomic from the scan, so its element type will not wrap in \
             atomic<...>. Fix: descend through vyre_foundation::visit::child_bodies \
             instead of a per-variant match.",
            sample.label()
        );
        assert!(
            targets
                .writes
                .contains(&Ident::from("nested_atomic_target")),
            "{} reports a nested atomic target that is not a write target, which emits \
             atomic<u32> inside var<storage, read>. Fix: keep both halves of BufferTargets \
             from one walk.",
            sample.label()
        );
    }
}

#[test]
fn a_nested_store_is_found_through_every_body_bearing_variant() {
    let marker = Node::store("nested_write_target", Expr::u32(0), Expr::u32(7));
    let samples = node_body_slot_samples(&marker);
    assert_samples_match_declared_shape(&samples, true);
    assert!(!samples.is_empty(), "the fixture set must not be empty");

    for sample in &samples {
        assert!(
            scan(&sample.node)
                .writes
                .contains(&Ident::from("nested_write_target")),
            "{} hides a nested store from the scan, so its destination is auto-downgraded to \
             ReadOnly and the emitted store is rejected with InvalidStorePointer.",
            sample.label()
        );
    }
}

#[test]
fn every_variant_that_names_a_write_reports_it_as_a_write_target() {
    let samples = node_variant_samples();
    assert_covers_every_node_variant(&samples);

    let mut writers = 0_usize;
    for sample in &samples {
        let expected: Vec<Ident> = node_buffer_refs(&sample.node)
            .writes
            .into_iter()
            .flatten()
            .map(Ident::from)
            .collect();
        if expected.is_empty() {
            continue;
        }
        writers += 1;
        let targets = scan(&sample.node);
        for name in expected {
            assert!(
                targets.writes.contains(&name),
                "{} writes `{name}` at the declaration site but the scan does not report it. \
                 Fix: take the write set from \
                 vyre_foundation::visit::node_buffer_refs, the exhaustive owner.",
                sample.label()
            );
        }
    }
    assert!(
        writers > 0,
        "no fixture names a write, so this loop proved nothing. Fix: check that \
         node_buffer_refs still reports writes for the Store and collective fixtures."
    );
}

#[test]
fn an_opaque_node_without_a_registered_scanner_fails_closed() {
    let node = Node::opaque(UnregisteredExtension);
    let error = scan_buffer_targets(std::slice::from_ref(&node), &mut BufferTargets::default())
        .expect_err("Fix: an extension payload that may perform an atomic must not scan as empty");
    let message = error.to_string();
    assert!(
        message.contains("unsupported opaque node"),
        "unexpected message: {message}"
    );
    assert!(message.contains("Fix:"), "unexpected message: {message}");
}

#[derive(Debug)]
struct UnregisteredExtension;

impl vyre_foundation::ir::NodeExtension for UnregisteredExtension {
    fn extension_kind(&self) -> &'static str {
        "vyre.emit_naga.test.unregistered_node"
    }

    fn debug_identity(&self) -> &str {
        "unregistered-node"
    }

    fn stable_fingerprint(&self) -> [u8; 32] {
        [0x3c; 32]
    }

    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn async_load_destination_is_a_write_target() {
    // Reproducer for the vfs::resolve panic: without this, BufferAccess
    // auto-inference downgrades the AsyncLoad destination to ReadOnly, naga
    // emits `var<storage, read>`, and the in-shader Store synthesized by
    // emit_synchronous_async_load fails validation with `InvalidStorePointer`.
    let node = Node::AsyncLoad {
        source: Ident::from("src_buf"),
        destination: Ident::from("dst_buf"),
        offset: Box::new(Expr::u32(0)),
        size: Box::new(Expr::u32(4096)),
        tag: Ident::from("dma_tag"),
    };
    let targets = scan(&node);
    assert!(
        targets.writes.contains(&Ident::from("dst_buf")),
        "AsyncLoad.destination must be tracked as a write target so BufferAccess auto-inference keeps it ReadWrite"
    );
    assert!(
        !targets.writes.contains(&Ident::from("src_buf")),
        "AsyncLoad.source is read, not written; must not be tracked as a write target"
    );
}

#[test]
fn async_store_destination_is_a_write_target() {
    let node = Node::AsyncStore {
        source: Ident::from("src_buf"),
        destination: Ident::from("dst_buf"),
        offset: Box::new(Expr::u32(0)),
        size: Box::new(Expr::u32(4096)),
        tag: Ident::from("dma_tag"),
    };
    let targets = scan(&node);
    assert!(targets.writes.contains(&Ident::from("dst_buf")));
    assert!(!targets.writes.contains(&Ident::from("src_buf")));
}

#[test]
fn async_load_destination_inside_nested_bodies_is_tracked() {
    // The synthesized Node::Loop body in vfs::resolve nests AsyncLoad through
    // Region/If layers; the walk must reach it.
    let node = Node::Region {
        generator: Ident::from("region"),
        source_region: None,
        body: std::sync::Arc::new(vec![Node::if_then(
            Expr::lt(Expr::InvocationId { axis: 0 }, Expr::u32(1)),
            vec![Node::AsyncLoad {
                source: Ident::from("src_buf"),
                destination: Ident::from("dst_buf"),
                offset: Box::new(Expr::var("file_hash")),
                size: Box::new(Expr::u32(4096)),
                tag: Ident::from("vfs_req"),
            }],
        )]),
    };
    assert!(scan(&node).writes.contains(&Ident::from("dst_buf")));
}
