use super::super::body_preorder::{let_names_preorder, walk_body_preorder};
use super::*;

fn async_load_bindings(nodes: &[Node]) -> Vec<(String, String, String)> {
    let mut bindings = Vec::new();
    walk_body_preorder(nodes, &mut |node| {
        if let Node::AsyncLoad {
            source,
            destination,
            tag,
            ..
        } = node
        {
            bindings.push((
                source.as_str().to_string(),
                destination.as_str().to_string(),
                tag.as_str().to_string(),
            ));
        }
    });
    bindings
}

#[test]
fn io_polling_uses_capability_tables_not_fake_resource_names() {
    let program = build_program_sharded_with_io_polling(64, &[]);
    let bindings = async_load_bindings(&program.entry);
    assert_eq!(bindings.len(), 1);
    let (source, destination, tag) = &bindings[0];
    assert_eq!(source, "io_source_capability_table");
    assert_eq!(destination, "io_destination_capability_table");
    assert_eq!(tag, "io_queue_dma");
    assert_ne!(source, "ssd_weights");
    assert_ne!(destination, "vram_cache");
}

#[test]
fn priority_builder_declares_explicit_ring_slots() {
    let program = build_program_priority_slots(64, 512, &[]);
    let ring = program
        .buffer("ring_buffer")
        .expect("Fix: priority megakernel must declare the ring buffer");
    assert_eq!(ring.count, 512 * SLOT_WORDS);
}

#[test]
fn direct_megakernel_defers_tenant_loads_until_status_is_published() {
    let body = persistent_body(64, &[]);
    let top_level_lets = body
        .iter()
        .filter_map(|node| match node {
            Node::Let { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
            top_level_lets,
            vec!["shutdown_flag", "lane_id", "slot_base"],
            "Fix: the persistent megakernel prologue must not load tenant metadata before proving the slot is claimable."
        );

    let names = let_names_preorder(&body);
    let observed = names
        .iter()
        .position(|name| *name == "observed_status")
        .expect("Fix: status load must gate the claim path");
    let tenant_mask = names
        .iter()
        .position(|name| *name == "tenant_mask")
        .expect("Fix: tenant authorization must still exist for published slots");
    assert!(
            observed < tenant_mask,
            "Fix: idle megakernel slots must skip tenant table loads; observed_status appears at {observed}, tenant_mask at {tenant_mask}."
        );
}

#[test]
fn empty_sharded_shared_builder_reuses_cached_program_arc() {
    let first = build_program_sharded_slots_shared(64, 256, &[]);
    let second = build_program_sharded_slots_shared(64, 256, &[]);

    assert!(
            Arc::ptr_eq(&first, &second),
            "Fix: empty megakernel template bootstraps must reuse the cached Arc<Program> instead of cloning the Program before compile."
        );
}

#[test]
fn empty_sharded_once_shared_builder_reuses_cached_program_arc() {
    let first = build_program_sharded_once_slots_shared(64, 256, &[]);
    let second = build_program_sharded_once_slots_shared(64, 256, &[]);

    assert!(
            Arc::ptr_eq(&first, &second),
            "Fix: one-shot megakernel dispatchers must reuse the cached Arc<Program> instead of rebuilding or cloning the Program on the hot path."
        );
}

#[test]
fn self_loading_miss_handler_program_contains_load_miss_bindings() {
    let program = build_program_with_self_loading_miss_handler(64, 256, &[]);
    let names = let_names_preorder(program.entry());
    assert!(
        names.iter().any(|n| *n == "resource_id"),
        "Fix: self-loading miss handler must bind resource_id (the \
         opaque consumer-defined identifier the IO queue carries)"
    );
    assert!(
        names.iter().any(|n| *n == "found_io_slot"),
        "Fix: self-loading miss handler must scan for an empty IO slot"
    );
    assert!(
        names.iter().any(|n| *n == "poll_done"),
        "Fix: self-loading miss handler must poll for DMA completion"
    );
}

#[test]
fn self_loading_miss_handler_does_not_include_async_load_nodes() {
    let program = build_program_with_self_loading_miss_handler(64, 256, &[]);
    let bindings = async_load_bindings(program.entry());
    assert_eq!(
        bindings.len(),
        0,
        "Fix: self-loading miss handler must not introduce AsyncLoad nodes; it writes to the IO queue and polls instead."
    );
}
