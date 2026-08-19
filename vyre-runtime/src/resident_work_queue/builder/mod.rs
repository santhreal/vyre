//! IR program builders  -  construct the megakernel `Program` from vyre IR.
//!
//! Two flavours:
//! - **Interpreted** (`build_program_sharded`)  -  If-tree opcode dispatch.
//! - **JIT** (`build_program_jit`)  -  payload processor fused directly.

use std::sync::Arc;

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

use super::atomic_relaxed::atomic_load_relaxed;
use super::handlers::{claimed_slot_bindings, claimed_slot_body, load_miss_body, OpcodeHandler};
use super::io::io_completion_poll_body;
use super::protocol::*;
use super::workspace_adapter::ResidentWorkspaceAdapter;
mod cache;
mod jit;
mod priority;
pub use jit::{build_program_jit, build_program_jit_slots, persistent_body_jit};
pub use priority::{
    build_program_priority, build_program_priority_slots, persistent_body_priority,
    persistent_body_priority_slots,
};

/// Build the default megakernel IR (256 lanes × 1 workgroup, no custom opcodes).
#[must_use]
pub fn build_program() -> Program {
    build_program_sharded(256, &[])
}

/// Build the megakernel IR with a custom workgroup size and optional
/// custom opcodes.
///
/// Buffers are declared with concrete `with_count(...)` sizes so the
/// backend readback layer allocates the right static staging size  -  a
/// `count=0` default reads back 4 bytes regardless of how much the
/// kernel wrote.
#[must_use]
pub fn build_program_sharded(workgroup_size_x: u32, opcodes: &[OpcodeHandler]) -> Program {
    build_program_sharded_slots(workgroup_size_x, workgroup_size_x.max(1), opcodes)
}

/// Build the megakernel IR for an explicit number of ring slots.
///
/// This is the production sharded ABI: `slot_count` sizes the ring buffer,
/// while `workgroup_size_x` controls lanes per workgroup. Dispatch must launch
/// `slot_count / workgroup_size_x` workgroups so every slot has an owning lane.
#[must_use]
pub fn build_program_sharded_slots(
    workgroup_size_x: u32,
    slot_count: u32,
    opcodes: &[OpcodeHandler],
) -> Program {
    build_program_sharded_slots_with_io(workgroup_size_x, slot_count, opcodes, false)
}

/// Build the sharded megakernel IR as a shared immutable template.
///
/// Empty opcode sets use the thread-local template cache directly, allowing
/// compile paths to avoid cloning the cached Program before wrapping it in
/// `Arc` again.
#[must_use]
pub fn build_program_sharded_slots_shared(
    workgroup_size_x: u32,
    slot_count: u32,
    opcodes: &[OpcodeHandler],
) -> Arc<Program> {
    if opcodes.is_empty() {
        return cache::cached_empty_sharded_program_shared(workgroup_size_x, slot_count, false);
    }
    Arc::new(build_program_sharded_slots(
        workgroup_size_x,
        slot_count,
        opcodes,
    ))
}

/// Build the sharded megakernel IR with a consumer-owned resident workspace.
#[must_use]
pub fn build_program_sharded_with_workspace_adapter(
    workgroup_size_x: u32,
    slot_count: u32,
    opcodes: &[OpcodeHandler],
    adapter: &impl ResidentWorkspaceAdapter,
) -> Program {
    wrap_persistent_megakernel_program_with_buffers(
        default_buffers_with_workspace_adapter(slot_count, adapter),
        workgroup_size_x,
        persistent_body_with_workspace_adapter(workgroup_size_x, opcodes, adapter),
    )
}

/// Build a finite one-pass sharded megakernel IR for host-submitted batches.
///
/// Unlike [`build_program_sharded_slots`], this program does not wrap the body
/// in `Node::forever`; each lane attempts to drain its owning slot once and the
/// dispatch returns. Use this for synchronous batch APIs that need a completion
/// report from the same queue submission.
#[must_use]
pub fn build_program_sharded_once_slots(
    workgroup_size_x: u32,
    slot_count: u32,
    opcodes: &[OpcodeHandler],
) -> Program {
    if opcodes.is_empty() {
        return cache::cached_empty_sharded_once_program(workgroup_size_x, slot_count);
    }
    wrap_megakernel_program(
        workgroup_size_x,
        slot_count,
        finite_body_with_io(workgroup_size_x, opcodes, false),
    )
}

/// Shared-Arc variant of [`build_program_sharded_once_slots`] for hot runtime
/// dispatchers that must not clone the megakernel template every launch.
#[must_use]
pub fn build_program_sharded_once_slots_shared(
    workgroup_size_x: u32,
    slot_count: u32,
    opcodes: &[OpcodeHandler],
) -> Arc<Program> {
    if opcodes.is_empty() {
        return cache::cached_empty_sharded_once_program_shared(workgroup_size_x, slot_count);
    }
    Arc::new(build_program_sharded_once_slots(
        workgroup_size_x,
        slot_count,
        opcodes,
    ))
}

/// Build a finite one-pass megakernel that reports completion through the
/// control buffer only.
///
/// Ring, debug, and IO buffers remain read-write device buffers, but their
/// host readback ranges are empty. This is the hot dispatcher path: completion
/// is already accumulated into control, so reading back the full ring/debug/IO
/// surfaces is redundant launch latency.
#[must_use]
pub fn build_program_sharded_once_slots_control_report_shared(
    workgroup_size_x: u32,
    slot_count: u32,
    opcodes: &[OpcodeHandler],
) -> Arc<Program> {
    if opcodes.is_empty() {
        return cache::cached_empty_sharded_once_control_report_program_shared(
            workgroup_size_x,
            slot_count,
        );
    }
    let mut buffers = default_buffers(slot_count);
    for buffer in buffers.iter_mut().skip(1) {
        buffer.output_byte_range = Some(0..0);
    }
    Arc::new(prepare_megakernel_program(Program::wrapped(
        buffers,
        [workgroup_size_x, 1, 1],
        finite_body_with_io(workgroup_size_x, opcodes, false),
    )))
}

/// Build the megakernel IR without the IO polling sidecar.
///
/// This is the dispatch path for host-provided [`crate::resident_work_queue::planner::ResidentWorkItem`]
/// queues. It keeps the executable kernel free of `AsyncLoad` nodes until the
/// runtime scheduler owns a concrete async-lowering pass.
#[must_use]
pub fn build_program_sharded_no_io(workgroup_size_x: u32, opcodes: &[OpcodeHandler]) -> Program {
    build_program_sharded_slots(workgroup_size_x, workgroup_size_x.max(1), opcodes)
}

/// Build the megakernel IR with the IO completion polling sidecar.
///
/// The lane body gains the IO queue recycler: every iteration scans the
/// compiled poll window and returns each slot the host completed with
/// `io_status::OK` or `io_status::ERROR` to `slot::EMPTY`. The device side
/// never claims a published request and never services the transfer; the
/// host pump owns both, so the returned program contains no async transfer
/// and needs no async-lowering pass.
#[must_use]
pub fn build_program_sharded_with_io_polling(
    workgroup_size_x: u32,
    opcodes: &[OpcodeHandler],
) -> Program {
    build_program_sharded_slots_with_io(workgroup_size_x, workgroup_size_x.max(1), opcodes, true)
}

/// Build the megakernel IR with a self-loading load-miss handler.
///
/// The persistent loop is extended with an [`opcode::LOAD_MISS`] handler.
/// When the GPU sees this opcode it scans the IO queue for an empty slot,
/// writes a DMA-read request, and polls until the host/runtime marks it
/// complete. The `arg0` field of the slot is the consumer's opaque
/// resource identifier; vyre does not interpret it.
#[must_use]
#[cfg(test)]
pub fn build_program_with_self_loading_miss_handler(
    workgroup_size_x: u32,
    slot_count: u32,
    opcodes: &[OpcodeHandler],
) -> Program {
    match try_build_program_with_self_loading_miss_handler(workgroup_size_x, slot_count, opcodes) {
        Ok(program) => program,
        Err(error) => panic!("{error}"),
    }
}

/// Fallible variant of `build_program_with_self_loading_miss_handler` (test-only panic shim exists; production uses this fallible entry).
pub fn try_build_program_with_self_loading_miss_handler(
    workgroup_size_x: u32,
    slot_count: u32,
    opcodes: &[OpcodeHandler],
) -> Result<Program, String> {
    let mut extended = Vec::new();
    let extended_len = opcodes.len().checked_add(1).ok_or_else(|| {
        "megakernel self-loading opcode extension count overflowed usize. Fix: split opcode handler sets before building the megakernel."
            .to_string()
    })?;
    vyre_foundation::allocation::try_reserve_vec_to_capacity(&mut extended, extended_len).map_err(|error| {
        format!(
            "megakernel self-loading opcode extension allocation failed: {error}. Fix: split opcode handler sets before building the megakernel."
        )
    })?;
    extended.extend_from_slice(opcodes);
    extended.push(OpcodeHandler {
        opcode: super::protocol::opcode::LOAD_MISS,
        body: load_miss_body(),
    });
    Ok(wrap_persistent_megakernel_program(
        workgroup_size_x,
        slot_count,
        persistent_body_with_io(workgroup_size_x, &extended, false),
    ))
}

fn build_program_sharded_slots_with_io(
    workgroup_size_x: u32,
    slot_count: u32,
    opcodes: &[OpcodeHandler],
    include_io_polling: bool,
) -> Program {
    if opcodes.is_empty() {
        return cache::cached_empty_sharded_program(
            workgroup_size_x,
            slot_count,
            include_io_polling,
        );
    }
    wrap_persistent_megakernel_program(
        workgroup_size_x,
        slot_count,
        persistent_body_with_io(workgroup_size_x, opcodes, include_io_polling),
    )
}

fn wrap_persistent_megakernel_program(
    workgroup_size_x: u32,
    slot_count: u32,
    body: Vec<Node>,
) -> Program {
    wrap_megakernel_program(workgroup_size_x, slot_count, vec![Node::forever(body)])
}

fn wrap_persistent_megakernel_program_with_buffers(
    buffers: Vec<BufferDecl>,
    workgroup_size_x: u32,
    body: Vec<Node>,
) -> Program {
    prepare_megakernel_program(Program::wrapped(
        buffers,
        [workgroup_size_x, 1, 1],
        vec![Node::forever(body)],
    ))
}

fn wrap_megakernel_program(workgroup_size_x: u32, slot_count: u32, body: Vec<Node>) -> Program {
    prepare_megakernel_program(Program::wrapped(
        default_buffers(slot_count),
        [workgroup_size_x, 1, 1],
        body,
    ))
}

fn prepare_megakernel_program(program: Program) -> Program {
    // Barrier elision is infallible because its working buffers are bounded by
    // the IR node count. Semantic optimization runs once in `lower_verified`.
    super::planner::elide_value_flow_barriers(program).0
}

/// Reserve sizes for the megakernel's four host-visible buffers. All
/// four go through the static-readback path so every buffer needs
/// a concrete `count` (u32 elements). The numbers mirror the wire
/// layout in `protocol.rs`:
///
/// - **control**: 128 u32 words covers SHUTDOWN, DONE_COUNT, EPOCH,
///   METRICS_BASE..METRICS_BASE+METRICS_SLOTS, OBSERVABLE_BASE, and
///   the 32-entry tenant-mask table.
/// - **ring_buffer**: `slot_count` slots × `SLOT_WORDS`.
///   `slot_count` must match host-published ring bytes and dispatch geometry.
/// - **debug_log**: cursor word + `debug::RECORD_CAPACITY` × 4-word records.
/// - **io_queue**: 64 slots × 8 words (source, destination,
///   offset_low, offset_high, size, status, tag, pad).
fn default_buffers(slot_count: u32) -> Vec<BufferDecl> {
    let ring_slots = slot_count.max(1);
    let control = BufferDecl::read_write("control", 0, DataType::U32).with_count(CONTROL_MIN_WORDS);
    let ring_buffer = BufferDecl::read_write("ring_buffer", 1, DataType::U32)
        .with_count(ring_slots.saturating_mul(SLOT_WORDS));
    let debug_log =
        BufferDecl::read_write("debug_log", 2, DataType::U32).with_count(debug::BUFFER_WORDS);
    let io_queue = BufferDecl::read_write("io_queue", 3, DataType::U32).with_count(64 * 8);
    vec![control, ring_buffer, debug_log, io_queue]
}

fn default_buffers_with_workspace_adapter(
    slot_count: u32,
    adapter: &impl ResidentWorkspaceAdapter,
) -> Vec<BufferDecl> {
    let mut buffers = default_buffers(slot_count);
    buffers.push(adapter.buffer_decl());
    buffers
}

/// The body that runs once per iteration per lane. Exposed for tests
/// and downstream crates that splice additional opcodes.
#[must_use]
pub fn persistent_body(workgroup_size_x: u32, opcodes: &[OpcodeHandler]) -> Vec<Node> {
    persistent_body_with_io(workgroup_size_x, opcodes, false)
}

/// Fallible persistent body builder with explicit staging-allocation reporting.
pub fn try_persistent_body(
    workgroup_size_x: u32,
    opcodes: &[OpcodeHandler],
) -> Result<Vec<Node>, String> {
    try_persistent_body_with_io(workgroup_size_x, opcodes, false)
}

fn persistent_body_with_io(
    workgroup_size_x: u32,
    opcodes: &[OpcodeHandler],
    include_io_polling: bool,
) -> Vec<Node> {
    assemble_lane_body(
        persistent_lane_prologue(workgroup_size_x),
        execute_slot_body(opcodes),
        include_io_polling,
    )
}

fn finite_body_with_io(
    workgroup_size_x: u32,
    opcodes: &[OpcodeHandler],
    include_io_polling: bool,
) -> Vec<Node> {
    assemble_lane_body(
        vec![Node::let_bind("lane_id", lane_id_expr(workgroup_size_x))],
        execute_slot_body(opcodes),
        include_io_polling,
    )
}

fn try_persistent_body_with_io(
    workgroup_size_x: u32,
    opcodes: &[OpcodeHandler],
    include_io_polling: bool,
) -> Result<Vec<Node>, String> {
    try_assemble_lane_body(
        persistent_lane_prologue(workgroup_size_x),
        execute_slot_body(opcodes),
        include_io_polling,
        "megakernel persistent body",
        "reduce fused IO/body staging before building the megakernel",
    )
}

/// Assemble one lane body: `prologue`, the slot-base binding, the slot body,
/// and the IO polling block when `include_io_polling`.
///
/// Reservation here is best effort: a failure costs a later reallocation and
/// nothing else, so it is absorbed. Callers that must report it instead use
/// [`try_assemble_lane_body`].
pub(super) fn assemble_lane_body(
    mut prologue: Vec<Node>,
    slot_body: Vec<Node>,
    include_io_polling: bool,
) -> Vec<Node> {
    if let Some(capacity) = lane_body_capacity(prologue.len(), include_io_polling) {
        let _ = vyre_foundation::allocation::try_reserve_vec_to_capacity(&mut prologue, capacity);
    }
    push_lane_body(&mut prologue, slot_body, include_io_polling);
    prologue
}

/// [`assemble_lane_body`] with the node reservation reported rather than
/// absorbed. `subject` names the body being built and `fix` states the
/// corrective action, both quoted verbatim into the error.
pub(super) fn try_assemble_lane_body(
    mut prologue: Vec<Node>,
    slot_body: Vec<Node>,
    include_io_polling: bool,
    subject: &str,
    fix: &str,
) -> Result<Vec<Node>, String> {
    let capacity = lane_body_capacity(prologue.len(), include_io_polling)
        .ok_or_else(|| format!("{subject} node reservation overflowed usize. Fix: {fix}."))?;
    vyre_foundation::allocation::try_reserve_vec_to_capacity(&mut prologue, capacity)
        .map_err(|error| format!("{subject} node reservation failed: {error}. Fix: {fix}."))?;
    push_lane_body(&mut prologue, slot_body, include_io_polling);
    Ok(prologue)
}

fn lane_body_capacity(prologue_len: usize, include_io_polling: bool) -> Option<usize> {
    prologue_len.checked_add(if include_io_polling { 3 } else { 2 })
}

fn push_lane_body(body: &mut Vec<Node>, slot_body: Vec<Node>, include_io_polling: bool) {
    body.push(direct_slot_base_binding());
    body.push(Node::Block(slot_body));
    if include_io_polling {
        body.push(Node::Block(io_completion_poll_body()));
    }
}

fn persistent_lane_prologue(workgroup_size_x: u32) -> Vec<Node> {
    vec![
        Node::let_bind(
            "shutdown_flag",
            atomic_load_relaxed("control", Expr::u32(control::SHUTDOWN)),
        ),
        Node::if_then(
            Expr::ne(Expr::var("shutdown_flag"), Expr::u32(0)),
            vec![Node::Return],
        ),
        Node::let_bind("lane_id", lane_id_expr(workgroup_size_x)),
    ]
}

fn direct_slot_base_binding() -> Node {
    Node::let_bind(
        "slot_base",
        Expr::mul(Expr::var("lane_id"), Expr::u32(SLOT_WORDS)),
    )
}

fn slot_tenant_id_load() -> Expr {
    Expr::load(
        "ring_buffer",
        Expr::add(Expr::var("slot_base"), Expr::u32(TENANT_WORD)),
    )
}

fn tenant_authorized_body(tenant_id: Expr, authorized_body: Vec<Node>) -> Vec<Node> {
    vec![
        Node::let_bind("tenant_id", tenant_id),
        Node::let_bind(
            "tenant_base",
            atomic_load_relaxed("control", Expr::u32(control::TENANT_BASE)),
        ),
        Node::let_bind(
            "tenant_mask",
            atomic_load_relaxed(
                "control",
                Expr::add(Expr::var("tenant_base"), Expr::var("tenant_id")),
            ),
        ),
        Node::if_then(
            Expr::ne(Expr::var("tenant_mask"), Expr::u32(0)),
            authorized_body,
        ),
    ]
}

fn lane_id_expr(workgroup_size_x: u32) -> Expr {
    Expr::add(
        Expr::mul(Expr::workgroup_x(), Expr::u32(workgroup_size_x)),
        Expr::local_x(),
    )
}

fn persistent_body_with_workspace_adapter(
    workgroup_size_x: u32,
    opcodes: &[OpcodeHandler],
    adapter: &impl ResidentWorkspaceAdapter,
) -> Vec<Node> {
    let mut body = adapter.bootstrap_nodes();
    body.extend(adapter.guard_nodes());
    body.extend(adapter.dispatch_nodes());
    body.extend(persistent_body_with_io(workgroup_size_x, opcodes, false));
    body
}

fn execute_slot_body(opcodes: &[OpcodeHandler]) -> Vec<Node> {
    execute_published_slot_body(claimed_slot_body(opcodes))
}

/// Read the lane's slot status and, when it is PUBLISHED, run `claimed_body`
/// behind the tenant-authorized claim.
pub(super) fn execute_published_slot_body(claimed_body: Vec<Node>) -> Vec<Node> {
    vec![
        Node::let_bind(
            "status_index",
            Expr::add(Expr::var("slot_base"), Expr::u32(STATUS_WORD)),
        ),
        Node::let_bind(
            "observed_status",
            atomic_load_relaxed("ring_buffer", Expr::var("status_index")),
        ),
        Node::if_then(
            Expr::eq(Expr::var("observed_status"), Expr::u32(slot::PUBLISHED)),
            tenant_authorized_claim_body(slot_tenant_id_load(), claimed_body),
        ),
    ]
}

fn tenant_authorized_claim_body(tenant_id: Expr, claimed_body: Vec<Node>) -> Vec<Node> {
    tenant_authorized_body(
        tenant_id,
        vec![
            // CAS PUBLISHED -> CLAIMED after authorization. This keeps
            // disabled tenants visible to the host instead of converting
            // their slots into stuck CLAIMED work.
            Node::let_bind(
                "prev_status",
                Expr::atomic_compare_exchange(
                    "ring_buffer",
                    Expr::var("status_index"),
                    Expr::u32(slot::PUBLISHED),
                    Expr::u32(slot::CLAIMED),
                ),
            ),
            Node::if_then(
                Expr::eq(Expr::var("prev_status"), Expr::u32(slot::PUBLISHED)),
                claimed_body,
            ),
        ],
    )
}

fn execute_already_claimed_slot_body(tenant_id: Expr, claimed_body: Vec<Node>) -> Vec<Node> {
    let mut body = vec![Node::let_bind(
        "status_index",
        Expr::add(Expr::var("slot_base"), Expr::u32(STATUS_WORD)),
    )];
    body.extend(tenant_authorized_body(tenant_id, claimed_body));
    body
}

// Inline: covers the crate-private `body_preorder::walk_body_preorder` and
// `let_names_preorder`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::super::body_preorder::{let_names_preorder, walk_body_preorder};
    use super::*;

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

    /// WHY: the IO polling megakernel serviced the DMA itself. It CAS-claimed a
    /// PUBLISHED slot the host pump also claims, read the transfer offset and
    /// size out of a GPU-writable queue word, started a workgroup-collective
    /// transfer under the divergent control flow of the single CAS winner, and
    /// published `slot::DONE`, a status no reader in the IO protocol accepts:
    /// the requesting lane in the LOAD_MISS handler polls for `io_status::OK`,
    /// so it spun to `u32::MAX` and the slot never returned to EMPTY. The IR
    /// validator refuses that offset and size under V139, so three builders
    /// produced a program the crate could build and never run.
    ///
    /// Closes: for every builder the module re-exports, a program the IR
    /// validator refuses and any device-side async transfer start; plus the
    /// status vocabulary and the claim the sidecar writes into an IO slot.
    ///
    /// Does not catch: an IO slot leak the validator cannot see, such as a host
    /// completion no lane ever recycles. Slot lifetime is host-observable state
    /// and belongs to the queue tests in `super::super::io`.
    mod built_program_contracts {
        use super::*;

        const PAIRING_WORKSPACE_BUFFER: &str = "pairing_resident_workspace";

        /// Contributes one workspace buffer and one bootstrap store, so the
        /// adapter path builds the same shape a consumer's adapter would.
        struct PairingWorkspaceAdapter;

        impl ResidentWorkspaceAdapter for PairingWorkspaceAdapter {
            fn buffer_decl(&self) -> BufferDecl {
                BufferDecl::output(PAIRING_WORKSPACE_BUFFER, 15, DataType::U32).with_count(4)
            }

            fn bootstrap_nodes(&self) -> Vec<Node> {
                vec![Node::store(
                    PAIRING_WORKSPACE_BUFFER,
                    Expr::u32(0),
                    Expr::u32(0),
                )]
            }
        }

        /// Every program the module's `build_program*` re-exports can produce.
        ///
        /// Held against the re-export list by
        /// `every_exported_program_builder_is_covered`, so a new builder turns
        /// this suite RED until it is called here.
        fn built_programs() -> Vec<(&'static str, Arc<Program>)> {
            let adapter = PairingWorkspaceAdapter;
            vec![
                ("build_program", Arc::new(build_program())),
                ("build_program_jit", Arc::new(build_program_jit(64, &[]))),
                (
                    "build_program_jit_slots",
                    Arc::new(build_program_jit_slots(64, 256, &[])),
                ),
                (
                    "build_program_priority",
                    Arc::new(build_program_priority(64, &[])),
                ),
                (
                    "build_program_priority_slots",
                    Arc::new(build_program_priority_slots(64, 256, &[])),
                ),
                (
                    "build_program_sharded",
                    Arc::new(build_program_sharded(64, &[])),
                ),
                (
                    "build_program_sharded_no_io",
                    Arc::new(build_program_sharded_no_io(64, &[])),
                ),
                (
                    "build_program_sharded_once_slots",
                    Arc::new(build_program_sharded_once_slots(64, 256, &[])),
                ),
                (
                    "build_program_sharded_once_slots_control_report_shared",
                    build_program_sharded_once_slots_control_report_shared(64, 256, &[]),
                ),
                (
                    "build_program_sharded_once_slots_shared",
                    build_program_sharded_once_slots_shared(64, 256, &[]),
                ),
                (
                    "build_program_sharded_slots",
                    Arc::new(build_program_sharded_slots(64, 256, &[])),
                ),
                (
                    "build_program_sharded_slots_shared",
                    build_program_sharded_slots_shared(64, 256, &[]),
                ),
                (
                    "build_program_sharded_with_io_polling",
                    Arc::new(build_program_sharded_with_io_polling(64, &[])),
                ),
                (
                    "build_program_sharded_with_workspace_adapter",
                    Arc::new(build_program_sharded_with_workspace_adapter(
                        64,
                        256,
                        &[],
                        &adapter,
                    )),
                ),
                (
                    "build_program_with_self_loading_miss_handler",
                    Arc::new(build_program_with_self_loading_miss_handler(64, 256, &[])),
                ),
            ]
        }

        fn async_transfer_tags(nodes: &[Node]) -> Vec<&str> {
            let mut started: Vec<&str> = Vec::new();
            walk_body_preorder(nodes, &mut |node| {
                if let Node::AsyncLoad { tag, .. } | Node::AsyncStore { tag, .. } = node {
                    started.push(tag.as_str());
                }
            });
            started
        }

        #[test]
        fn every_built_program_passes_ir_validation() {
            for (name, program) in built_programs() {
                let errors = vyre_foundation::validate::validate(&program);
                assert!(
                    errors.is_empty(),
                    "Fix: `{name}` builds a program the IR validator refuses: {errors:?}"
                );
            }
        }

        #[test]
        fn no_built_program_starts_a_device_side_transfer() {
            for (name, program) in built_programs() {
                let started = async_transfer_tags(program.entry());
                assert!(
                    started.is_empty(),
                    "Fix: `{name}` starts async transfer tags {started:?} on the device. The host pump claims a published IO slot and services the copy, so a device-side request is published into the queue and its completion polled instead."
                );
            }
        }

        #[test]
        fn every_exported_program_builder_is_covered() {
            let mut exported: Vec<&str> = include_str!("../mod.rs")
                .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                .filter(|token| token.starts_with("build_program"))
                .collect();
            exported.sort_unstable();
            exported.dedup();

            let mut covered: Vec<&str> =
                built_programs().into_iter().map(|(name, _)| name).collect();
            covered.sort_unstable();

            assert_eq!(
                exported, covered,
                "Fix: call every re-exported `build_program*` builder in built_programs; an uncalled one has no validation or device-transfer coverage."
            );
        }

        #[test]
        fn the_io_sidecar_recycles_completions_without_claiming_them() {
            let program = build_program_sharded_with_io_polling(64, &[]);

            let mut atomics: Vec<String> = Vec::new();
            vyre_foundation::visit::walk_exprs(&program, |expr| {
                if let Expr::Atomic { op, buffer, .. } = expr {
                    if buffer.as_str() == "io_queue" {
                        atomics.push(format!("{op:?}"));
                    }
                }
            });
            assert!(
                atomics.is_empty(),
                "Fix: the IO sidecar runs {atomics:?} on io_queue. Claiming a published request is the host pump's transition, and a device claim the host never sees strands the request."
            );

            let mut written: Vec<Expr> = Vec::new();
            walk_body_preorder(program.entry(), &mut |node| {
                if let Node::Store { buffer, value, .. } = node {
                    if buffer.as_str() == "io_queue" {
                        written.push(value.clone());
                    }
                }
            });
            assert_eq!(
                written,
                vec![Expr::u32(slot::EMPTY)],
                "Fix: the IO sidecar must write exactly one IO slot status, `slot::EMPTY`, recycling a slot the host completed. Any other value is a device-side lifecycle transition the host protocol has no reader for."
            );
        }
    }
}
