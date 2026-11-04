//! Core `fuse_programs` family + multi-program implementation.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::execution_plan::SchedulingPolicy;
use crate::ir::{BinOp, BufferAccess, BufferDecl, Expr, Ident, Node, Program};
use crate::visit::{any_descendant, for_each_expr};

use super::alpha_rename::{multiply_declared_names, push_alpha_renamed_arm_entry_node, ArmRenamer};
use super::collectors::collect_buffer_targets;
use super::divergence::{
    has_divergent_invocation_gated_store, has_launch_geometry_dependent_write,
};
use super::{
    FusionError, FusionOverDispatchError, FusionSelfAliasingError, FusionWorkgroupGeometryError,
};

/// Combine `programs` into one fused [`Program`]. Returns the input verbatim
/// for 0 or 1 program; multi-program runs go through the full hazard tracker.
///
/// # Errors
///
/// Returns [`FusionError`] when the batch contains conflicting buffer aliases,
/// non-composable self-fusion, or over-dispatches the shared launch geometry.
pub fn fuse_programs(programs: &[Program]) -> Result<Program, FusionError> {
    match programs.len() {
        0 => Ok(Program::empty()),
        1 => Ok(programs[0].clone()),
        _ => fuse_programs_multi(programs),
    }
}

/// Fuse `programs` when the caller already owns a `Vec`.
///
/// For a single program this returns that value directly (no deep clone).
/// Multi-arm batches delegate to the same implementation as [`fuse_programs`].
///
/// # Errors
///
/// Returns [`FusionError`] under the same conditions as [`fuse_programs`].
#[inline]
#[must_use]
pub fn fuse_programs_vec(mut programs: Vec<Program>) -> Result<Program, FusionError> {
    match programs.len() {
        0 => Ok(Program::empty()),
        1 => {
            let Some(program) = programs.pop() else {
                return Ok(Program::empty());
            };
            Ok(program)
        }
        _ => fuse_programs_multi(programs.as_slice()),
    }
}

/// How a fused arm's local names and scope relate to the other arms.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArmNamespace {
    /// Arms are **independent** programs (inter-rule batch fusion, the
    /// megakernel builder). Each arm allocates temps from its own counter,
    /// so two arms can reuse the same temp name for different values. The
    /// fuser alpha-renames every arm-local name with the arm index and wraps
    /// each arm body in its own `Block` scope, so the reused names cannot
    /// collide in the combined program.
    Isolated,
    /// Arms are **sub-programs of one rule** that share a single global temp
    /// namespace (one monotonic `temp_counter` per `LowerCtx`; recursion is
    /// handled by the fixpoint operator, not by re-instantiating bodies, so
    /// no name is ever reused for two values). Renaming such names is not
    /// only unnecessary, it is actively wrong: a value produced in one arm
    /// (`let __cmp_N = load(__quant_flag_…)`) and consumed in another arm
    /// (`Var(__cmp_N)`) must keep ONE consistent name and live in ONE shared
    /// scope, or the consumer references an undeclared variable. Shared arms
    /// are therefore spliced flat, no per-arm rename, no per-arm `Block`
    /// preserving decl→use linkage across the merge boundary.
    Shared,
}

/// Merge sub-programs that share one rule's global temp namespace into a
/// single program, preserving decl→use linkage across the merge boundary.
///
/// Same hazard analysis, buffer union, binding renumbering, and barrier
/// insertion as [`fuse_programs`]; the only difference is that arm-local
/// names and scopes are **shared**, not isolated (see the internal `ArmNamespace`).
/// This is the correct primitive for shared-scope composition where the merged
/// arms must reference each other's local names, alpha-renaming would desync a
/// flag/readback from its in-program consumer.
///
/// # Errors
///
/// Returns [`FusionError`] under the same conditions as [`fuse_programs`].
pub fn merge_programs_shared(programs: &[Program]) -> Result<Program, FusionError> {
    match programs.len() {
        0 => Ok(Program::empty()),
        1 => Ok(programs[0].clone()),
        _ => fuse_programs_multi_with(programs, ArmNamespace::Shared),
    }
}

fn fuse_programs_multi(programs: &[Program]) -> Result<Program, FusionError> {
    fuse_programs_multi_with(programs, ArmNamespace::Isolated)
}

fn fuse_programs_multi_with(
    programs: &[Program],
    namespace: ArmNamespace,
) -> Result<Program, FusionError> {
    reject_non_composable_self_fusion(programs)?;

    // ------------------------------------------------------------------
    // Single pass over programs: collect entries, atomics, buffers,
    // hazards, and workgroup size in one go.
    // ------------------------------------------------------------------
    let mut merged_buffers: Vec<BufferDecl> = Vec::new();
    let mut name_to_index: FxHashMap<Ident, usize> = FxHashMap::default();
    let mut next_binding = 0_u32;

    let mut read_arms_per_buffer: FxHashMap<Ident, Vec<usize>> = FxHashMap::default();
    // Track write-arm history per buffer so a later READER can force
    // a barrier after the earlier writer. Without this, the fused
    // kernel runs writer + reader in the same launch with no
    // synchronization, and the reader sees stale data from threads
    // that haven't completed the writer's body yet  -  the exact
    // "stack_overflow_gets misses node 39" mode.
    let mut write_arms_per_buffer: FxHashMap<Ident, Vec<usize>> = FxHashMap::default();
    let mut barrier_after_arm: FxHashSet<usize> = FxHashSet::default();
    // Arms whose writes are derived from launch geometry need a grid-level
    // fence before later arms read them. A workgroup barrier waits only for
    // the current block, so it cannot order "block 0 writes offsets, block 1
    // reads offsets" shapes inside a fused launch.
    let mut grid_sync_writer_arms: FxHashSet<usize> = FxHashSet::default();

    let mut fused_workgroup = [1u32, 1, 1];
    let mut max_arm_threads: u64 = 1;

    let mut arm_entries: Vec<Vec<Node>> = Vec::with_capacity(programs.len());

    // Shared-namespace merge prefixes ONLY names declared in ≥2 arms (genuine
    // collisions, e.g. a primitive's internal `acc`). A name declared in
    // exactly one arm, including a value produced in one arm and consumed in
    // another (`let __cmp_N = …` / `Var(__cmp_N)`), is globally unique and
    // stays unrenamed so the decl→use link survives. Isolated fusion renames
    // every name (the set is unused for that mode).
    let multiply_declared: FxHashSet<Ident> = match namespace {
        ArmNamespace::Isolated => FxHashSet::default(),
        ArmNamespace::Shared => {
            let entries: Vec<&[Node]> = programs.iter().map(Program::entry).collect();
            multiply_declared_names(&entries)
        }
    };

    for (arm_idx, prog) in programs.iter().enumerate() {
        // Walk entry nodes once: clone into segment and collect both
        // atomic targets (writes) and Load targets (reads). Buffers
        // referenced inside the body but NOT declared in the arm's
        // own `buffers()` table  -  produced by an earlier arm  -  only
        // surface here. Without this, RAW hazards across arms that
        // read shared scalars (e.g. broadcast reading the scalar
        // written by a single-thread `bitset_any`) get no barrier
        // and silently produce stale reads on threads that haven't
        // observed the writer's flush.
        let entry = prog.entry();
        let mut segment = Vec::with_capacity(entry.len());
        let mut atomic_targets: FxHashSet<Ident> = FxHashSet::default();
        let mut load_targets: FxHashSet<Ident> = FxHashSet::default();
        let mut store_targets: FxHashSet<Ident> = FxHashSet::default();
        let mut divergent_store_seen = false;
        for node in entry {
            match namespace {
                ArmNamespace::Isolated => {
                    push_alpha_renamed_arm_entry_node(&mut segment, node, arm_idx);
                }
                ArmNamespace::Shared => {
                    ArmRenamer::shared(arm_idx, &multiply_declared)
                        .push_entry_node(&mut segment, node);
                }
            }
            collect_buffer_targets(
                node,
                &mut load_targets,
                &mut store_targets,
                &mut atomic_targets,
            );
            if has_divergent_invocation_gated_store(node, false) {
                divergent_store_seen = true;
            }
        }
        if divergent_store_seen || has_launch_geometry_dependent_write(prog.entry()) {
            grid_sync_writer_arms.insert(arm_idx);
        }
        arm_entries.push(segment);

        let mut arm_reads: FxHashSet<Ident> = FxHashSet::default();
        let mut arm_explicit_writes: FxHashSet<Ident> = FxHashSet::default();
        classify_and_merge_arm_buffers(
            prog,
            &mut arm_reads,
            &mut arm_explicit_writes,
            &mut merged_buffers,
            &mut name_to_index,
            &mut next_binding,
        );

        // Body-level reads from buffers declared by EARLIER arms.
        // The arm's own buffers().iter() loop already populated
        // `arm_reads` for declared ReadOnly inputs; this adds any
        // additional reads inferred from `Expr::Load` references.
        for target in &load_targets {
            arm_reads.insert(target.clone());
        }
        // Body-level stores to buffers declared by earlier arms.
        for target in &store_targets {
            arm_explicit_writes.insert(target.clone());
        }

        // Atomic writes count only for buffers not already read or explicitly written.
        let mut arm_writes = arm_explicit_writes.clone();
        for target in &atomic_targets {
            if !arm_reads.contains(target) && !arm_explicit_writes.contains(target) {
                arm_writes.insert(target.clone());
            }
        }

        // F-IR-22: WAR hazard  -  for each buffer this arm writes, if
        // any previous arm read it, mark a barrier after every such
        // earlier read arm so the new write can't clobber the read.
        for write_buf in &arm_writes {
            if let Some(read_arms) = read_arms_per_buffer.get(write_buf) {
                for &read_arm in read_arms {
                    barrier_after_arm.insert(read_arm);
                }
            }
        }

        // RAW hazard  -  for each buffer this arm reads, if any
        // previous arm wrote it, the writer's results must be
        // visible before this read. Insert a barrier after every
        // such earlier writer arm. Required because the fused
        // kernel runs as one backend launch; without a barrier,
        // threads in this arm may execute the load before the
        // writer arm's threads have completed their store, yielding
        // stale data and silently dropping rule findings (recall=0
        // mode previously observed on `stack_overflow_gets` for
        // node ids past the subgroup boundary).
        for read_buf in &arm_reads {
            if let Some(write_arms) = write_arms_per_buffer.get(read_buf) {
                for &write_arm in write_arms {
                    barrier_after_arm.insert(write_arm);
                }
            }
        }

        // Update read tracking for later arms.
        for read_buf in &arm_reads {
            read_arms_per_buffer
                .entry(read_buf.clone())
                .or_default()
                .push(arm_idx);
        }
        // Update write tracking for later RAW detection.
        for write_buf in &arm_writes {
            write_arms_per_buffer
                .entry(write_buf.clone())
                .or_default()
                .push(arm_idx);
        }

        // Workgroup size tracking.
        let wg = prog.workgroup_size();
        fused_workgroup[0] = fused_workgroup[0].max(wg[0]);
        fused_workgroup[1] = fused_workgroup[1].max(wg[1]);
        fused_workgroup[2] = fused_workgroup[2].max(wg[2]);
        let arm_threads = u64::from(wg[0]) * u64::from(wg[1]) * u64::from(wg[2]);
        max_arm_threads = max_arm_threads.max(arm_threads);
    }

    reject_workgroup_geometry_change(programs, fused_workgroup)?;

    let combined_entry = flatten_arm_entries(
        arm_entries,
        &barrier_after_arm,
        &grid_sync_writer_arms,
        programs.len(),
        namespace,
    );
    reject_overdispatch(fused_workgroup, max_arm_threads)?;

    // `Program::wrapped` builds a fresh program, which resets the metadata
    // flags. `non_composable_with_self` describes the fused body just as much
    // as it described the arm it came from: the fused program now CONTAINS
    // that body, so fusing it again with another copy of the same arm carries
    // the identical hazard. Carrying the OR forward is what lets
    // `reject_non_composable_self_fusion` see it on a second round.
    //
    // `entry_op_id` is deliberately left cleared. It names one certified
    // operation, and a program built from several arms is not that operation
    // even when every arm happens to share an id.
    let non_composable = programs.iter().any(Program::is_non_composable_with_self);
    Ok(
        Program::wrapped(merged_buffers, fused_workgroup, combined_entry)
            .with_non_composable_with_self(non_composable),
    )
}

fn classify_and_merge_arm_buffers(
    prog: &Program,
    arm_reads: &mut FxHashSet<Ident>,
    arm_explicit_writes: &mut FxHashSet<Ident>,
    merged_buffers: &mut Vec<BufferDecl>,
    name_to_index: &mut FxHashMap<Ident, usize>,
    next_binding: &mut u32,
) {
    for buf in prog.buffers() {
        let name = Ident::from(buf.name());
        match buf.access() {
            BufferAccess::ReadOnly | BufferAccess::Uniform => {
                arm_reads.insert(name.clone());
            }
            BufferAccess::ReadWrite => {
                arm_explicit_writes.insert(name.clone());
            }
            _ => {}
        }
        if let Some(&idx) = name_to_index.get(&name) {
            let existing = &mut merged_buffers[idx];
            let initialized_before_use = existing.is_backend_allocated_output();
            let access = buf.access();
            upgrade_buffer_access(existing, &access);
            if buf.count > existing.count {
                existing.count = buf.count;
            }
            if buf.is_output() {
                existing.is_output = true;
                existing.pipeline_live_out = true;
            }
            if initialized_before_use && existing.access() == BufferAccess::ReadWrite {
                // A prior arm produced the storage before a later arm read it.
                // Keep that first-write fact after access widening so launch
                // planning allocates the carrier instead of demanding host bytes.
                existing.pipeline_live_out = true;
            }
        } else {
            let mut merged = buf.clone();
            if merged.access() != BufferAccess::Workgroup {
                merged.binding = *next_binding;
                *next_binding += 1;
            }
            name_to_index.insert(Ident::from(merged.name()), merged_buffers.len());
            merged_buffers.push(merged);
        }
    }
}

fn reject_non_composable_self_fusion(programs: &[Program]) -> Result<(), FusionError> {
    let mut seen_op_ids: FxHashMap<String, bool> = FxHashMap::default();
    for prog in programs {
        let key = prog
            .entry_op_id()
            .map_or_else(|| fallback_composition_key(prog), ToString::to_string);
        let is_non_comp = prog.is_non_composable_with_self();
        match seen_op_ids.get_mut(&key) {
            Some(has_non_comp) if *has_non_comp || is_non_comp => {
                return Err(FusionError::SelfAliasing(FusionSelfAliasingError {
                    op_id: key,
                    fix: "rename the second parser's workgroup buffer or split into two separate dispatches",
                }));
            }
            Some(_) => {}
            None => {
                seen_op_ids.insert(key, is_non_comp);
            }
        }
    }
    Ok(())
}

/// Refuse to widen the workgroup of an arm that reasons about its own.
///
/// The fused geometry is the axis-wise maximum over the arms. For an arm whose
/// invocations are independent that is only a launch-size change. For an arm
/// that synchronizes its workgroup or keeps state in workgroup memory it is a
/// semantic change: the arm guards its body for its own width, so under a wider
/// workgroup the extra invocations skip the guarded body and never reach the
/// barrier the working invocations wait on. A workgroup barrier that is not
/// reached by every invocation in the workgroup is undefined, and in practice
/// the result is intermittently wrong rather than reliably wrong.
///
/// The third shape needs no barrier and no workgroup memory. An arm can hold
/// its running result in read-write storage and admit it with a guard on its
/// workgroup identity alone, which is exact only while that workgroup is one
/// invocation wide. Measured: `nn::top_k` declares `[1, 1, 1]` and guards its
/// insertion scan on `workgroup_id.x == 0`; fused behind a 256-wide
/// elementwise arm, all 256 invocations of workgroup 0 ran the same insertion
/// and the result named one input lane twice. An arm that constrains the
/// invocation as well, the `workgroup_id.x == 0 && local_id.x == 0` pattern, is
/// exact under any width and stays fusable.
///
/// Failing closed is the only correct answer here. Fusion cannot rewrite the
/// arm for the wider geometry, and quietly emitting the racy kernel gives the
/// caller a program that passes most of the time.
fn reject_workgroup_geometry_change(
    programs: &[Program],
    fused_workgroup: [u32; 3],
) -> Result<(), FusionError> {
    for (arm, prog) in programs.iter().enumerate() {
        let arm_workgroup = prog.workgroup_size();
        if arm_workgroup == fused_workgroup {
            continue;
        }
        let uses_workgroup_memory = prog
            .buffers()
            .iter()
            .any(|buf| buf.access() == BufferAccess::Workgroup);
        let synchronizes = has_barrier(prog.entry());
        let serial_under_workgroup_guard = relies_on_single_invocation_workgroup(prog);
        let reason = match (
            uses_workgroup_memory,
            synchronizes,
            serial_under_workgroup_guard,
        ) {
            (true, true, _) => "keeps state in workgroup memory and synchronizes its workgroup",
            (true, false, _) => "keeps state in workgroup memory sized for its own workgroup",
            (false, true, _) => "synchronizes its workgroup with a barrier",
            (false, false, true) => {
                "keeps a running result in read-write storage under a guard on its workgroup \
                 identity, so every invocation the wider workgroup adds repeats the same \
                 read-modify-write"
            }
            (false, false, false) => continue,
        };
        return Err(FusionError::WorkgroupGeometry(
            FusionWorkgroupGeometryError {
                arm,
                arm_workgroup,
                fused_workgroup,
                reason,
                fix: "dispatch this arm separately, or rebuild it for the wider workgroup before fusing",
            },
        ));
    }
    Ok(())
}

/// Is there a barrier anywhere in this node sequence?
///
/// Descent comes from [`any_descendant`], the one owner of which node variants
/// nest. The hand-written match this replaces ended in `_ => false`, and a
/// barrier hidden inside an unrecognised nesting variant makes a fused arm look
/// unsynchronized when it is not.
fn has_barrier(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| {
        any_descendant(node, &mut |n| {
            matches!(n, Node::Barrier { .. } | Node::LogicalBarrier { .. })
        })
    })
}

/// Does this program's serial body rely on being one invocation wide?
///
/// A program that keeps a running result in read-write storage and admits that
/// body on its workgroup identity alone is exact only while the workgroup holds
/// one invocation. Run it wider, by fusion or by a rebuild, and every added
/// invocation repeats the same read-modify-write over the same slots. Naming
/// the invocation instead costs nothing in its own geometry and is exact in
/// any, so this reports a program that has not done so.
///
/// Fusion refuses to widen such an arm, and the op catalog holds every
/// registered single-invocation program to the invocation-exact form.
#[must_use]
pub fn relies_on_single_invocation_workgroup(prog: &Program) -> bool {
    keeps_state_in_read_write_storage(prog)
        && guards_on_workgroup_identity(prog)
        && !guards_on_invocation_identity(prog)
}

/// Does this arm carry a result across invocations in read-write storage?
///
/// Read-write storage is the only place an arm can keep a running value that
/// outlives one invocation without declaring workgroup memory, so it is what
/// makes a repeated body observable rather than merely wasteful.
fn keeps_state_in_read_write_storage(prog: &Program) -> bool {
    prog.buffers()
        .iter()
        .any(|buf| buf.access() == BufferAccess::ReadWrite)
}

/// Does this arm decide what to run from its physical workgroup or logical tile
/// identity?
fn guards_on_workgroup_identity(prog: &Program) -> bool {
    compares_against(prog, &|expr| {
        matches!(expr, Expr::WorkgroupId { .. } | Expr::LogicalTileId { .. })
    })
}

/// Does this arm decide what to run from its invocation identity?
///
/// Either id answers: a guard on the local invocation is exact inside one
/// workgroup, and a guard on the global invocation is exact across the launch.
fn guards_on_invocation_identity(prog: &Program) -> bool {
    compares_against(prog, &|expr| {
        matches!(
            expr,
            Expr::LocalId { .. }
                | Expr::InvocationId { .. }
                | Expr::LogicalIndex { .. }
                | Expr::LogicalWithinTileId { .. }
        )
    })
}

/// Is any identity the predicate accepts an operand of a comparison?
///
/// A comparison is what turns an id into a decision. An id used as an index is
/// not a guard: every invocation runs that body and addresses its own element,
/// which is exactly the shape a wider workgroup is safe for.
fn compares_against(prog: &Program, is_identity: &dyn Fn(&Expr) -> bool) -> bool {
    let mut found = false;
    for_each_expr(prog.entry(), |expr| {
        if let Expr::BinOp { op, left, right } = expr {
            if matches!(
                op,
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
            ) && (is_identity(left.as_ref()) || is_identity(right.as_ref()))
            {
                found = true;
            }
        }
    });
    found
}

fn flatten_arm_entries(
    arm_entries: Vec<Vec<Node>>,
    barrier_after_arm: &FxHashSet<usize>,
    grid_sync_writer_arms: &FxHashSet<usize>,
    program_count: usize,
    namespace: ArmNamespace,
) -> Vec<Node> {
    let total_nodes: usize = arm_entries.iter().map(Vec::len).sum();
    let mut combined_entry = Vec::with_capacity(total_nodes + program_count);
    for (arm_idx, segment) in arm_entries.into_iter().enumerate() {
        match namespace {
            // Isolated arms each get their own `Block` scope so reused
            // arm-local names cannot collide across arms.
            ArmNamespace::Isolated => combined_entry.push(Node::Block(segment)),
            // Shared arms splice flat into the one rule-wide scope, so a
            // `let` in an earlier arm stays visible to a later arm's use.
            ArmNamespace::Shared => combined_entry.extend(segment),
        }
        if barrier_after_arm.contains(&arm_idx) {
            // Workgroup `SeqCst` is sufficient only when the
            // prior write is uniform across the launch. Launch-geometry
            // dependent writes must become a top-level `GridSync`, where the
            // runtime split pass can lower the fused program into globally
            // ordered dispatch segments.
            let ordering = if grid_sync_writer_arms.contains(&arm_idx) {
                crate::memory_model::MemoryOrdering::GridSync
            } else {
                crate::memory_model::MemoryOrdering::SeqCst
            };
            combined_entry.push(Node::logical_barrier(ordering));
        }
    }
    combined_entry
}

fn reject_overdispatch(fused_workgroup: [u32; 3], max_arm_threads: u64) -> Result<(), FusionError> {
    let fused_threads = u64::from(fused_workgroup[0])
        * u64::from(fused_workgroup[1])
        * u64::from(fused_workgroup[2]);
    let policy = SchedulingPolicy::standard();
    if policy.allow_fused_threads(fused_threads, max_arm_threads) {
        return Ok(());
    }
    Err(FusionError::OverDispatch(FusionOverDispatchError {
        max_arm_threads,
        fused_threads,
        fix: "split the batch or use per-arm dispatch; axis-wise max exceeds the shared over-dispatch policy",
    }))
}

pub(super) fn fallback_composition_key(prog: &Program) -> String {
    let mut hasher = blake3::Hasher::new();
    for buf in prog.buffers() {
        hasher.update(buf.name().as_bytes());
        hasher.update(&[0]);
    }
    for dim in prog.workgroup_size() {
        hasher.update(&dim.to_le_bytes());
    }
    hasher.update(&(prog.entry().len() as u64).to_le_bytes());
    format!("{}", hasher.finalize().to_hex())
}

/// Upgrade `buffer.access` to the more permissive of the two modes.
pub(super) fn upgrade_buffer_access(buffer: &mut BufferDecl, other: &BufferAccess) {
    let current = buffer.access();
    buffer.access = match (&current, &other) {
        (BufferAccess::ReadWrite, _)
        | (_, BufferAccess::ReadWrite)
        | (BufferAccess::WriteOnly, BufferAccess::ReadOnly | BufferAccess::Uniform)
        | (BufferAccess::ReadOnly | BufferAccess::Uniform, BufferAccess::WriteOnly) => {
            BufferAccess::ReadWrite
        }
        (BufferAccess::WriteOnly, BufferAccess::WriteOnly) => BufferAccess::WriteOnly,
        (BufferAccess::Uniform, _) | (_, BufferAccess::Uniform) => BufferAccess::Uniform,
        (BufferAccess::Workgroup, _) | (_, BufferAccess::Workgroup) => BufferAccess::Workgroup,
        _ => BufferAccess::ReadOnly,
    };
    // Keep kind in sync with the upgraded access.
    buffer.kind = match buffer.access {
        BufferAccess::ReadOnly => crate::ir::MemoryKind::Readonly,
        BufferAccess::Uniform => crate::ir::MemoryKind::Uniform,
        BufferAccess::Workgroup => crate::ir::MemoryKind::Shared,
        _ => crate::ir::MemoryKind::Global,
    };
}
