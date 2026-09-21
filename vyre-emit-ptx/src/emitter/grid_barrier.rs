//! Whole-grid barrier lowering.
//!
//! Owns the monotonic-counter cooperative barrier that a
//! `MemoryOrdering::GridSync` barrier lowers to, and the refusal when the
//! barrier sits somewhere that lowering would be unsound. Workgroup-scope
//! barriers are a single instruction and are emitted from the op dispatch.

use std::fmt::Write as _;

use super::BodyCtx;
use crate::reg::PtxType;
use crate::EmitError;

impl BodyCtx<'_> {
    /// Emit a whole-grid barrier (`MemoryOrdering::GridSync`) as a
    /// monotonic-counter cooperative barrier.
    ///
    /// The kernel is launched cooperatively (every CTA co-resident), so the
    /// `i`-th grid barrier releases once the module-scope counter
    /// `_vyre_grid_barrier` reaches `(i+1) * gridSize`: i.e. every CTA has
    /// arrived at barrier `i`. The counter only grows within a launch, so the
    /// strictly increasing per-barrier release targets are race-robust: a CTA
    /// that races ahead only pushes the counter higher, never below an earlier
    /// barrier's target, so no CTA is ever falsely released. The host zeroes
    /// the counter before each cooperative launch (the fixpoint loop re-launches
    /// the kernel on one serialized stream).
    ///
    /// Per-CTA `bar.sync` brackets the global phase: the first publishes the
    /// CTA's prior global writes (`membar.gl`) and converges all lanes; the
    /// per-CTA leader records arrival and spins on the counter; the second
    /// releases the CTA once the leader observes the target, and a trailing
    /// `membar.gl` acquires the other CTAs' published writes. This is the
    /// standard sense-free grid barrier and is correct only when every CTA
    /// executes the same barrier sequence, guaranteed here because GridSync
    /// barriers are top-level (never under divergent control flow) and the
    /// full-workgroup entry keeps every lane live through the barrier.
    pub(super) fn emit_grid_sync_barrier(&mut self) -> Result<(), EmitError> {
        if self.grid_sync_loop_depth > 0 {
            return Err(EmitError::InvalidDescriptor(
                "MemoryOrdering::GridSync barrier inside a loop body cannot be lowered to the \
                 monotonic-counter cooperative grid barrier: the release target is fixed at emit \
                 time to (barrier_index + 1) * gridSize, but a loop emits its body ONCE, so every \
                 iteration after the first finds the counter already at or past that target and \
                 the barrier silently becomes a no-op, leaving the grid unsynchronized with no \
                 error. Fix: unroll the loop so each GridSync barrier is a distinct top-level \
                 barrier with its own release target (this is what \
                 vyre_libs::fixpoint::persistent_fixpoint::persistent_fixpoint_grid does, \
                 emitting max_iterations top-level waves), or split the program at the barrier and \
                 let the host loop re-launch each segment."
                    .to_string(),
            ));
        }
        let index = self.grid_barrier_index;
        self.grid_barrier_index = index.checked_add(1).ok_or_else(|| {
            EmitError::InvalidDescriptor("grid-sync barrier index overflow".into())
        })?;
        // Release target multiplier for this barrier: (index + 1).
        let multiplier = index.checked_add(1).ok_or_else(|| {
            EmitError::InvalidDescriptor("grid-sync barrier target overflow".into())
        })?;

        let sum = self.alloc(PtxType::U32);
        let tmp = self.alloc(PtxType::U32);
        let grid = self.alloc(PtxType::U32);
        let dim = self.alloc(PtxType::U32);
        let target = self.alloc(PtxType::U32);
        let old = self.alloc(PtxType::U32);
        let cur = self.alloc(PtxType::U32);
        let bar_addr = self.alloc(PtxType::U64);
        let not_leader = self.alloc(PtxType::Bool);
        let waiting = self.alloc(PtxType::Bool);
        let spin = self.alloc_label("grid_sync_spin");
        let skip = self.alloc_label("grid_sync_skip");

        let _ = writeln!(
            self.text,
            "    // --- grid.sync barrier #{index} (monotonic counter, release at {multiplier}*gridSize) ---"
        );
        // Publish this CTA's prior global writes, then converge every lane so
        // all writes have flushed before the leader records arrival.
        let _ = writeln!(self.text, "    membar.gl;");
        let _ = writeln!(self.text, "    bar.sync 0;");
        // Per-CTA leader = (tid.x | tid.y | tid.z) == 0.
        let _ = writeln!(self.text, "    mov.u32 {sum}, %tid.x;");
        let _ = writeln!(self.text, "    mov.u32 {tmp}, %tid.y;");
        let _ = writeln!(self.text, "    or.b32 {sum}, {sum}, {tmp};");
        let _ = writeln!(self.text, "    mov.u32 {tmp}, %tid.z;");
        let _ = writeln!(self.text, "    or.b32 {sum}, {sum}, {tmp};");
        let _ = writeln!(self.text, "    setp.ne.u32 {not_leader}, {sum}, 0;");
        let _ = writeln!(self.text, "    @{not_leader} bra {skip};");
        // gridSize = nctaid.x * nctaid.y * nctaid.z.
        let _ = writeln!(self.text, "    mov.u32 {grid}, %nctaid.x;");
        let _ = writeln!(self.text, "    mov.u32 {dim}, %nctaid.y;");
        let _ = writeln!(self.text, "    mul.lo.u32 {grid}, {grid}, {dim};");
        let _ = writeln!(self.text, "    mov.u32 {dim}, %nctaid.z;");
        let _ = writeln!(self.text, "    mul.lo.u32 {grid}, {grid}, {dim};");
        let _ = writeln!(self.text, "    mul.lo.u32 {target}, {grid}, {multiplier};");
        let _ = writeln!(self.text, "    mov.u64 {bar_addr}, _vyre_grid_barrier;");
        let _ = writeln!(self.text, "    atom.global.add.u32 {old}, [{bar_addr}], 1;");
        let _ = writeln!(self.text, "{spin}:");
        let _ = writeln!(self.text, "    ld.volatile.global.u32 {cur}, [{bar_addr}];");
        let _ = writeln!(self.text, "    setp.lt.u32 {waiting}, {cur}, {target};");
        let _ = writeln!(self.text, "    @{waiting} bra {spin};");
        let _ = writeln!(self.text, "{skip}:");
        // Release the CTA once the leader has seen every CTA arrive, then
        // acquire the other CTAs' freshly published global writes.
        let _ = writeln!(self.text, "    bar.sync 0;");
        let _ = writeln!(self.text, "    membar.gl;");
        Ok(())
    }
}
