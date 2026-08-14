//! Per-kernel PTX emission state.
//!
//! Owns [`BodyCtx`]: the accumulating output text, the per-class register and
//! label counters, the maps from descriptor operand id to register, and the
//! flags that make a lowering refuse rather than emit something unsound. It
//! owns no emission: every method that writes an instruction is implemented
//! on `BodyCtx` in a sibling module named for the concept it emits.

use rustc_hash::{FxHashMap, FxHashSet};
use vyre_lower::descriptor::Name;
use vyre_lower::BindingSlot;

use crate::reg::Reg;
use crate::PtxEmitOptions;

pub(super) struct BodyCtx<'a> {
    pub(super) options: PtxEmitOptions,
    pub(super) text: String,
    /// Per-class next register index. Pre-decremented `<N>` register
    /// declarations are sized from these counters.
    pub(super) next_pred: u32,
    pub(super) next_b16: u32,
    pub(super) next_u32: u32,
    pub(super) next_i32: u32,
    pub(super) next_f32: u32,
    pub(super) next_u64: u32,
    /// Per-body next label index for branch targets.
    pub(super) next_label: u32,
    /// Map from descriptor operand id → PTX register holding its value.
    pub(super) operand_to_reg: FxHashMap<u32, Reg>,
    /// Descriptor result ids known to be literal u32 values. Used to
    /// fold constant buffer indices into immediate address offsets.
    pub(super) u32_literals: FxHashMap<u32, u32>,
    /// Map from binding slot → its loaded global pointer register.
    pub(super) slot_to_ptr: FxHashMap<u32, Reg>,
    /// Map from shared-memory binding slot → PTX shared symbol.
    pub(super) slot_to_shared_symbol: FxHashMap<u32, String>,
    /// Read-only global bindings with enough spatial reuse to route loads
    /// through CUDA's non-coherent/read-only cache path (`ld.global.nc`).
    pub(super) read_only_cache_slots: FxHashSet<u32>,
    /// Tags whose native cp.async groups have been committed but not yet
    /// waited. Keeping the wait at AsyncWait, instead of immediately after
    /// AsyncLoad, lets independent compute overlap with global-to-shared DMA.
    pub(super) pending_cp_async_tags: FxHashSet<Name>,
    /// Active structured-loop induction values keyed by loop variable.
    pub(super) loop_indices: FxHashMap<Name, Reg>,
    /// Per source-level loop-carrier name: the PTX register that
    /// carries the current value across iterations. Allocated by
    /// `LoopCarrierInit`, written by `LoopCarrierEnd`, read by
    /// `LoopCarrier`. Persists for the life of the kernel emission so
    /// post-loop reads pick up the loop's final value.
    pub(super) named_carriers: FxHashMap<Name, Reg>,
    /// Result-id of every `LoopCarrier` op, mapped back to its
    /// carrier name so the binder can return the carrier register
    /// directly to consumers.
    pub(super) named_carrier_result_ids: FxHashMap<u32, Name>,
    /// Per-slot cached register holding the buffer's element count.
    /// Preloaded from params metadata at entry so all branch arms see
    /// dominated length registers. Without this clamp PTX speculative
    /// loads from `Expr::select` arms can read past the buffer end
    /// (WGSL clamps automatically; PTX does not).
    pub(super) slot_to_length_reg: FxHashMap<u32, Reg>,
    /// Binding slot lookup table. Emission performs this lookup for
    /// every memory op; keep it O(1) instead of scanning the layout.
    pub(super) slot_to_binding: FxHashMap<u32, &'a BindingSlot>,
    /// True when the descriptor contains barriers or shared memory and
    /// every lane in the launched workgroup must remain live through the
    /// preamble. In this mode memory side effects use per-op bounds
    /// predicates instead of relying on an entry-wide element-count exit.
    pub(super) full_workgroup_entry: bool,
    /// Sequential index of the next `MemoryOrdering::GridSync` barrier
    /// emitted in this kernel. Each whole-grid barrier `i` waits on the
    /// module-scope counter reaching `(i+1) * gridSize` (see
    /// [`Self::emit_grid_sync_barrier`]); the indices must be assigned in
    /// emission order so every CTA agrees on each barrier's release target.
    pub(super) grid_barrier_index: u32,
    /// Nesting depth of enclosing `StructuredForLoop` bodies during emission.
    ///
    /// A `MemoryOrdering::GridSync` barrier is ONLY correct at a static position
    /// that executes at most once per launch, because its release target is
    /// computed at EMIT time from [`Self::grid_barrier_index`] and is therefore a
    /// compile-time constant multiple of `gridSize`. A loop emits its body once
    /// and branches back, so a barrier inside a loop would reuse one fixed
    /// target across every iteration: after the first iteration the monotonic
    /// counter is already at or past that target and the spin never waits, which
    /// silently degrades a whole-grid barrier into a no-op. Tracked so
    /// [`Self::emit_grid_sync_barrier`] can refuse instead.
    pub(super) grid_sync_loop_depth: u32,
    /// Result ids whose value is provably identical in EVERY invocation of the
    /// grid.
    ///
    /// Populated as ops are emitted, which is sound because the descriptor is
    /// SSA-ordered: an operand is always recorded before any consumer reads it.
    /// Absence means "not proven", never "proven varying", so every consumer
    /// must treat a missing id as non-uniform.
    ///
    /// This exists for [`Self::emit_return`]. A `Return` lowers to a branch to
    /// the single kernel exit, and a branch that only SOME invocations take is
    /// only safe if no synchronization follows: the invocations that left can
    /// never arrive at a later `bar.sync`, and the ones that stayed wait for
    /// them forever. Grid uniformity (not merely per-CTA uniformity) is the
    /// requirement, because a whole CTA leaving early strands the remaining
    /// CTAs at a cooperative grid barrier just as surely.
    pub(super) uniform_results: FxHashSet<u32>,
    /// Number of enclosing conditional bodies whose condition is NOT in
    /// [`Self::uniform_results`].
    ///
    /// Nonzero means control flow reached this point through a branch that some
    /// invocations may not have taken, so a `Return` here would be divergent.
    /// Tracked so [`Self::emit_return`] can refuse instead of emitting a branch
    /// that can hang the kernel.
    pub(super) nonuniform_cond_depth: u32,
}
