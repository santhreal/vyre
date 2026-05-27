# VYRE_SCHED_TRANSFORM_DEEPER  -  Deep audit

Summary

This audit inspects the optimizer scheduler and transform/DCE/CSE rewrite code paths. Major risks: pass-scheduler dirty/available invariants can cause unsatisfied-requirement errors or missed re-runs; Region nodes are treated as opaque by many transforms (DCE/CSE/rewrite), producing correctness gaps unless region-inline runs first; Cow/borrow/owned conversions are mostly correct but contain hotspots and a few semantic pitfalls that can cause incorrect rewrites or unnecessary allocations.

Quality score: 5/10  -  many correctness-sensitive areas, some already patched but several latent issues requiring fixes.

Closure status  -  2026-04-29 scoped scheduler/transform pass

| Finding | Status | Source / proof |
|---|---|---|
| 1 / 2 scheduler intra-iteration invalidation can make later passes unschedulable | fixed | `vyre-foundation/src/optimizer/scheduler.rs` keeps current-iteration availability stable and queues invalidated passes in `next_dirty`; `scheduler_tests::invalidating_prior_requirement_does_not_break_current_iteration` passed. |
| 5 / 6 / 7 DCE treats `Node::Region` as opaque | fixed | `eliminate_dead_lets.rs` and `eliminate_unreachable.rs` now recurse into Region bodies; `dce_descends_into_region_bodies` and `dce_region_live_ins_propagate_to_outer_scope` passed. |
| 8 CSE treats `Node::Region` as opaque | stale/fixed before this pass | `transform/optimize/cse/impl_csectx.rs` descends into Region bodies under a scoped CSE context and clears observed state at the boundary. |
| 10 rewrite treats `Node::Region` as opaque | stale/fixed before this pass | `optimizer/rewrite.rs` rewrites Region bodies and preserves borrowed identity when unchanged. |

Findings (severity | file:line | description | suggested fix)

1. CRITICAL | vyre-foundation/src/optimizer/scheduler.rs:254-268 | run_once may return UnsatisfiedRequirement when an earlier pass in the same iteration invalidates a prerequisite that was already marked available. Removing a prerequisite from `available` makes later pending passes appear blocked and causes an error. | Change scheduling to tolerate intra-iteration invalidation: either (A) re-enqueue invalidated passes immediately (insert into `pending` before continuing), or (B) don't remove already-processed names from `available` while still able to schedule dependent passes in the same iteration; prefer (A)  -  re-run invalidated prerequisites within the same iteration before continuing dependent passes.

2. HIGH | vyre-foundation/src/optimizer/scheduler.rs:280-296 | `next_dirty`/`available.remove()` logic tries to address SKIPed earlier passes but doesn't cover the case where an invalidated pass was already processed earlier and removing it causes dependent passes to fail. | Rework invalidation policy: when a pass invalidates `X` and `X` is not yet re-run this iteration, ensure `X` is added to `pending` (or otherwise scheduled) so dependencies don't fail; add unit tests covering "invalidate already-available prerequisite".

3. HIGH | vyre-foundation/src/optimizer/scheduler.rs:215-223 & 238-236 | Dirty-tracking invariants are subtle: scheduler only calls `analyze()` for passes present in `dirty`. A pass may need to run because an earlier pass in the same iteration mutated the program but failed to mark it dirty (race in invalidation semantics). | Tighten semantics: document and enforce that `dirty` represents *all* passes that might change effect of analyze; consider calling `analyze()` for light-weight passes even if not dirty, or make invalidation eagerly mark dependent passes dirty at the moment of change.

4. HIGH | vyre-foundation/src/optimizer.rs:246-263 | `registered_passes()` appends builtin passes first then externals in registration order. If callers expect region-inline/CSE/DCE ordering, it's not enforced here  -  transforms that depend on region_inline being run first can be incorrect. | Either document and assert the required ordering in `registered_pass_registrations()` or add an explicit builtin `region_inline` pass (or scheduling metadata) so required transforms run in the optimizer pipeline in stable order.

5. CRITICAL | vyre-foundation/src/transform/optimize/dce/eliminate_dead_lets.rs:101-109 | `Node::Region` is preserved by copying/cloning the body instead of recursing; DCE does not run inside Regions. This leads to dead-lets and unreachable statements surviving when callers forget to run region-inlining first. | Either (A) have DCE recurse into Region bodies (honoring Region semantics), or (B) assert (with a debug-mode invariant) that region_inline ran earlier. Prefer (A) or make region-inline part of the canonical optimizer pipeline so CSE/DCE always sees inlined bodies.

6. HIGH | vyre-foundation/src/transform/optimize/dce/eliminate_unreachable.rs:82-90 | Same Region opacity in unreachable-elim: `If/Loop/Block` descend but Region is treated as opaque clone. Unreachable code in a Region won't be removed. | Same fix as #5: propagate elimination into Region bodies or canonicalize Regions before optimize.

7. HIGH | vyre-foundation/src/transform/optimize/dce/dce.rs:9-13 | dce() calls eliminate_dead_lets -> eliminate_unreachable -> eliminate_dead_lets on the program entry, but because Regions are opaque earlier, ordering matters and callers can accidentally skip elimination inside Regions. | Make region handling explicit in dce(): either call region-inline at start or document that caller must run region-inline; add an assertion or a mode to recurse into Region bodies.

8. MEDIUM | vyre-foundation/src/transform/optimize/cse/impl_csectx.rs:193-201 | `Node::Region` is cloned and returned verbatim by CSE. CSE cannot deduplicate or clear observed state across Region boundaries. | Either inline small Regions before CSE (see region_inline.rs) or implement CSE traversal that optionally descends into Region bodies when safe.

9. MEDIUM | vyre-foundation/src/transform/optimize/cse/impl_csectx.rs:64-87 | `Node::Let` calls `self.expr(value).into_owned()` immediately (forced ownership) before checking `expr_has_effect`. This eagerly clones/allocates even when no rewrite is needed. | Use Cow consistently: only materialize into Owned when you will mutate/record it. Convert to `let maybe_value = self.expr(value); if matches!(maybe_value, Cow::Borrowed(_)) { ... } else { let value = maybe_value.into_owned(); ... }` to avoid unnecessary clones.

10. MEDIUM | vyre-foundation/src/optimizer/rewrite.rs:128-131 & 218-224 | `rewrite_node_cow` and `rewrite_expr` treat `Node::Region` and `Expr::Opaque` as Borrowed (opaque). Combined with (5)/(6) this hides transform opportunities. | Provide an explicit API for callers to specify whether Regions should be traversed. Add `rewrite_with_regions(program, true)` or similar.

11. MEDIUM | vyre-foundation/src/transform/optimize/cse/impl_csectx.rs:290-308 | When CSE substitutes an expression with a canonical `Expr::var(existing)` it returns Cow::Owned(var). Replacing expressions with variable references is only safe if the canonical variable is provably immutable in the region; current invalidation/clear_observed_state may not fully guarantee soundness around async ops or extension nodes. | Strengthen invariants: ensure `record_insert` keys are only inserted when the current observed state ensures immutability; add unit tests for loops, async loads/stores and extensions to validate no incorrect aliasing occurs. Consider using a strict SSA-like validation before substituting across assignments.

12. MEDIUM | vyre-foundation/src/transform/optimize/dce/collect_expr_refs.rs:9-11 | `collect_expr_refs` interns variable names into a HashSet<String> (allocates). This is a hotspot during large rewrites. | Use `HashSet<Arc<str>>` or `HashSet<Ident>` (Ident already exists) to avoid repeated allocations and improve performance; or provide an API that accepts an allocator or borrow set.

13. MEDIUM | vyre-foundation/src/transform/optimize/dce/expr_has_effect.rs:42-46 | `Expr::Opaque(extension)` calls `extension.cse_safe()` and treats non-cse-safe as effectful. That is conservative, but if extension authors mis-report `cse_safe` it changes optimization soundness silently. | Add unit tests and a debug-time validator for `Opaque` extensions that validates `cse_safe` vs `stable_fingerprint()` behavior; document extension contract requiring correctness.

14. LOW | vyre-foundation/src/transform/optimize/cse/impl_csectx.rs:369-385 | `rewrite_args` eagerly clones earlier unchanged args into a new Vec once any following arg is owned. This is correct but causes extra clones in hot paths. | Micro-optim: preallocate `Vec<Expr>` with exact capacity and clone only once, or use Cow<[Expr]> trick already used elsewhere; consider an arena-backed variant to avoid repeated heap allocations.

15. LOW | vyre-foundation/src/optimizer/rewrite.rs:30-46 | `rewrite_program` returns (Program,false) for borrowed entry and (program.with_rewritten_entry(entry), true) for owned entry. Good; however callers sometimes call `PassResult::from_programs` elsewhere  -  inconsistent habits can reintroduce clones. | Add guidance in comments and tests: call `PassResult::unchanged(program)` when unchanged to avoid cloning.

16. MEDIUM | vyre-foundation/src/ir_inner/model/expr.rs:408-449 | `Expr::call(op_id, args)` stores args Vec<Expr>. The transform helpers push Vec clones during rewrite  -  this is a major allocation hotspot. | Adopt Cow<[Expr]> or arena-backed arguments consistently across transforms (audit already shows rewrite.rs and cse use some Cow patterns; unify into single `Cow`-first design across transforms).

17. MEDIUM | vyre-foundation/src/ir_inner/model/arena.rs:46-57 | `ExprArena::alloc` stores raw pointers; API notes mention single-writer semantics, but there are many transform paths that clone-owned Expr trees and could inadvertently mix arena-allocated and owned Exprs. | Document and test arena invariants; consider an explicit type `ArenaExprRef` vs `Expr` to prevent accidental mixing that could lead to use-after-reset bugs.

18. MEDIUM | vyre-foundation/src/transform/optimize/dce/eliminate_dead_lets.rs:11-19 | The loop uses `.into_iter().take(reachable_len).rev()` which clones nodes into `kept` by pushing Node::let_bind with Node::let_bind(&name, value) that reconstructs node. For some Node variants this reconstructs and clones unnecessarily. | Use `mem::take`/`swap_remove` where possible and reuse `Node` instances when safe; or use Cow for Node rewriting to avoid allocations when program unchanged.

19. LOW | vyre-foundation/src/transform/optimize/dce/reachable_prefix.rs:4-9 | `reachable_prefix` only recognizes `Node::Return` as terminator. New control-flow constructs (Trap/Resume/IndirectDispatch) may also change reachability semantics. | Ensure all terminating control constructs are covered or document why only `Return` is considered.

20. LOW | vyre-foundation/src/optimizer.rs:345-350 | `requirements_satisfied` accepts `PassMetadata` (by value) not by ref  -  minor perf. | Change signature to take `&PassMetadata` to avoid copying small arrays repeatedly.

21. MEDIUM | vyre-foundation/src/transform/inline/region_inline.rs:100-116 | `inline_nodes_into` uses `Arc::try_unwrap(body)` fall-back to clone. Good attempt to avoid copies; however `std::sync::Arc::try_unwrap` is rare in hotspot and fallback clones may be expensive. | Keep but add metrics/logging on fallback frequency; consider an API to request exclusive ownership earlier at construction time when inlining is desired.

22. MEDIUM | vyre-foundation/src/transform/optimize/cse/expr_key.rs:21-40 | ExprKey variants are comprehensive but use `Arc<str>` and `SmallVec`  -  verify stable_fingerprint() injection for Opaque variants is collision-free. | Add fuzz tests comparing `Expr::Opaque` stable_fingerprint vs `ExprKey::Opaque` hash to assert injectivity; document expectations for extension authors.


Verdict

REJECT (needs fixes): multiple correctness risks found. The two highest-severity issues are: (a) scheduler intra-iteration invalidation causing UnsatisfiedRequirement errors and possible scheduler-level failures (see #1,#2), and (b) Region opacity across many transform passes leaving dead code / unreachable code unoptimized unless region-inline is guaranteed to run first (see #5,#6,#10). Both are correctness-impacting and must be fixed before release.

Suggested immediate action list

- Add unit tests that reproduce scheduler UnsatisfiedRequirement by invalidating an earlier processed prerequisite within the same iteration and assert scheduler either reorders or re-enqueues instead of erroring.
- Make Region inline a canonical initial optimizer step or make transforms descend into Region bodies; add regression tests for DCE/CSE inside Regions.
- Add sanitizer/debug assertions in debug builds that fail-fast if `registered_passes()` ordering would break transform preconditions.
- Micro-optimizations: adopt Cow/arena patterns consistently for Expr/Node rewrites; reduce String allocations in live sets.

I can implement a conservative fix for the scheduler (re-enqueue invalidated prerequisites within the same iteration) and make DCE descend into Regions (or add an assert requiring region-inline). Specify preference and I will produce the patch with tests and run cargo test + cargo clippy -- -D warnings.
