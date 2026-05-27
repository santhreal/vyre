# Doc-Truth Audit  -  2026-04-19

Every prose claim in user-facing docs checked against filesystem +
source reality. Each row: CLAIM, REALITY, VERDICT, LOCATION.

**Verdict key:** `TRUE` / `FALSE` / `STALE` (was true, no longer is) /
`VAGUE` (unfalsifiable) / `ASPIRATIONAL` (explicitly labeled future).

## Source-of-truth counters (2026-04-19)

| Counter | Value |
|---|---|
| Unique op ids in registry (distinct `dialect.op_name`) | 87 |
| .rs files in vyre-core/src/ | 1073 |
| Published crates | 8 (vyre-core, vyre-spec, vyre-macros, vyre-primitives, vyre-reference, vyre-wgpu, backends/photonic, backends/spirv) |
| Conform crates | 4 (vyre-conform-{spec,generate,enforce,runner}) |
| Total test files (all crates) | 299 |
| Bench files | 3 (registration_overhead.rs, primitives_showcase.rs, vs_cpu_baseline.rs) |
| Wire format version | 1 (not 2  -  despite commit messages claiming v2) |
| Workspace compile errors | 3–9 (fluctuating during in-flight migration) |
| check_release_signoff.sh gates passing | 12/16 |
| check_base_monument.sh prereqs passing | 5/9 |

## README.md audit

| Claim | Reality | Verdict |
|---|---|---|
| "one of the first GPU-first IRs" | Subjective, can't verify | VAGUE |
| "zero-overhead abstraction on GPU" | Registration overhead bench shows 1.95 ns warm lookup  -  ✓ for that one path. Dispatch hot path still has O(N) buffer pool scan + per-call validation. | STALE  -  bench proves one path; thesis-level claim overshoots |
| "machine-verified semantic contract" | Conform runner exists but `.internals/certs/` is not populated; certificates are not yet signed. | FALSE (today) |
| "three substrates" / "three backends" | 4 registered (wgpu, spirv, photonic, reference). Claim is a lower bound, so technically true but outdated. | STALE |
| IEEE-754 reference to sin/cos | References "declared ULP tolerance"  -  accurate for current state | TRUE |

## docs/ARCHITECTURE.md audit

| Claim | Reality | Verdict |
|---|---|---|
| "DialectRegistry is process-wide OnceLock<FrozenIndex>" | Verified at vyre-core/src/dialect/registry.rs | TRUE |
| "Every `Opaque` variant is additive-only" | Wire tag 0x80 encoded; encoder + decoder handle DataType/BinOp/UnOp Opaque. Node::Opaque tag not yet wired. | PARTIAL |
| "Four CI laws (A/B/C/D/H) green on every commit" | Today: A red (closed IR enums remain), B red (string WGSL remains), C red (capability negotiation), H varies. | FALSE |
| "vyre-core/src/dialect/* per-dialect" | 24 dialects found, matches. | TRUE |
| "Every backend declares supported_ops()" | `check_capability_negotiation.sh` fails in current run. | FALSE |

## docs/THESIS.md audit

| Claim | Reality | Verdict |
|---|---|---|
| "GPU kernels are provable" | Symbolic proof path doesn't exist; witness-set check is the best we do. | ASPIRATIONAL  -  should be flagged as such |
| "Every op has a signed certificate" | `.internals/certs/` empty. conform runner has ed25519 scaffolding only. | FALSE |
| "Reference interpreter is the oracle" | `vyre-reference` exists; 13 files of reference code STILL live in `vyre-core/src/ops/*/reference/`  -  so the oracle is split. | PARTIAL |
| "Three-substrate byte-identical parity" | `examples/three_substrate_parity/` ships xor-1M only, not full primitive corpus. | PARTIAL |

## docs/VISION.md audit

Not yet rigorously graded; claims are framed as future-state ("we will
ship"), so most are ASPIRATIONAL by construction. Audit scheduled for
v0.6.0 cycle  -  at that point every "we will" becomes a graded claim.

## RELEASE.md audit

Not audited against reality  -  it IS the spec. Claims live there as
targets, not facts. Only sections §1–§12 are mid-execution.

## Known lies that would auto-correct with small edits

1. **README op count**  -  no explicit number in README today; if one
   lands ("150+ ops"), immediately wrong: filesystem has 87.
2. **"four CI laws green"**  -  rewrite to "four CI laws enforced; current
   pass rate 12/16 gates, remaining 4 in §-migration."
3. **"Every op has a signed certificate"**  -  rewrite to "certificate
   scaffolding in place; signing + population land with §10."
4. **"Reference interpreter is the oracle"**  -  rewrite to note that 13
   reference files remain in vyre-core/src/ops pending §9 migration.

## Actions (for whoever updates docs)

- Every claim rewrite must include a citation to `.internals/audits/`
  or a `scripts/check_*.sh` that enforces it.
- A claim without a matching gate is a lie waiting to happen.
- Before v1.0: every `FALSE` + `PARTIAL` row above must be either
  (a) fixed in code, or (b) reworded to ASPIRATIONAL with explicit
  "future" framing.

## Meta

This audit is written once. It will rot the moment the next commit
lands. Rerun by:
1. Regenerating source-of-truth counters (first section).
2. Walking each row and re-verifying. The `VERDICT` column is the
   change signal.
