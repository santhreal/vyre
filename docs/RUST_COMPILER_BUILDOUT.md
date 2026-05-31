# Rust compiler build-out + weir upgrade plan (authoritative)

Single roadmap for the GPU-first Rust frontend in the vyre tree. Companion to
`parsing-and-frontends.md` (VAST design) and `PARSING_EXECUTION_PLAN.md` (the C
path). The Rust path mirrors the C separation of concerns.

## 0. Mission and headline

- **Next release headline:** a working Rust **borrow checker** on the
  nano-subset, byte-for-byte and accept/reject-identical to `rustc` on a defined
  corpus, built by composing **weir** dataflow analyses.
  **Status (2026-05-30):** the borrow checker is implemented and rustc-gated via
  a dedicated `crate::borrowck` NLL engine (see P2). The "composing weir" form of
  the mission is not yet met — tracked as an open refactor, not a behaviour gap.
- **Trajectory:** grow incrementally into a whole Rust compiler. Each grammar
  widening ships with its own `rustc` differential gate. No widening without a
  gate.
- **Standing constraint:** every surface meets the Santh testing contract
  (Section 4). The contract is the spine of this plan, not a trailer.

## 1. Current reality (audited 2026-05-29; updated 2026-05-30)

| Piece | Path | State |
|-------|------|-------|
| Lexer | `vyre-libs::parsing::rust::lex` | CPU lexer real (cooked, keyword-aware), fail-closed on oversized tokens (>u16 len / >u32 offset); GPU lexer plan exists, dispatch **not** wired. |
| Parser | `vyre-libs::parsing::rust::parse` | Recursive-descent nano-subset: `fn`/`let`-`mut`/assignment/compound-assignment (`+=`/`-=`)/`return`/`while`/`for name in start..end`/`if`-`else`(incl. `else if`)/blocks/calls(fwd refs)/`&`/`&mut`/deref/`!`/unary-minus + full binop precedence (`+ - * / % == != < > <= >= && ||`). Depth-guarded; u128 literal-overflow gate. |
| Sema | `vyre-libs::parsing::rust::sema` | **Implemented + gated.** `resolve` (scopes/shadowing/calls), `typeck` (i32/bool/refs, E0308/E0061/E0614/E0384/E0425, `&mut`->`&` coercion), `borrow_check` (mutability E0596 + escape E0597 + conflicts E0499/E0502 via a CFG + NLL loan-liveness engine in `crate::borrowck`). |
| Lower | `vyre-libs::parsing::rust::lower` | **Implemented + gated.** AST -> `vyre::ir::Program`: param buffers, alpha-renamed locals, return/store, if-else terminal lowering, straight-line call inlining, counted-`while` -> `Node::Loop`, and `for start..end` -> `Node::Loop` with signed i32 source semantics over u32 IR bounds. |
| Driver | `vyre-frontend-rust` | Thin orchestrator: api, pipeline, object (stub), tests (14). |
| Oracle | `vyre-frontend-rust/tests` + `vyre-libs` rust-parser oracles | Byte-level lexer differential vs `rustc_lexer`; `rustc_differential` (accept/reject vs live rustc); `rust_lower_exec_oracle` (lower+run vs AST interp + rustc); `rust_sema_borrow_oracle` + `rustc_nll_facts`; `adversarial_parse_depth`. All passing. |
| Dataflow engine | `libs/dataflow/weir` | 56.9k LOC, 1420 tests. Has `live`, `must_init`, `def_use`, `dominators`, `may_alias`, `points_to`, `escape`, `range`, `fixed_point_*`, `ifds`, each with CPU oracle + GPU path. IFDS exploded path is single-lane on GPU (perf P0). |

## 2. Invariants (this plan must not violate)

1. **Boundary:** algorithms (lex/parse/sema/lower) live in
   `vyre-libs::parsing::rust`; the driver only orchestrates + emits artifacts.
   Mirrors `parsing::c`. Substrate never depends on the driver.
2. **Tier rule:** lowering to `Program` is Tier-3 substrate. Borrow analysis
   composes weir; it does not reimplement dataflow.
3. **No fakes:** an unimplemented stage returns a loud `Err` with `Fix:`, never a
   passthrough, empty `Program`, `todo!()`, or doc-only limitation.
4. **rustc is the oracle:** correctness for every Rust surface is a differential
   against `rustc` / `rustc_lexer` / `syn` on a committed corpus. Accept exactly
   what rustc accepts; reject exactly what it rejects.
5. **weir parity:** every weir analysis is bit-identical CPU-oracle vs GPU on its
   conformance matrix; a missing intrinsic errors, never silently falls to CPU.
6. **No silent GPU fallback** anywhere (lexer, dataflow, dispatch).

## 3. Workstreams (parallel tracks)

- **W0 Foundation/hygiene** (fast, here): honest scorecard from the project
  gates; zero the ~78 facade warnings (vyre-primitives 49 + vyre-driver-cuda 29);
  clear the 145 orphan `__law7_split` dirs; decide build C-compiler policy
  (distcc vs `CC=gcc`).
- **W1 Frontend correctness** (CPU, here): real `resolve` -> `typeck` -> `lower`
  for the nano-subset, each `rustc`/reference gated.
- **W2 Borrow checker** (headline; CPU here, GPU proof on desktop): AST -> CFG,
  compose weir, borrow/ownership lattice, diagnostics, `borrow_oracle` vs rustc.
- **W3 weir upgrades + testing:** fix IFDS single-lane GPU (PERF-003/4/5); bring
  the analyses the borrow checker needs to full contract depth.
- **W4 Testing contract completion** (cross-cutting spine; Section 4).
- **W5 Org / LAW7:** split 392 files >500 LOC; dedup; orphan-dir cleanup.
- **W6 Perf:** GPU perf P0s (PERF-001..008, desktop-verified); IR-level Layer-1
  optimizations.

## 4. The testing contract, applied (the part that is "not complete enough")

Every surface below needs the full contract: **positive truth + negative twin +
adversarial + proptest (10k+) + differential + scale + e2e through the real
binary**, with file/line/value assertions, never `assert!(is_ok)`.

### Frontend, per surface
| Surface | Required tests |
|---------|----------------|
| Lexer | rustc_lexer byte-differential (have) + exhaustive 1/2-byte token cases + proptest over valid+hostile bytes + span/UTF-8 invariants + golden corpus + fuzz target. |
| Parser | `syn`/rustc AST-shape differential on a corpus + every production positive/negative/adversarial + error-span assertions + golden ASTs + proptest + fuzz. |
| resolve | rustc differential (same name-resolution accept/reject) + shadowing/scope adversarial + proptest. |
| typeck | rustc differential (same type accept/reject) + mismatch/negative cases + proptest. |
| borrow check | **rustc borrow differential** on a corpus: use-after-move, double `&mut`, `&`+`&mut`, return-dangling, reassign-while-borrowed + proptest-generated programs + adversarial. |
| lower | run lowered IR through `vyre-reference`, assert semantics vs an expected oracle + differential reference-vs-backend. |

### Whole-path (every user-visible surface)
- `parse_rust_bytes` and `compile_unit` x default + >=3 flag combos: snapshot output + error + exit/result.
- Each output/evidence format x >=1 realistic input, byte-compared to a fixture.
- Every error path produced from a real input that triggers it (loud-fail tests, already started in `smoke.rs`).

### Module-to-module integration pairs (real A -> real B -> assert B)
`lex -> parse`, `parse -> resolve`, `resolve -> typeck`, `typeck -> borrow`,
`borrow -> lower`, `lower -> vyre-reference`. Each: happy + malformed-A + empty-A.

### Dogfood + release gate
- A corpus of real nano-subset Rust files (sliced from real crates) run end to
  end; `tests/dogfood` scenario with expected diagnostics + exit + duration band.

### weir, per analysis (live, must_init, def_use, dominators, may_alias, points_to, escape, ifds, range)
- CPU-oracle vs GPU **bit-identical** on the conformance matrix.
- proptest over random CFGs/graphs (cycles, unreachable, self-loops, diamonds).
- differential vs a textbook reference (e.g. dominators vs Lengauer-Tarjan).
- scale test (30M-edge class) + criterion perf + adversarial graphs.
- resident/megakernel/batch paths each exercised through the real device path.

## 5. Phases (ordered, done-when)

### P0 - Foundation and honest baseline
- [ ] Run project gates on the committed tree; record real lint-shape / tripwire / op-inventory numbers. Verify: `cargo xtask lint-shape-tests`, the `scripts/check_*.sh` set.
- [ ] Zero the ~78 facade warnings. Verify: `cargo check -p vyre-primitives -p vyre-driver-cuda` warning-free.
- [ ] Clear 145 orphan `__law7_split` dirs; simplify the 3 scanners that special-case them. Verify: `find ... -name __law7_split | wc -l` == 0; gates still green.
- [ ] Build-C-compiler policy decided (distcc installed here, or `CC=gcc` pinned in a documented dev override).

### P1 - Frontend semantics for the nano-subset  (DONE — verified 2026-05-30)
- [x] `sema::resolve`: real scope graph + name resolution; rustc differential gate.
- [x] `sema::typeck`: real type environment for `i32`/`bool`/refs; rustc differential gate.
- [x] `lower`: nano-subset AST -> `Program`; reference-oracle semantic gate.
- Done-when: `compile_unit` with semantics enabled succeeds on the corpus and matches rustc accept/reject; no honest-Err left on the wired path.
- Verify: `cargo test -p vyre-frontend-rust --test rustc_differential` and `cargo test -p vyre-libs --features rust-parser --test rust_lower_exec_oracle` (lower+run vs AST interp + live rustc).

### P2 - Borrow checker (release headline)  (IMPLEMENTED + gated — see note)
- [x] `sema::cfg`: nano-subset AST -> CFG (basic blocks, edges, def/use sites).
- [x] Borrow/ownership conflict + escape + mutability checks (E0499/E0502/E0597/E0596) with NLL loan-liveness.
- [x] `tests/rust_sema_borrow_oracle.rs` + `rustc_nll_facts.rs`: rustc differential on the borrow corpus.
- **Note (coherence):** the shipped conflict checker uses a dedicated CFG + NLL loan-liveness engine in `crate::borrowck`, NOT (yet) a composition of weir `live`/`must_init`/`def_use` as this plan's W2/P2 originally specified. The accept/reject behaviour is gated against rustc; the weir-composition refactor (to satisfy invariant 2's "composes weir; does not reimplement dataflow") remains open. Track as a P3 follow-up, not a correctness gap.
- Done-when (original): borrow checker accepts/rejects exactly as rustc on the corpus — MET; weir-composition invariant — OPEN.

### P3 - weir depth + perf
- [ ] Fix IFDS single-lane GPU grid + serial scans (PERF-003/4/5); parity preserved.
- [ ] Bring `live`/`must_init`/`def_use`/`dominators` to full contract depth (Section 4 weir block).
- Done-when: weir analyses used by the borrow checker meet contract; IFDS GPU is competitive (desktop-verified).

### P4 - Grammar widening (incremental toward whole compiler)
- [ ] One construct per task (loop/break, struct, enum, generics, traits, ...),
      each: parser + sema + lower + rustc differential gate, in the same task.
- **Landed so far (each rustc-differential + exec-oracle gated):** `while`
  (counted-loop lowering), half-open range `for i in a..b`, unary minus (`-x`),
  compound assignment (`+=`/`-=`).
- **Verified OUT_OF_SUBSET against the current IR** (do not attempt without new
  IR primitives): `loop`/`break` (no infinite-loop/break node in `Node`),
  `i64`/`u64` arithmetic (rejected by the IR validator's cross-backend contract).
  `else if` is already supported. Next tractable: `u32` (primitives exist; needs
  literal-type inference), or one structured data construct with the same
  parse -> sema -> lower path.

### P5 (parallel/ongoing) - Org + GPU perf
- [ ] LAW7 split of 392 files >500 LOC (worst-first: `release_workloads.rs` 2594).
- [ ] GPU perf P0s PERF-001/002/006/007/008 (desktop-verified).

## 6. Parallelism and execution policy (DECIDED 2026-05-29: single agent)

The work is highly parallel, but the policy is **single agent, no subagents or
workers**, consistent with `vyre/AGENTS.md` (implementation stays accountable in
the main agent). Parallelism is realized only by: many small verified commits,
read-only kimi audit fan-out where useful, and the fleet (multiple machines on
the NFS share working different crates). No implementation fanout, no
test-generation workers.

## 7. Risks

- GPU runtime proof for weir/borrow perf needs the RTX 5090 desktop (santhserver
  has the 3080 Ti but no CUDA toolkit); CPU correctness is fully verifiable here.
- Borrow-checker scope creep: hold the nano-subset line; widen only via P4 tasks.
- "Green gate" theater: the prior legendary gate ran on uncommitted WIP with a
  static assertion estimate. This plan requires executed tests, not estimates.
