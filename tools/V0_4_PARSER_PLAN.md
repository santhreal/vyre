# v0.4 vyre-frontend-c Parser Plan — Agent Handoff

**Status anchor (2026-05-04):** First divergence sweep on the 52-file
internal corpus produced `match=0 divergent=4 vyre_failed=45 clang_failed=3`.
After categorising vyre stderr: **43/45 vyre_failed are missing-include
errors (environmental, not parser bugs)**, 1 is a real buffer-size bug,
and the 4 "divergent" results reflect an abstraction-level mismatch
between vyre's typed-VAST and clang's semantic AST — *not* a real parser
disagreement. Tree-sitter is the right oracle going forward; clang
requires headers and rejects almost everything.

This plan is the work to take that signal to release.

---

## Headline (do not move)

vyre-frontend-c v0.4 ships when **all four** are true:

1. `vyrec --parse-only` parses every `.c` file in *at least three* Linux
   subsystems (target list: `kernel/`, `mm/`, `fs/ext4/`) with **zero**
   parser-stage crashes. "Parse-only" means: no preprocessor, no codegen,
   no link — just produce a canonical AST.
2. Per-subsystem divergence vs. tree-sitter (the `tree-sitter-c` grammar)
   is **≤ 1% of files**, and every divergent file has either (a) a
   regression test pinning the disagreement and a documented rationale
   for which side is correct, or (b) an open ticket with reproducer.
3. The canonical AST type lives in its own crate (`vyre-c-ast`). Both
   the GPU pipeline and a CPU fallback (if added) lower into the same
   shape. The crate has no GPU/driver dependencies.
4. `vyre-lints` (the lego-block lint) reports zero violations across
   `vyre-frontend-c`, `vyre-c-ast`, `vyre-c-preproc` (new). Crate
   boundaries are enforced.

If 1–4 hold, ship. If not, headline is red.

---

## Substrate splits the agent must land

Today `vyre-frontend-c` does too much: lex, parse, preprocess, codegen,
link, test fixtures. v0.4 separates this into:

- `vyre-c-ast`  — **new crate**. Owns the canonical AST type
  (`CAst`, `CAstNode`, `CAstKind`). Pure data, `serde`-serialisable, no
  GPU deps. Conversion fn: `from_typed_vast(typed_vast: &[u32]) -> CAst`.
  This is the shape every downstream consumer (semantic, codegen, lints)
  reads.
- `vyre-c-preproc` — **new crate**. Owns include resolution, macro
  expansion, conditional compilation. Optional dependency of
  `vyre-frontend-c`; absent for `--parse-only` mode.
- `vyre-frontend-c` — **slimmed**. Lex + parse only. Produces `CAst`.
  Calls `vyre-c-preproc` only if the caller asks for it.
- `vyrec` (binary in `tools/vyrec`) — **slimmed**. Adds `--parse-only`
  flag. In parse-only mode, runs frontend-c without preproc, dumps the
  CAst as JSON to stdout (or file via `-o`), exits.

The lego-block lint already exists (`vyre-lints`); make sure new crates
get added to its allowlist correctly. Reverse dependencies from
`vyre-c-ast` → `vyre-frontend-c` are forbidden and the lint must catch
them.

---

## Phase 1 — Parser is preprocessor-free (blocker for everything else)

**Outcome:** `vyrec --parse-only foo.c` succeeds on any syntactically
valid C source, even when its `#include`s are unresolvable.

**Tasks (in order):**

1. **Audit `vyre-frontend-c/src/pipeline.rs` for include-resolution
   sites.** The error string today is `vyre-frontend-c: system #include
   <X> not found in -I search path`. Find every `?` propagation that
   surfaces this.
2. **Make include resolution opt-in.** Default behaviour for the parser
   should be: emit the `#include` directive as a `CAstKind::IncludeDirective`
   token (path-string preserved) and continue. *Do not* attempt to
   resolve, *do not* fail. The current "find header in -I path" code
   moves to `vyre-c-preproc`.
3. **Add `--parse-only` flag to `vyrec`.**
   - Skip `link_c11_executable` entirely.
   - Skip codegen.
   - Run only: lex → vast build → annotate typedef → classify →
     expr-shape → cfg/gotos → ast-shunting → CAst conversion.
   - Output: serialised `CAst` JSON to `-o` path (default stdout).
4. **Smoke gate:** the divergence-gate sweep on the existing 52-file
   corpus must drop `vyre_failed` from 45 → ≤ 2. The remaining ≤ 2 are
   the real bugs (e.g. the `'kinds' expected 4 bytes` issue). Open
   tickets for those, do not block Phase 1 on them.

**Verification:**
- New unit tests in `vyre-frontend-c/tests/parse_only/`. At minimum:
  - `unresolvable_system_include.c` — `#include <does_not_exist.h>` then
    `int main(void){return 0;}`. Test asserts: parse succeeds, the
    resulting CAst contains exactly one `IncludeDirective` node with
    path `"does_not_exist.h"` and one `FunctionDefinition` node.
  - `relative_include.c` — `#include "neighbor.h"`. Same assertions
    with system=false.
  - `nested_include_in_macro.c` — adversarial: `#define INC <foo.h>`
    followed by `#include INC`. Document the chosen behaviour
    explicitly (recommended: emit as `IncludeDirective` with
    `path: TokenSequence` instead of `path: String`; do not expand
    the macro in parse-only mode).

---

## Phase 2 — Canonical AST crate (`vyre-c-ast`)

**Outcome:** A typed Rust enum that is the contract for "this is what
vyre's C parser produces."

**Tasks:**

1. **Define `CAstKind`.** Mirrors the C11 grammar's structural categories,
   not the GPU-storage kind ids. Concrete enum variants required (NOT
   exhaustive — extend as the conversion code requires):
   - `TranslationUnit { items: Vec<NodeId> }`
   - `IncludeDirective { system: bool, path: IncludePath }`
   - `FunctionDefinition { specifiers, declarator, body }`
   - `Declaration { specifiers, declarators }`
   - `CompoundStatement { items }`
   - `ExpressionStatement { expr }`
   - `IfStatement { cond, then_branch, else_branch }`
   - `ForStatement { init, cond, step, body }`
   - `WhileStatement { cond, body }`
   - `ReturnStatement { value }`
   - `BinaryExpr { op, lhs, rhs }`
   - `UnaryExpr { op, operand }`
   - `CallExpr { callee, args }`
   - `IdentifierRef { name }`
   - `IntLiteral`, `FloatLiteral`, `StringLiteral`, `CharLiteral`
   - `TypedefName { name }` (post-typedef-disambiguation)
   - `Attribute { name, args }` (GNU `__attribute__`)
   - `Typeof { operand }`
   - `Asm { ... }` (extended asm)
   - `GenericSelection { ... }` (`_Generic`)
   - `Unknown { vyre_kind: u32, range: ByteRange }` — escape hatch for
     anything the conversion can't classify yet. Logged as a warning,
     not an error. Counted by the gate.
2. **Define `CAst`.**
   - `nodes: Vec<CAstNode>` indexed by `NodeId` (newtype `u32`).
   - `root: NodeId`.
   - `source: Arc<str>` (or `Arc<[u8]>`) so byte ranges resolve.
   - `serde::{Serialize, Deserialize}` derive.
3. **Implement `from_typed_vast`.** Takes the post-shunting typed VAST
   blob and reshapes it into `CAst`. This is the actual algorithmic work:
   walking the parent/sibling links and producing a tree. Bound the
   stack via an explicit `Vec` (no recursion).
4. **Round-trip test.** `parse(src).serialise().deserialise()` must
   equal the original.

**Verification:**
- Each variant in `CAstKind` has at least one positive fixture in
  `vyre-c-ast/tests/fixtures/` and an assertion that
  `from_typed_vast(parse(fixture)).contains_kind(THE_KIND)`.
- `Unknown` count on the existing 52-file corpus is reported. v0.4 ships
  when `Unknown` count is ≤ 5% of total nodes across the corpus.

---

## Phase 3 — Tree-sitter oracle in the divergence gate

**Outcome:** The `divergence-gate.py` script compares vyre's `CAst`
against tree-sitter-c's CST and reports per-file divergences at a
matching abstraction level.

**Tasks:**

1. **Add tree-sitter to the gate.** Use the Python `tree_sitter` and
   `tree_sitter_c` packages.
   - `pip install tree-sitter tree-sitter-c` (record in
     `tools/requirements.txt`).
   - Replace `run_clang` with `run_tree_sitter`.
2. **Define a kind-mapping table** in
   `tools/divergence_kind_map.json`. Format:
   ```json
   {
     "tree_sitter_to_vyre": {
       "function_definition": "FunctionDefinition",
       "if_statement": "IfStatement",
       "binary_expression": "BinaryExpr",
       ...
     }
   }
   ```
   The agent populates this incrementally — start with the 20 most
   common tree-sitter kinds, add as divergences surface.
3. **Per-node comparison.** For each file:
   - Walk both trees in DFS order.
   - At each node, compare `(mapped_kind, child_count)`.
   - First disagreement = divergence; record byte range and both kinds.
4. **Tighten the gate.** Replace the coarse `Shape.differs_from` with
   the per-node walk. Keep the histogram as a sanity check.

**Verification:**
- Re-sweep the 52-file corpus. Expected outcome: `match` count ≥ 40
  after Phases 1+2+3. Remaining divergences are the actionable work.

---

## Phase 4 — Linux subsystem corpus + per-subsystem completeness reports

**Outcome:** A reproducible benchmark of vyre-frontend-c against
tree-sitter on real Linux subsystems, surfaced as a versioned scoreboard.

**Tasks:**

1. **Vendor a Linux source snapshot.** Pin a specific tag (e.g.
   `v6.10`). Layout under
   `vyre-frontend-c/tests/linux_subsystems/<subsys>/`. Three
   targets for v0.4: `kernel/`, `mm/`, `fs/ext4/`. Do NOT vendor the
   whole kernel — only these directories' `.c` and `.h` files.
2. **Per-subsystem sweep target.** Add to
   `tools/divergence-gate.py` a `--subsystem` flag that scopes the sweep
   to one directory and emits per-subsystem JSON + a markdown report
   (`tools/reports/<subsys>.md`).
3. **Markdown report template:**
   ```
   # Subsystem: kernel/
   - Files scanned: N
   - Match: N (P%)
   - Divergent: M (Q%)
   - Vyre-failed: K (with breakdown by error class)
   - Tree-sitter-failed: L (almost always 0; flag if not)

   ## Top 10 divergence patterns
   1. (kind, kind) = (FunctionDefinition, declaration) — N occurrences
   ...

   ## Open tickets
   - [link to issue / fixture name]
   ```
4. **CI hook.** A new GitHub workflow `parser-completeness.yml`:
   - Runs the sweep on each subsystem.
   - Fails if `match%` regresses by more than 0.5 percentage points
     vs. the committed baseline at `tools/baselines/<subsys>.json`.

**Verification:**
- Reports committed under `tools/reports/`.
- Headline metric per subsystem: `match%`. v0.4 ships when all three
  target subsystems hit ≥ 99%.

---

## Phase 5 — Hardening, regressions, docs

**Outcome:** The substrate is durable and a future maintainer can
extend it without breaking the contract.

**Tasks:**

1. **Adversarial fixtures.** For every `CAstKind` variant, write at
   least one adversarial fixture: macro-injected, comment-injected,
   unicode-identifier, GNU-extension, etc. Goal: 3 adversarials per
   kind. Each must produce the same `CAstKind` as the positive twin
   (or, if differing behaviour is intentional, a documented different
   kind plus a comment explaining why).
2. **CRATE.md per crate.** `vyre-c-ast/CRATE.md` and
   `vyre-c-preproc/CRATE.md` document the public API contract,
   stability guarantees, and the lego-block boundary rules.
3. **Public API audit.** Run `cargo public-api` against the new crates;
   commit the snapshot. Future PRs must update the snapshot when changing
   public API.
4. **Doc the parse-only mode in vyrec's `--help`.** One paragraph in
   `tools/vyrec/README.md` explaining the parse-only contract and what
   it does NOT do.

---

## Hard rules for the agent (do not violate)

1. **No GPU dependencies in `vyre-c-ast`.** Drag in `vyre-foundation`
   only if absolutely necessary, and document why.
2. **No fixture-shaped fixes.** If a divergence is fixed by hardcoding
   a token byte sequence in a kernel, the divergence-gate's hardcode
   lint will reject the patch. Real fixes extend the classification
   table or the kind-mapping JSON.
3. **No `#[ignore]` on tests** to silence failures. If a test cannot
   pass with the current engine, the engine has a real bug and the test
   stays failing until the engine is fixed.
4. **No deletion of code that compiles.** "Unused" import or "dead"
   function inside the new crates needs to be audited as a migration
   signal first; only delete after explicit author confirmation.
5. **Adversarial test required for every fix.** Every PR that fixes a
   parser bug ships a positive twin AND a negative/adversarial twin in
   the same PR. The gate enforces this.
6. **Commit messages must include the metric delta.** Format:
   `phase4: kernel/ match% 87.3 → 91.2 (+3.9)` — keeps the scoreboard
   present in `git log`.

---

## What the human (peer / CC / user) does, not the agent

- Approves the canonical `CAstKind` enum surface (Phase 2 task 1).
  Agent proposes; human ratifies before downstream work commits.
- Approves the kind-mapping table additions when they're non-obvious
  (e.g. is `attributed_declarator` in tree-sitter the same as vyre's
  `Declaration` with attributes attached, or a separate kind?).
- Picks the Linux kernel tag to vendor (Phase 4 task 1).
- Decides scope expansion: if Phase 4's three target subsystems hit
  99% easily, do we add `drivers/usb/core/`?

---

## Success state at end of v0.4

```
$ vyrec --parse-only kernel/sched/core.c | jq '.nodes | length'
12847

$ tools/divergence-gate.py sweep \
    --corpus vyre-frontend-c/tests/linux_subsystems/kernel \
    --vyrec target/release/vyrec \
    --oracle tree-sitter \
    --out reports/kernel.json \
    --subsystem kernel
sweep: 1432 files | match=1431 divergent=1 vyre_failed=0 tree_sitter_failed=0

$ cat tools/reports/kernel.md
# Subsystem: kernel/
- Files scanned: 1432
- Match: 1431 (99.93%)
- Divergent: 1 (0.07%)
- Vyre-failed: 0
- Tree-sitter-failed: 0
...
```

If the agent gets here on `kernel/`, `mm/`, and `fs/ext4/`, v0.4 ships.
