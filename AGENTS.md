## VYRE Agent Rules

`AGENTS.md` is the only authoritative agent instruction file for this workspace.
`CLAUDE.md` and `GEMINI.md` are compatibility redirects and must not contain
separate policy. If an agent-specific convention is still relevant, add it here
in neutral form so every tool receives the same architecture, quality, and
validation contract.

- Do not use subagents/workers for this repository while the release stabilization policy is active.
- Do not use native Codex workers for implementation fanout, test generation, test expansion, or audits.
- Do not use `codex-agents` workers for this repository; that bridge is inactive for this release.
- Work stays in the main agent so architecture decisions, implementation, validation, and commits remain directly accountable.
- New tests belong in crate `tests/` directories unless an existing inline test must be updated to match a changed contract.
- Assume a GPU exists. Probe failures are configuration failures and must be reported loudly, not silently skipped.
- No stubs, no evasion: no `todo!()`, `unimplemented!()`, `panic!("not implemented")`, empty implementations, fake default returns, or comments that document a limitation instead of fixing the implementation.
- No deletion as evasion: a broken import, dead-code warning, or orphaned module is a migration signal. Re-wire or re-implement unless the user explicitly approves deletion or the owning subsystem is proven obsolete.
- Tests assert contracts, not shape. Every behavior change needs a proving test that would fail on the previous broken behavior and an adversarial case for the relevant boundary.
- No `Co-Authored-By` lines, AI attribution, emoji, or celebration/status filler in commits or repository-authored docs.
- Run validation with bounded build parallelism: prefer `CARGO_BUILD_JOBS=1 cargo test ...` for focused cargo validation unless a repository wrapper is explicitly required by the task.
- Concrete backend details live only in their concrete driver crates. Shared crates (`vyre-foundation`, `vyre-driver`, `vyre-runtime`, `vyre-core`, `vyre-primitives`, `vyre-libs`, `vyre-intrinsics`, and conform harness crates) must use neutral terms such as primary text, primary binary, secondary text, native module, backend, target, device, and artifact. Do not introduce hardcoded concrete driver names, shader dialect names, backend-specific error strings, or compatibility reexports outside the owning driver crate.
- The canonical optimization control plane is `docs/optimization/README.md`. Before assigning or doing optimization work, use its ownership map, patch contract, op matrix, and benchmark targets. Older performance plans and audits are evidence only unless that directory delegates to them.
- Performance work has a strict two-layer boundary:
  - Layer 1 is IR-level math-pure optimization. Rewrites such as Granlund-Montgomery constant division, Lemire-style constant remainder, exact-division simplification, FMA synthesis, and shift-add decomposition transform `Expr`/`Node` to equivalent IR and live in `vyre-foundation/src/optimizer/passes/` so every backend inherits them before lowering.
  - Layer 2 is backend lowering strategy. Target-dependent choices such as dual-issue FP32/INT32 scheduling, tensor-core batching, native multiply-high selection, or 16-bit decomposition do not change program semantics and live only inside the owning concrete backend lowering strategy modules. Shared crates may define neutral strategy traits and capability records, but concrete chip/API names and emission details stay in the backend crate.
