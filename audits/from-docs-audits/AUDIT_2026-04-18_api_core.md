# API Surface Audit — vyre-core

**Date:** 2026-04-18  
**Scope:** `vyre-core/src/lib.rs` and every `pub` item reachable from the crate root.  
**Methodology:** Manual inspection of `pub mod` trees, `pub use` re-exports, trait method counts, function arity, `#[non_exhaustive]` coverage, and doc-comment completeness.  
**Findings:** 25

---

## Legend

| Severity | Meaning |
|----------|---------|
| **Critical** | Breaks downstream stability guarantees, leaks implementation detail, or expands the frozen API surface without intent. |
| **High** | Will cause breakage when the type/feature evolves; missing forward-compatibility guard. |
| **Medium** | Degrades API ergonomics, clutters public docs, or violates internal style rules (F8, no globs, >5 args). |

---

## Findings

### Accidentally pub — internal helpers leaked through `pub mod` trees

**API-01** — `vyre-core/src/ops.rs:26` — **Critical** — `pub use signatures::*;` is a glob re-export that leaks every symbol from the internal `signatures` module into `vyre::ops`. Any rename, deletion, or addition in `signatures` becomes a breaking change for downstream callers.  
**Fix:** Replace with an explicit `pub use signatures::{A, B, C};` listing only the stable operation signatures.

**API-02** — `vyre-core/src/ir/serial/text.rs:297` — **Critical** — `pub fn encode_text_body` is an internal hex-envelope helper for the canonical text format. It is not a stable serialization entry point (`Program::to_wire` / `Program::from_text` are).  
**Fix:** Change to `pub(crate)` or inline into `to_text`.

**API-03** — `vyre-core/src/ir/serial/text.rs:319` — **Critical** — `pub fn push_usize` is a string-append micro-helper used only by `encode_text_body`.  
**Fix:** Change to `pub(crate)`.

**API-04** — `vyre-core/src/ir/serial/text.rs:340` — **Critical** — `pub fn push_hex_byte` is a string-append micro-helper used only by `encode_text_body`.  
**Fix:** Change to `pub(crate)`.

**API-05** — `vyre-core/src/ir/serial/text.rs:349` — **Critical** — `pub fn parse_wire_bytes_line` is an internal text-format parser helper.  
**Fix:** Change to `pub(crate)`.

**API-06** — `vyre-core/src/ir/serial/text.rs:365` — **Critical** — `pub fn hex_nibble` is an internal nibble parser duplicated in `ops/security_detection/detector_support/decode.rs`.  
**Fix:** Change to `pub(crate)` and deduplicate with the security-detection helper.

**API-07** — `vyre-core/src/ir/serial/text.rs:377` — **Critical** — `pub fn truncate` is an internal string-truncation utility.  
**Fix:** Change to `pub(crate)`.

**API-08** — `vyre-core/src/ir/engine/prefix.rs:44` — **High** — `pub fn brace_depth_prefix` is a host-side CPU helper explicitly documented as *"NOT part of the vyre IR"*, yet it is reachable as `vyre::ir::engine::prefix::brace_depth_prefix`.  
**Fix:** Change the entire `prefix` module to `pub(crate)` or move to a private `internal` path.

**API-09** — `vyre-core/src/ir/engine/prefix.rs:84` — **High** — `pub fn nested_depth_prefix` — same rationale as API-08.  
**Fix:** Change to `pub(crate)`.

**API-10** — `vyre-core/src/ir/engine/prefix.rs:125` — **High** — `pub fn newline_prefix_sum` — same rationale as API-08.  
**Fix:** Change to `pub(crate)`.

**API-11** — `vyre-core/src/ir/engine/token_match_filter.rs:27` — **High** — `pub fn filter_matches_by_token` is a host-side convenience utility documented as *"NOT part of the vyre IR"*, reachable as `vyre::ir::engine::token_match_filter::filter_matches_by_token`.  
**Fix:** Change to `pub(crate)`.

**API-12** — `vyre-core/src/ir/engine/token_match_filter.rs:55` — **High** — `pub fn filter_code_matches` — same rationale as API-11.  
**Fix:** Change to `pub(crate)`.

**API-13** — `vyre-core/src/ir/transform/inline.rs:58` — **High** — `pub fn default_resolver` is internal inliner plumbing that resolves op IDs through the private registry. Exposing it lets downstream depend on the registry lookup contract.  
**Fix:** Change to `pub(crate)`.

**API-14** — `vyre-core/src/ir/transform/inline.rs:77` — **High** — `pub fn input_arg_map` is an internal helper that maps callee buffers to caller expressions.  
**Fix:** Change to `pub(crate)`.

**API-15** — `vyre-core/src/ir/transform/inline.rs:90` — **High** — `pub fn input_buffers` is an internal helper that filters buffers by access mode.  
**Fix:** Change to `pub(crate)`.

**API-16** — `vyre-core/src/ir/serial/wire/framing.rs:54` — **Medium** — `pub mod impl_reader;` exposes the low-level byte-reader implementation module for the wire decoder. Callers should use `Program::from_wire`, not raw reader methods.  
**Fix:** Change to `pub(crate) mod impl_reader;`.

**API-17** — `vyre-core/src/ir/serial/wire/encode.rs:164` — **Medium** — `pub mod put_expr;` exposes the expression tag-and-payload encoder module. The stable entry point is `Program::to_wire`.  
**Fix:** Change to `pub(crate) mod put_expr;`.

**API-18** — `vyre-core/src/ir/serial/wire/encode.rs:173` — **Medium** — `pub mod put_node;` — same rationale as API-17.  
**Fix:** Change to `pub(crate) mod put_node;`.

### Missing `#[non_exhaustive]` — growable public types

**API-19** — `vyre-core/src/ir/model/expr.rs:113` — **High** — `pub enum Expr` is the core IR expression enum. New expression variants are added regularly (e.g., recent `Opaque` extension). Without `#[non_exhaustive]`, downstream exhaustive `match` arms break on every addition.  
**Fix:** Add `#[non_exhaustive]`.

**API-20** — `vyre-core/src/ir/model/node.rs:24` — **High** — `pub enum Node` is the core IR statement enum. New node kinds (e.g., `Speculate`, `AsyncLoad`) land frequently.  
**Fix:** Add `#[non_exhaustive]`.

**API-21** — `vyre-core/src/ir/model/node_kind.rs:29` — **High** — `pub enum Value` is the interpreter value type. New scalar types (`F16`, `BF16`, `Tensor`, etc.) expand it.  
**Fix:** Add `#[non_exhaustive]`.

**API-22** — `vyre-core/src/ir/model/program.rs:19` — **High** — `pub enum MemoryKind` classifies buffer memory (storage, uniform, workgroup). New GPU memory classes (e.g., shared scratch, tile memory) are planned.  
**Fix:** Add `#[non_exhaustive]`.

**API-23** — `vyre-core/src/ir/model/program.rs:36` — **High** — `pub enum CacheLocality` hints buffer cache behavior. New hints (e.g., streaming, persistent) will be added.  
**Fix:** Add `#[non_exhaustive]`.

**API-24** — `vyre-core/src/ir/memory_model.rs:5` — **High** — `pub enum MemoryOrdering` defines IR-level memory ordering. New GPU memory models (e.g., `ReleaseAcquire`) are on the roadmap.  
**Fix:** Add `#[non_exhaustive]`.

**API-25** — `vyre-core/src/ops/metadata.rs:28` — **High** — `pub enum Backend` lists target backends. New backends (e.g., `Metal`, `Cuda`) will be added.  
**Fix:** Add `#[non_exhaustive]`.

**API-26** — `vyre-core/src/ops/metadata.rs:68` — **High** — `pub enum Category` classifies operations. New categories are introduced as the op library expands.  
**Fix:** Add `#[non_exhaustive]`.

**API-27** — `vyre-core/src/ops/metadata.rs:149` — **High** — `pub enum Compose` defines composition strategies. New strategies (e.g., `StreamCompose`) are planned.  
**Fix:** Add `#[non_exhaustive]`.

**API-28** — `vyre-core/src/ir/serial/text.rs:84` — **High** — `pub enum TextParseError` carries parse-failure variants. New text-format validation checks will add variants.  
**Fix:** Add `#[non_exhaustive]`.

**API-29** — `vyre-core/src/cert/fingerprint.rs:12` — **High** — `pub struct ProbeObservation` will grow new probe fields (e.g., `warp_size`, `fast_math`).  
**Fix:** Add `#[non_exhaustive]`.

**API-30** — `vyre-core/src/cert/fingerprint.rs:47` — **High** — `pub struct BackendFingerprint` may gain new digest formats or versioning fields.  
**Fix:** Add `#[non_exhaustive]`.

**API-31** — `vyre-core/src/optimizer.rs:21` — **High** — `pub struct PassMetadata` will grow new scheduling fields (e.g., `cost_model`, `parallel_ok`).  
**Fix:** Add `#[non_exhaustive]`.

**API-32** — `vyre-core/src/optimizer.rs:92` — **High** — `pub enum OptimizerError` will grow new scheduling failure modes.  
**Fix:** Add `#[non_exhaustive]`.

### Stub / useless doc comments on `pub` items

**API-33** — `vyre-core/src/ir/serial/text.rs:297` — **Medium** — `encode_text_body` doc comment is the literal stub `"encode_text_body function."`. It explains nothing about semantics, errors, or invariants.  
**Fix:** Write a real doc comment describing the hex-envelope format, OOM bounds, and return value.

**API-34** — `vyre-core/src/ir/serial/text.rs:319` — **Medium** — `push_usize` doc comment is `"push_usize function."`.  
**Fix:** Write a real doc comment or make `pub(crate)` and delete the stub.

**API-35** — `vyre-core/src/ir/serial/text.rs:340` — **Medium** — `push_hex_byte` doc comment is `"push_hex_byte function."`.  
**Fix:** Write a real doc comment or make `pub(crate)`.

**API-36** — `vyre-core/src/ir/serial/text.rs:349` — **Medium** — `parse_wire_bytes_line` doc comment is `"parse_wire_bytes_line function."`.  
**Fix:** Write a real doc comment or make `pub(crate)`.

**API-37** — `vyre-core/src/ir/serial/text.rs:365` — **Medium** — `hex_nibble` doc comment is `"hex_nibble function."`.  
**Fix:** Write a real doc comment or make `pub(crate)`.

**API-38** — `vyre-core/src/ir/serial/text.rs:377` — **Medium** — `truncate` doc comment is `"truncate function."`.  
**Fix:** Write a real doc comment or make `pub(crate)`.

### `pub fn` with >5 args — unscoped functions

**API-39** — `vyre-core/src/ir/transform/compiler/recursive_descent.rs:52` — **Medium** — `pub fn parse` takes 6 positional arguments (`tokens`, `transitions`, `start_state`, `accept_state`, `max_stack`, `max_callbacks`). This is an unscoped parameter list that should be collapsed into a `ParseConfig` struct.  
**Fix:** Introduce `ParseConfig { tokens, transitions, start_state, accept_state, max_stack, max_callbacks }` and take `config: &ParseConfig`.

**API-40** — `vyre-core/src/ops/compression/deflate_core.rs:179` — **Medium** — `pub fn repeat_value` takes 6 positional arguments (`reader`, `out`, `base`, `bits`, `limit`, `value`).  
**Fix:** Scope into a `RepeatConfig` struct or make `pub(crate)`.

**API-41** — `vyre-core/src/ops/hash/reference/blake3.rs:158` — **Medium** — `pub fn g` takes 7 positional arguments (`v`, `a`, `b`, `c`, `d`, `x`, `y`).  
**Fix:** Scope the index quartet into a `GIndices` struct or make `pub(crate)` (this is an internal compression round helper).

**API-42** — `vyre-core/src/ops/hash/reference/blake2.rs:151` — **Medium** — `pub fn g32` takes 7 positional arguments.  
**Fix:** Same as API-41 — scope into a struct or make `pub(crate)`.

**API-43** — `vyre-core/src/ops/hash/reference/blake2.rs:163` — **Medium** — `pub fn g64` takes 7 positional arguments.  
**Fix:** Same as API-41 — scope into a struct or make `pub(crate)`.

### God traits — `pub trait` with >7 methods (violates F8 split)

**API-44** — `vyre-core/src/ir/visit.rs:65` — **Critical** — `pub trait ExprVisitor` declares 19 methods (one per `Expr` variant). This is a god trait: any backend, interpreter, or optimizer that only cares about a subset of expressions must still see the entire surface area. Adding a new `Expr` variant forces a breaking change on every implementor.  
**Fix:** Split into focused sub-traits (`LiteralVisitor`, `ArithmeticVisitor`, `MemoryVisitor`, etc.) and provide a blanket adapter so existing implementors continue to compile.

**API-45** — `vyre-core/src/ir/visit.rs:328` — **Critical** — `pub trait NodeVisitor` declares 13 methods (one per `Node` variant). Same god-trait problem as API-44.  
**Fix:** Split into `ControlFlowVisitor`, `MemoryVisitor`, `AsyncVisitor`, etc., with a blanket adapter.

### Traits marked `pub` that should be `pub(crate)` — internal-only plumbing

**API-46** — `vyre-core/src/ops/cooperative.rs:28` and `vyre-core/src/ops/primitive/subgroup_scan.rs:48`, `subgroup_shuffle.rs:64`, `subgroup_ballot.rs:56`, `subgroup_broadcast.rs:24`, `subgroup_reduce.rs:24` — **High** — The `pub trait WgslOp` is duplicated identically across 6 public submodules. It is an internal WGSL-lowering contract, not a stable frontend-facing API. Exposing it lets downstream implement the trait and pretend to be a WGSL primitive, which has no supported use case.  
**Fix:** Define a single `pub(crate) trait WgslOp` in `lower::wgsl` and remove all 6 public duplicates.

---

## Summary by Category

| Category | Count |
|----------|-------|
| Accidentally pub / leak implementation detail | 18 |
| Missing `#[non_exhaustive]` | 14 |
| Stub / useless doc comments | 6 |
| `pub fn` with >5 args | 5 |
| God trait (>7 methods) | 2 |
| Trait that should be `pub(crate)` | 1 (6 sites) |
| **Total distinct findings** | **46** |

## Recommended Priority Order

1. **Critical:** Remove the `pub use signatures::*;` glob (API-01). This is the highest-leakage item.
2. **Critical:** Split `ExprVisitor` and `NodeVisitor` (API-44, API-45) or seal them behind a private super-trait to prevent downstream implementations.
3. **High:** Add `#[non_exhaustive]` to `Expr`, `Node`, `Value`, `MemoryKind`, `CacheLocality`, `MemoryOrdering`, `TextParseError`, and the optimizer/cert/routing structs (API-19 through API-32).
4. **High:** Reduce the 6 duplicate `WgslOp` traits to a single `pub(crate)` definition (API-46).
5. **High:** Make all `ir::engine::prefix::*` and `ir::engine::token_match_filter::*` helpers `pub(crate)` (API-08 through API-12).
6. **Medium:** Delete or replace the 6 stub doc comments in `ir/serial/text.rs` (API-33 through API-38).
7. **Medium:** Scope the >5-arg public functions into config structs (API-39 through API-43).
