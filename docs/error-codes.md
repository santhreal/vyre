# vyre Error Codes

Applies to Vyre 0.7.2.

This document is the canonical registry of stable error and warning codes
surfaced through the foundation-owned diagnostic protocol. Codes are
append-only within a schema version. A new code must define its stage,
corrective action, and retry class before release.

Every message emitted through [`Diagnostic`](../vyre-foundation/src/lib.rs)
carries one of these codes. Tooling (LSP clients, CI annotators, editor
extensions) keys rules off the code, not the prose; prose is free to drift
across versions as long as the code stays stable.

## Code families

| Family | Severity | Source                                      |
|--------|----------|---------------------------------------------|
| `E-*`  | error    | General errors surfaced via `Diagnostic`    |
| `W-*`  | warning  | Deprecations, soft-failed invariants        |
| `V###` | error    | Program-validation rules (`validate_program`) |
| `B-*`  | error    | Backend dispatch / capability errors        |
| `C-*`  | error    | Conformance verdict failures                |

## Validation codes (`V###`)

Validation emits structured fields: code, validation phase, typed location,
cause, corrective action, and retry class. Human-readable rendering retains
the code and corrective action, but callers key behavior on the structured
fields rather than rendered prefixes.

All validation codes below use diagnostic stage `validate` and retry class
`never`.

| Code | Invariant | Corrective action |
|------|-----------|--------------|
| `V008` | Duplicate local binding (shadowing) | Choose a unique local name, or pass `ValidationOptions::with_shadowing(true)` to allow nested shadowing. |
| `V009` | Atomic on non-writable buffer | Declare the buffer with `BufferAccess::ReadWrite`. |
| `V010` | Barrier reached by only part of a workgroup (divergent barrier) | Move the barrier to uniform control flow. |
| `V011` | Assignment to loop variable | Loop variables are immutable, so rename instead. |
| `V012` | Unsupported cast between two DataTypes | Use a supported casts.md conversion or rewrite the expression before validation. |
| `V013` | Bytes load/store on buffer without `bytes_extraction = true` | Use a typed buffer (U32/I32/F32/…), or declare the buffer with `.with_bytes_extraction(true)` when the op is a bytes-extraction op like `decode.base64`. |
| `V014` | Atomic on buffer with non-u32 element type | Atomics only support U32 elements, so retype the buffer. |
| `V016` | Unknown op id in `Expr::Call` | Use a registered op id or add the op to core::ops::*. |
| `V018` | Program nesting depth exceeds `DEFAULT_MAX_NESTING_DEPTH` | Flatten nested If/Loop/Block structures or split the program before lowering. |
| `V019` | Program has more than `DEFAULT_MAX_NODE_COUNT` nodes | Split the program into smaller kernels or run an optimization pass before lowering. |
| `V020` | Call to non-inlinable op | Lower this op through its dedicated backend path or rewrite the caller with explicit IR. |
| `V021` | Call argument count mismatches callee's ReadOnly/Uniform input count | Pass exactly one argument per input buffer in binding order. |
| `V022` | Program or callee declares too many outputs | Mark at most one result buffer with `BufferDecl::output(...)`. |
| `V023` | Cast to `Bytes` is unsupported in WGSL lowering | Use buffer load/store directly for byte data. |
| `V025` | Atomic on workgroup buffer is outside the portable memory model | Use a storage ReadWrite buffer for atomics. |
| `V027` | Atomic index has wrong type (expected `u32`) | Cast the index to U32 before the atomic. |
| `V028` | Fma operand has wrong type (expected `f32`) | Cast the operand to F32 before Fma, or use the integer mul/add form explicitly. |
| `V029` | Select branches have mismatched types | Cast both branches to the same type before Select. |
| `V030` | Opaque Expr extension fails invariant (empty extension_kind/debug_identity/missing result_type/validate_extension failure) | Return a stable non-empty `extension_kind`, a human-readable `debug_identity`, and an explicit `result_type`, and pass `validate_extension`. |
| `V031` | Opaque Node extension fails invariant (empty extension_kind/debug_identity/validate_extension failure) | Return a stable non-empty `extension_kind`, a human-readable `debug_identity`, and pass `validate_extension`. |
| `V032` | Duplicate sibling `let` binding in the same region | Rename one binding, or move one declaration into an inner Block/Region/Loop if a new scope is intended. |
| `V033` | Expression nesting exceeds `DEFAULT_MAX_EXPR_DEPTH` | Split the expression into intermediate let-bindings before lowering. |
| `V034` | Backend does not support the requested cast target | Choose a target type the backend supports, or validate against a backend that advertises that cast support. |
| `V035` | Narrowing cast may truncate high bits | Use a non-narrowing target, or prove the source value fits before casting. |
| `V036` | Constant store index exceeds the declared buffer element count | Keep constant store indices inside the declared element range. |
| `V041` | Subgroup expressions used without backend subgroup support | Validate with `ValidationOptions::with_backend(backend)` where `backend.supports_subgroup_ops() == true`, or remove subgroup ops before lowering. |
| `V042` | Program validation error 042 | See diagnostic output. |
| `V043` | Program validation error 043 | See diagnostic output. |
| `V044` | Program validation error 044 | See diagnostic output. |
| `V045` | Program validation error 045 | See diagnostic output. |
| `V046` | Distributed collective node validation failure | Validate with backend collective support, use matching collective buffer element types, declare every referenced buffer, and keep collective buffers in device/global storage. |
| `V047` | Bitwise subgroup reduction given an f32 operand | Use an integer operand for `And`/`Or`/`Xor`, or use `Add`/`Mul`/`Min`/`Max` for a float reduction. |
| `V051` | Buffer reference used where a value is expected | A buffer reference is legal only as a call argument. Pass it directly to a composite op, or read an element with `Expr::Load`. |
| `V052` | Call passes a reference to an undeclared buffer | Declare the buffer in `Program::buffers`. |
| `V053` | Value passed for a `buffer<T>` parameter | Pass `Expr::buffer_ref(name)` naming the buffer the op should read. |
| `V054` | Referenced buffer's element type does not match the signature | Pass a buffer whose element type matches `buffer<T>`, or change the op signature. |
| `V055` | Synchronizing loop exit is unordered against the back edge | Put an unconditional barrier after the early exit, as the final node in the loop body. |
| `V056` | Backend does not support one operation used by the program | Choose a backend that supports the operation or register its implementation. |
| `V057` | atomic value type `…` does not match required `u32` | Ensure the atomic operand is U32. |
| `V058` | compare-exchange expected type `…` does not match required `u32` | Ensure Expr::Atomic.expected is U32. |
| `V059` | compare-exchange atomic is missing expected value | Set Expr::Atomic.expected for AtomicOp::CompareExchange. |
| `V060` | non-compare-exchange atomic includes an expected value | Use Expr::Atomic.expected only with AtomicOp::CompareExchange. |
| `V061` | atomic on unknown buffer `…` | Declare it in Program::buffers. |
| `V063` | store to non-writable buffer `…` | Declare it with BufferAccess::ReadWrite, BufferAccess::WriteOnly, or BufferAccess::Workgroup. |
| `V064` | store to unknown buffer `…` | Declare it in Program::buffers. |
| `V065` | load from unknown buffer `…` | Declare it in Program::buffers. |
| `V066` | Reference to an undeclared variable | Add a declaration before this use. |
| `V067` | buflen of unknown buffer `…` | Declare it in Program::buffers. |
| `V068` | invocation/workgroup ID axis … out of range | Use 0 (x), 1 (y), or 2 (z). |
| `V070` | Linear, affine, or relevant buffer use count violates its declared discipline | Add or delete buffer uses to satisfy the discipline, or select the intended `LinearType`. |
| `V083` | buffer `…` declared shape predicate `…` but has count=… | Change the count to satisfy the predicate, or relax the predicate. |
| `V084` | 64-bit integer arithmetic used where the shared IR supports only portable 32-bit arithmetic | Express the operation as a U32 pair with explicit carry/borrow, or use a backend-specific op whose schema declares native 64-bit arithmetic. |
| `V085` | Saturating arithmetic `…` received left=`…`, right=`…`; legal set is only U32 in the current lowering | Cast both operands to U32, or clamp explicitly for I32/F32. |
| `V086` | AbsDiff has left=`…`, right=`…` and can overflow (i32::MIN - i32::MAX invokes target-text signed-integer UB) | Cast operands to U32 before AbsDiff, or rewrite as an explicit branch. |
| `V087` | binary operation `…` … operand has type `…`, but numeric arithmetic expects one of `u32`, `i32`, or `f32` | Cast the operand to U32 or I32 before arithmetic, or rewrite to avoid mixing logical and arithmetic operators. |
| `V088` | binary operation `…` operands have mismatched numeric types: left=`…`, right=`…` (legal set: U32, I32, F32) | Cast one operand so both sides share a type (target-text has no implicit promotion). |
| `V089` | binary operation `Mod` … operand must be `u32` or `i32`, got `…`. Legal set for Mod is integer-only | Cast both operands to the same integer type before modulo. |
| `V090` | binary operation `Mod` operands have mismatched integer types: left=`…`, right=`…` | Cast one operand so both sides share the same integer type. |
| `V091` | binary operation `…` left operand has type `…`; legal integer set is `u32` or `i32` | Cast the left operand to U32 or I32. |
| `V092` | binary operation `…` right operand has type `…`; legal integer set is `u32` or `i32` | Cast the right operand to U32 or I32. |
| `V093` | Integer operation operands have mismatched types | Cast both operands to the same integer type. |
| `V094` | binary operation `…` … operand has type `…`; shift/rotate operands must be `u32` | Cast the operand to U32 before shifting/rotating. |
| `V095` | binary operation `…` … operand has type `…`; logical And/Or operands must be `u32` or `bool` | Cast the operand to U32 or Bool. |
| `V096` | binary comparison `…` operands have mismatched types: left=`…`, right=`…`. Comparisons require matching types | Cast both operands to the same type before comparing. |
| `V097` | Subgroup operation used without backend subgroup capability evidence | Validate with ValidationOptions::with_backend(backend) where `backend.supports_subgroup_ops() == true`, or remove the subgroup-dependent operation before lowering. |
| `V098` | Negation operand violates the portable total-arithmetic contract | Use `0 - x` for wrapping i32 negation, cast to U32 before Negate, or guard with Select(i32::MIN, 0, -x). |
| `V099` | unary operation `…` operand has type `…`, but legal set is U32, I32, or F32 | Cast or rewrite the operand to U32/I32/F32. |
| `V100` | unary operation `LogicalNot` operand has type `…`; legal set is `u32` or `bool` | Cast or rewrite the operand to produce U32 or Bool. |
| `V101` | unary operation `…` operand has type `…`; legal integer set is `u32`, `i32`, or `u64` | Cast or rewrite the operand to produce U32, I32, or U64. |
| `V102` | unary operation `…` operand has type `…`; legal set for math ops is `f32` | Cast or rewrite the operand to produce F32. |
| `V103` | unary operation `…` operand has type `…`; unpack ops require a 32-bit integer (`u32` or `i32`) word | Cast or rewrite the operand to produce U32 or I32. |
| `V104` | unary operation `…` is not recognized | Use a known UnOp variant from this enum (`Negate`, `LogicalNot`, `BitNot`, `Popcount`, `Clz`, `Ctz`, `ReverseBits`, `Sin`, `Cos`, `Exp`, `Log`, `Log2`, `Exp2`, `Tan`, `Acos`, `Asin`, `Atan`, `Tanh`, `Sinh`, `Cosh`, `Abs`, `Sqrt`, `InverseSqrt`, `Reciprocal`, `Floor`, `Ceil`, `Round`, `Trunc`, `Sign`, `IsNan`, `IsInf`, `IsFinite`, `Unpack4Low`, `Unpack4High`, `Unpack8Low`, `Unpack8High`). |
| `V105` | Program lacks one top-level Region | Construct runnable programs with Program::wrapped or add one top-level Region. |
| `V106` | workgroup_size[…] is 0 | All workgroup dimensions must be >= 1. |
| `V107` | duplicate buffer name `…` | Each buffer must have a unique name. |
| `V108` | duplicate binding slot … (buffer `…`) | Each buffer must have a unique binding. |
| `V109` | workgroup buffer `…` has count 0 | Declare a positive element count. |
| `V110` | output buffer `…` uses unsupported element type `…` | Output buffers must use fixed-width scalar or vector element types, not Array or Tensor. |
| `V111` | malformed validation frame stream: PopScope without matching PushScope | Rebuild the program through the structured IR builder before validation. |
| `V112` | unreachable statements after `return` | Remove statements after `return` or reorder them. |
| `V114` | malformed validation frame stream: loop variable `…` inserted outside any scope | Rebuild the program through the structured IR builder before validation. |
| `V115` | region `…` is marked non-composable with itself but appears multiple times in one fused program | Split the parser into separate dispatches, or give each instance distinct scratch storage before fusion. |
| `V116` | Fused nodes mix non-atomic reads and atomic access without an ordering barrier | Insert `Node::barrier()` between the read path and the atomic path, or rename the buffers before fusion. |
| `V118` | malformed validation frame stream: let binding `…` appeared outside any scope | Rebuild the program through the structured IR builder before validation. |
| `V119` | assignment to buffer `…` requires read-write storage but declared access is `…` | Use a read-write/output buffer or store into a mutable local binding. |
| `V120` | Assignment targets an undeclared variable | Add a declaration before this assignment. |
| `V121` | Store value type does not match the buffer element type | Cast the value to the buffer element type or use a compatible store type. |
| `V122` | Node::Store buffer `…` index has type `…` but must be `u32` | Cast the index to U32 before storing. |
| `V123` | Node::If condition has type `…` but must be `u32` or `bool` | Cast or rewrite the condition expression to produce `u32` or `bool`. |
| `V124` | Node::Loop from-bound has type `…`; legal loop bound type is `u32` | Cast the `from` bound to `u32`. |
| `V125` | Node::Loop to-bound has type `…`; legal loop bound type is `u32` | Cast the `to` bound to `u32`. |
| `V126` | indirect dispatch offset … is not 4-byte aligned | Use an offset aligned to a u32 dispatch count tuple. |
| `V127` | indirect dispatch references unknown buffer `…` | Declare the count buffer before validation. |
| `V128` | async stream tag is empty | Use a stable non-empty tag to pair AsyncLoad and AsyncWait nodes. |
| `V129` | malformed barrier visitor dispatch | Rebuild the program through the structured IR builder before validation. |
| `V130` | backend-allocated output buffer `…` has no static element count or output byte range | Declare the output with `.with_count(n)`, or use `.with_output_byte_range(0..0)` for a genuinely empty output. |

Codes `V024`, `V026`, `V037`-`V040`, `V048`-`V050`, `V062`, `V069`,
`V071`-`V082`, `V113`, `V117`, and codes above `V130` are reserved slots.
Allocate through this registry before emitting a new diagnostic.

## General errors (`E-*`)

| Code | Description | Fix template |
|------|-------------|--------------|
| `E-IR-001` | Decode of unknown Opaque extension id (wire format) | Link the crate that registers the extension id, then re-decode. |
| `E-IR-002` | Buffer zero-count with non-empty shape payload | Reject the non-canonical Program bytes. |
| `E-IR-003` | Diagnostic catalog carries a code not listed in `docs/error-codes.md` | Add the code to the registry before shipping. |

## Warnings (`W-*`)

| Code | Description | Fix template |
|------|-------------|--------------|
| `W-DEPREC-001` | Deprecated op id in use | Migrate to the replacement op listed in the deprecation registry. |

## Backend codes (`B-*`)

| Code | Description | Fix template |
|------|-------------|--------------|
| `B-CAP-001` | Backend does not support this op's capability class | Pick a backend that supports this op's capabilities, or use a different op. |
| `B-CAP-002` | Backend factory refused to construct (no GPU adapter, missing driver) | Fix the adapter issue per the error's `Fix:` prose, or skip this backend. |
| `B-CAP-003` | Unsupported feature, for example a dispatch request on an emission-only target | Use a backend whose `supports_dispatch` returns true. |

## Backend ErrorCode stable ids

| Variant | code | description |
|---------|------|-------------|
| `DeviceOutOfMemory` | 1001 | Backend device reported insufficient memory during allocation, staging, or dispatch. |
| `UnsupportedFeature` | 1002 | The selected backend lacks a feature required by the program or dispatch policy. |
| `PoisonedLock` | 1003 | A synchronization lock was poisoned after a panic while held. |
| `KernelCompileFailed` | 1004 | Kernel source compilation or validation failed for WGSL, SPIR-V, PTX, Metal IR, or another backend source format. |
| `DispatchFailed` | 1005 | Queue submission, command execution, readback, or dispatch completion failed. |
| `InvalidProgram` | 1006 | The submitted program violates backend constraints or the portable program contract. |
| `Unknown` | 1999 | Legacy or unclassified backend failure produced without a more specific machine-readable code. |

## Pipeline codes (`P-*`, `vyre-runtime::PipelineError`)

| Code | Variant | Description | Fix template |
|------|---------|-------------|--------------|
| `P-URING-001` | `IoUringSyscall { syscall, errno, fix }` | A raw `io_uring_setup` / `mmap` / `io_uring_enter` / `io_uring_register` syscall returned an errno. | Per-variant `fix:` string names the remediation; typical causes are kernel too old, missing CAP_SYS_ADMIN for SQPOLL on <5.13, or exhausted `max_map_count`. |
| `P-URING-002` | `QueueFull { queue, fix }` | The submission or completion queue rejected a request because it is full, out of bounds, or a slot is still in flight. | Drain completions with `AsyncUringStream::poll` or `UringCompletionPump::poll`, then retry. For backpressure-triggered rejections on `publish_slot`, wait for the kernel to advance `control[DONE_COUNT]`. |
| `P-URING-003` | `NotLinux` | `io_uring` or `futex_waitv` was requested on a non-Linux host. | Run on Linux 5.16+ or use artifact submission without an io_uring completion pump. |
| `P-URING-004` | `NvmePassthroughDisabled` | `submit_nvme_passthrough` was called without the `uring-cmd-nvme` feature. | Add `features = ["uring-cmd-nvme"]` to `vyre-runtime` in your `Cargo.toml`; requires Linux 6.0+. |
| `P-BACKEND-001` | `Backend(msg)` | A backend error bubbled up from `Megakernel::bootstrap` or `Megakernel::dispatch`. | Inspect the wrapped message; usually a validation error on the IR or an OOM during pipeline creation. |

## Conformance codes (`C-*`)

| Code | Description | Fix template |
|------|-------------|--------------|
| `C-LAW-001` | Backend output disagreed with reference on witnessed input | Fix the backend lowering or rewrite the op to honor the declared `AlgebraicLaw`. |
| `C-DET-001` | Backend produced non-deterministic output across seeds | Remove the non-deterministic code path; conform bans silent nondeterminism. |

## Adding a new code

1. Pick the next unused integer in the appropriate family.
2. Add a row to this document with the code, description, and `Fix:` template.
3. Emit the code via `Diagnostic` with the matching `code` field.
4. CI verifies every code emitted in source appears in this registry
   (see `scripts/check_error_codes_cataloged.sh`).

## Policy

- **Append-only.** Never reuse a retired code. Retiring a code leaves a
  row behind with a `Retired: v<X.Y.Z>` note.
- **Code is the stable key.** Prose may drift.
- **`Fix:` is mandatory.** Every variant carries actionable remediation.
- **No stringly-typed errors.** Every error-path surfaces a structured
  code; the prose is a formatting detail.


## Uncataloged Legacy / Auto-migrated Codes
| Code | Description | Fix template |
|------|-------------|--------------|
| `E-CSR` | Migrated code | See diagnostic output. |
| `E-DATAFLOW` | Migrated code | See diagnostic output. |
| `E-DECODE` | Migrated code | See diagnostic output. |
| `E-DECODE-CONFIG` | Migrated code | See diagnostic output. |
| `E-DECOMPRESS` | Migrated code | See diagnostic output. |
| `E-DFA` | Migrated code | See diagnostic output. |
| `E-GPU` | Migrated code | See diagnostic output. |
| `E-INLINE-ARG-COUNT` | Migrated code | See diagnostic output. |
| `E-INLINE-CYCLE` | Migrated code | See diagnostic output. |
| `E-INLINE-NON-INLINABLE` | Migrated code | See diagnostic output. |
| `E-INLINE-NO-OUTPUT` | Migrated code | See diagnostic output. |
| `E-INLINE-OUTPUT-COUNT` | Migrated code | See diagnostic output. |
| `E-INLINE-UNKNOWN-OP` | Migrated code | See diagnostic output. |
| `E-INTERP` | Migrated code | See diagnostic output. |
| `E-LOWERING` | Migrated code | See diagnostic output. |
| `E-PREFIX` | Migrated code | See diagnostic output. |
| `E-RULE-EVAL` | Migrated code | See diagnostic output. |
| `E-SERIALIZATION` | Migrated code | See diagnostic output. |
| `E-TEST` | Migrated code | See diagnostic output. |
| `E-TOML-PARSE` | Migrated code | See diagnostic output. |
| `E-UNKNOWN` | Migrated code | See diagnostic output. |
| `E-WIRE-UNKNOWN-DIALECT` | Migrated code | See diagnostic output. |
| `E-WIRE-UNKNOWN-OP` | Migrated code | See diagnostic output. |
| `E-WIRE-VALIDATION` | Migrated code | See diagnostic output. |
| `E-WIRE-VERSION` | Migrated code | See diagnostic output. |
| `E-X` | Migrated code | See diagnostic output. |
| `W-DEPRECATED` | Migrated code | See diagnostic output. |
| `W-OP-DEPRECATED` | Migrated code | See diagnostic output. |
| `W-TOML-BAD-OP-ID` | Migrated code | See diagnostic output. |
| `W-TOML-UNREADABLE` | Migrated code | See diagnostic output. |
| `E-TOML-BAD-CATEGORY` | Migrated code | See diagnostic output. |
| `E-TOML-BAD-OP-ID` | Migrated code | See diagnostic output. |
| `E-TOML-DIALECT-DIR-ENTRY` | Migrated code | See diagnostic output. |
| `E-TOML-DIALECT-DIR-MISSING` | Migrated code | See diagnostic output. |
| `E-TOML-DIALECT-DIR-UNREADABLE` | Migrated code | See diagnostic output. |
| `E-TOML-DUPLICATE-OP` | Migrated code | See diagnostic output. |
| `E-TOML-EMPTY-DIALECT` | Migrated code | See diagnostic output. |
| `E-TOML-EMPTY-DIALECT-PATH` | Migrated code | See diagnostic output. |
| `E-TOML-EMPTY-VERSION` | Migrated code | See diagnostic output. |
| `E-TOML-MANIFEST-REJECTED` | Migrated code | See diagnostic output. |
| `E-TOML-UNREADABLE` | Migrated code | See diagnostic output. |
