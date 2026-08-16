//! The validation-rule catalog.
//!
//! One owner for every stable `V###` rule: the phase allowed to emit it, the
//! invariant it enforces, and the correction it offers. Tooling reads the
//! correction from here rather than scraping a document, and
//! `docs/generated/error-codes.toml` is rendered from this table, so the
//! published catalog cannot describe a rule that no longer runs.

use super::validation_error::ValidationPhase;

/// One stable validation rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidationRule {
    /// Stable rule identity: `V` followed by three digits.
    pub code: &'static str,
    /// The sole validator phase allowed to emit this rule.
    pub phase: ValidationPhase,
    /// The invariant a rejected program violated.
    pub invariant: &'static str,
    /// The correction offered to the author of the rejected program.
    pub corrective_action: &'static str,
}

/// Every registered validation rule, ordered by code.
///
/// A rule must appear here before a `ValidationCode` naming it can
/// deserialize, so this table is what "registered" means.
pub(crate) const VALIDATION_RULES: &[ValidationRule] = &[
    ValidationRule {
        code: "V008",
        phase: ValidationPhase::Node,
        invariant: "Duplicate local binding (shadowing)",
        corrective_action: "Choose a unique local name, or pass `ValidationOptions::with_shadowing(true)` to allow nested shadowing.",
    },
    ValidationRule {
        code: "V009",
        phase: ValidationPhase::Memory,
        invariant: "Atomic on non-writable buffer",
        corrective_action: "Declare the buffer with `BufferAccess::ReadWrite`.",
    },
    ValidationRule {
        code: "V010",
        phase: ValidationPhase::Memory,
        invariant: "Barrier reached by only part of a workgroup (divergent barrier)",
        corrective_action: "Move the barrier to uniform control flow.",
    },
    ValidationRule {
        code: "V011",
        phase: ValidationPhase::Node,
        invariant: "Assignment to loop variable",
        corrective_action: "Loop variables are immutable, so rename instead.",
    },
    ValidationRule {
        code: "V012",
        phase: ValidationPhase::Expression,
        invariant: "Unsupported cast between two DataTypes",
        corrective_action: "Use a supported casts.md conversion or rewrite the expression before validation.",
    },
    ValidationRule {
        code: "V013",
        phase: ValidationPhase::Memory,
        invariant: "Bytes load/store on buffer without `bytes_extraction = true`",
        corrective_action: "Use a typed buffer (U32/I32/F32/…), or declare the buffer with `.with_bytes_extraction(true)` when the op is a bytes-extraction op like `decode.base64`.",
    },
    ValidationRule {
        code: "V014",
        phase: ValidationPhase::Memory,
        invariant: "Atomic on buffer with non-u32 element type",
        corrective_action: "Atomics only support U32 elements, so retype the buffer.",
    },
    ValidationRule {
        code: "V016",
        phase: ValidationPhase::Expression,
        invariant: "Unknown op id in `Expr::Call`",
        corrective_action: "Use a registered op id or add the op to core::ops::*.",
    },
    ValidationRule {
        code: "V018",
        phase: ValidationPhase::Limits,
        invariant: "Program nesting depth exceeds `DEFAULT_MAX_NESTING_DEPTH`",
        corrective_action: "Flatten nested If/Loop/Block structures or split the program before lowering.",
    },
    ValidationRule {
        code: "V019",
        phase: ValidationPhase::Limits,
        invariant: "Program has more than `DEFAULT_MAX_NODE_COUNT` nodes",
        corrective_action: "Split the program into smaller kernels or run an optimization pass before lowering.",
    },
    ValidationRule {
        code: "V020",
        phase: ValidationPhase::Expression,
        invariant: "Call to non-inlinable op",
        corrective_action: "Lower this op through its dedicated backend path or rewrite the caller with explicit IR.",
    },
    ValidationRule {
        code: "V021",
        phase: ValidationPhase::Expression,
        invariant: "Call argument count mismatches callee's ReadOnly/Uniform input count",
        corrective_action: "Pass exactly one argument per input buffer in binding order.",
    },
    ValidationRule {
        code: "V022",
        phase: ValidationPhase::Expression,
        invariant: "Program or callee declares too many outputs",
        corrective_action: "Mark at most one result buffer with `BufferDecl::output(...)`.",
    },
    ValidationRule {
        code: "V023",
        phase: ValidationPhase::Expression,
        invariant: "Cast to `Bytes` is unsupported in secondary text lowering",
        corrective_action: "Use buffer load/store directly for byte data.",
    },
    ValidationRule {
        code: "V025",
        phase: ValidationPhase::Memory,
        invariant: "Atomic on workgroup buffer is outside the portable memory model",
        corrective_action: "Use a storage ReadWrite buffer for atomics.",
    },
    ValidationRule {
        code: "V027",
        phase: ValidationPhase::Memory,
        invariant: "Atomic index has wrong type (expected `u32`)",
        corrective_action: "Cast the index to U32 before the atomic.",
    },
    ValidationRule {
        code: "V028",
        phase: ValidationPhase::Type,
        invariant: "Fma operand has wrong type (expected `f32`)",
        corrective_action: "Cast the operand to F32 before Fma, or use the integer mul/add form explicitly.",
    },
    ValidationRule {
        code: "V029",
        phase: ValidationPhase::Expression,
        invariant: "Select branches have mismatched types",
        corrective_action: "Cast both branches to the same type before Select.",
    },
    ValidationRule {
        code: "V030",
        phase: ValidationPhase::Expression,
        invariant: "Opaque Expr extension fails invariant (empty extension_kind/debug_identity/missing result_type/validate_extension failure)",
        corrective_action: "Return a stable non-empty `extension_kind`, a human-readable `debug_identity`, and an explicit `result_type`, and pass `validate_extension`.",
    },
    ValidationRule {
        code: "V031",
        phase: ValidationPhase::Node,
        invariant: "Opaque Node extension fails invariant (empty extension_kind/debug_identity/validate_extension failure)",
        corrective_action: "Return a stable non-empty `extension_kind`, a human-readable `debug_identity`, and pass `validate_extension`.",
    },
    ValidationRule {
        code: "V032",
        phase: ValidationPhase::Node,
        invariant: "Duplicate sibling `let` binding in the same region",
        corrective_action: "Rename one binding, or move one declaration into an inner Block/Region/Loop if a new scope is intended.",
    },
    ValidationRule {
        code: "V033",
        phase: ValidationPhase::Limits,
        invariant: "Expression nesting exceeds `DEFAULT_MAX_EXPR_DEPTH`",
        corrective_action: "Split the expression into intermediate let-bindings before lowering.",
    },
    ValidationRule {
        code: "V034",
        phase: ValidationPhase::Expression,
        invariant: "Backend does not support the requested cast target",
        corrective_action: "Choose a target type the backend supports, or validate against a backend that advertises that cast support.",
    },
    ValidationRule {
        code: "V035",
        phase: ValidationPhase::Type,
        invariant: "Narrowing cast may truncate high bits",
        corrective_action: "Use a non-narrowing target, or prove the source value fits before casting.",
    },
    ValidationRule {
        code: "V036",
        phase: ValidationPhase::Node,
        invariant: "Constant store index exceeds the declared buffer element count",
        corrective_action: "Keep constant store indices inside the declared element range.",
    },
    ValidationRule {
        code: "V041",
        phase: ValidationPhase::Expression,
        invariant: "Subgroup expressions used without backend subgroup support",
        corrective_action: "Validate with `ValidationOptions::with_backend(backend)` where `backend.supports_subgroup_ops() == true`, or remove subgroup ops before lowering.",
    },
    ValidationRule {
        code: "V042",
        phase: ValidationPhase::Memory,
        invariant: "Atomic read-modify-write uses a memory ordering the operation does not accept",
        corrective_action: "Use Relaxed, Acquire, Release, AcqRel, or SeqCst for atomic read-modify-write operations.",
    },
    ValidationRule {
        code: "V043",
        phase: ValidationPhase::Memory,
        invariant: "Barrier uses a memory ordering that does not synchronize memory",
        corrective_action: "Use Acquire, Release, AcqRel, or SeqCst; use no barrier at all for Relaxed.",
    },
    ValidationRule {
        code: "V044",
        phase: ValidationPhase::Type,
        invariant: "Binary `Div` or `Mod` has a statically-zero divisor",
        corrective_action: "Guard the divisor, use Select to substitute a non-zero value, or reject the input before building IR.",
    },
    ValidationRule {
        code: "V045",
        phase: ValidationPhase::Node,
        invariant: "Assignment value type does not match the declared binding or buffer element type",
        corrective_action: "Cast the value to the declared type, or introduce a binding or buffer with the intended type.",
    },
    ValidationRule {
        code: "V046",
        phase: ValidationPhase::Node,
        invariant: "Distributed collective node validation failure",
        corrective_action: "Validate with backend collective support, use matching collective buffer element types, declare every referenced buffer, and keep collective buffers in device/global storage.",
    },
    ValidationRule {
        code: "V047",
        phase: ValidationPhase::Expression,
        invariant: "Bitwise subgroup reduction given an f32 operand",
        corrective_action: "Use an integer operand for `And`/`Or`/`Xor`, or use `Add`/`Mul`/`Min`/`Max` for a float reduction.",
    },
    ValidationRule {
        code: "V051",
        phase: ValidationPhase::Expression,
        invariant: "Buffer reference used where a value is expected",
        corrective_action: "A buffer reference is legal only as a call argument. Pass it directly to a composite op, or read an element with `Expr::Load`.",
    },
    ValidationRule {
        code: "V052",
        phase: ValidationPhase::Expression,
        invariant: "Call passes a reference to an undeclared buffer",
        corrective_action: "Declare the buffer in `Program::buffers`.",
    },
    ValidationRule {
        code: "V053",
        phase: ValidationPhase::Expression,
        invariant: "Value passed for a `buffer<T>` parameter",
        corrective_action: "Pass `Expr::buffer_ref(name)` naming the buffer the op should read.",
    },
    ValidationRule {
        code: "V054",
        phase: ValidationPhase::Expression,
        invariant: "Referenced buffer's element type does not match the signature",
        corrective_action: "Pass a buffer whose element type matches `buffer<T>`, or change the op signature.",
    },
    ValidationRule {
        code: "V055",
        phase: ValidationPhase::Memory,
        invariant: "Synchronizing loop exit is unordered against the back edge",
        corrective_action: "Put an unconditional barrier after the early exit, as the final node in the loop body.",
    },
    ValidationRule {
        code: "V056",
        phase: ValidationPhase::Capability,
        invariant: "Backend does not support one operation used by the program",
        corrective_action: "Choose a backend that supports the operation or register its implementation.",
    },
    ValidationRule {
        code: "V057",
        phase: ValidationPhase::Memory,
        invariant: "atomic value type `…` does not match required `u32`",
        corrective_action: "Ensure the atomic operand is U32.",
    },
    ValidationRule {
        code: "V058",
        phase: ValidationPhase::Memory,
        invariant: "compare-exchange expected type `…` does not match required `u32`",
        corrective_action: "Ensure Expr::Atomic.expected is U32.",
    },
    ValidationRule {
        code: "V059",
        phase: ValidationPhase::Memory,
        invariant: "compare-exchange atomic is missing expected value",
        corrective_action: "Set Expr::Atomic.expected for AtomicOp::CompareExchange.",
    },
    ValidationRule {
        code: "V060",
        phase: ValidationPhase::Memory,
        invariant: "non-compare-exchange atomic includes an expected value",
        corrective_action: "Use Expr::Atomic.expected only with AtomicOp::CompareExchange.",
    },
    ValidationRule {
        code: "V061",
        phase: ValidationPhase::Memory,
        invariant: "atomic on unknown buffer `…`",
        corrective_action: "Declare it in Program::buffers.",
    },
    ValidationRule {
        code: "V063",
        phase: ValidationPhase::Memory,
        invariant: "store to non-writable buffer `…`",
        corrective_action: "Declare it with BufferAccess::ReadWrite, BufferAccess::WriteOnly, or BufferAccess::Workgroup.",
    },
    ValidationRule {
        code: "V064",
        phase: ValidationPhase::Memory,
        invariant: "store to unknown buffer `…`",
        corrective_action: "Declare it in Program::buffers.",
    },
    ValidationRule {
        code: "V065",
        phase: ValidationPhase::Memory,
        invariant: "load from unknown buffer `…`",
        corrective_action: "Declare it in Program::buffers.",
    },
    ValidationRule {
        code: "V066",
        phase: ValidationPhase::Expression,
        invariant: "Reference to an undeclared variable",
        corrective_action: "Add a declaration before this use.",
    },
    ValidationRule {
        code: "V067",
        phase: ValidationPhase::Expression,
        invariant: "buflen of unknown buffer `…`",
        corrective_action: "Declare it in Program::buffers.",
    },
    ValidationRule {
        code: "V068",
        phase: ValidationPhase::Expression,
        invariant: "invocation/workgroup ID axis … out of range",
        corrective_action: "Use 0 (x), 1 (y), or 2 (z).",
    },
    ValidationRule {
        code: "V070",
        phase: ValidationPhase::Program,
        invariant: "Linear, affine, or relevant buffer use count violates its declared discipline",
        corrective_action: "Add or delete buffer uses to satisfy the discipline, or select the intended `LinearType`.",
    },
    ValidationRule {
        code: "V083",
        phase: ValidationPhase::Program,
        invariant: "buffer `…` declared shape predicate `…` but has count=…",
        corrective_action: "Change the count to satisfy the predicate, or relax the predicate.",
    },
    ValidationRule {
        code: "V084",
        phase: ValidationPhase::Type,
        invariant: "64-bit integer arithmetic used where the shared IR supports only portable 32-bit arithmetic",
        corrective_action: "Express the operation as a U32 pair with explicit carry/borrow, or use a backend-specific op whose schema declares native 64-bit arithmetic.",
    },
    ValidationRule {
        code: "V085",
        phase: ValidationPhase::Type,
        invariant: "Saturating arithmetic `…` received left=`…`, right=`…`; legal set is only U32 in the current lowering",
        corrective_action: "Cast both operands to U32, or clamp explicitly for I32/F32.",
    },
    ValidationRule {
        code: "V086",
        phase: ValidationPhase::Type,
        invariant: "AbsDiff has left=`…`, right=`…` and can overflow (i32::MIN - i32::MAX invokes target-text signed-integer UB)",
        corrective_action: "Cast operands to U32 before AbsDiff, or rewrite as an explicit branch.",
    },
    ValidationRule {
        code: "V087",
        phase: ValidationPhase::Type,
        invariant: "binary operation `…` … operand has type `…`, but numeric arithmetic expects one of `u32`, `i32`, or `f32`",
        corrective_action: "Cast the operand to U32 or I32 before arithmetic, or rewrite to avoid mixing logical and arithmetic operators.",
    },
    ValidationRule {
        code: "V088",
        phase: ValidationPhase::Type,
        invariant: "binary operation `…` operands have mismatched numeric types: left=`…`, right=`…` (legal set: U32, I32, F32)",
        corrective_action: "Cast one operand so both sides share a type (target-text has no implicit promotion).",
    },
    ValidationRule {
        code: "V089",
        phase: ValidationPhase::Type,
        invariant: "binary operation `Mod` … operand must be `u32` or `i32`, got `…`. Legal set for Mod is integer-only",
        corrective_action: "Cast both operands to the same integer type before modulo.",
    },
    ValidationRule {
        code: "V090",
        phase: ValidationPhase::Type,
        invariant: "binary operation `Mod` operands have mismatched integer types: left=`…`, right=`…`",
        corrective_action: "Cast one operand so both sides share the same integer type.",
    },
    ValidationRule {
        code: "V091",
        phase: ValidationPhase::Type,
        invariant: "binary operation `…` left operand has type `…`; legal integer set is `u32` or `i32`",
        corrective_action: "Cast the left operand to U32 or I32.",
    },
    ValidationRule {
        code: "V092",
        phase: ValidationPhase::Type,
        invariant: "binary operation `…` right operand has type `…`; legal integer set is `u32` or `i32`",
        corrective_action: "Cast the right operand to U32 or I32.",
    },
    ValidationRule {
        code: "V093",
        phase: ValidationPhase::Type,
        invariant: "Integer operation operands have mismatched types",
        corrective_action: "Cast both operands to the same integer type.",
    },
    ValidationRule {
        code: "V094",
        phase: ValidationPhase::Type,
        invariant: "binary operation `…` … operand has type `…`; shift/rotate operands must be `u32`",
        corrective_action: "Cast the operand to U32 before shifting/rotating.",
    },
    ValidationRule {
        code: "V095",
        phase: ValidationPhase::Type,
        invariant: "binary operation `…` … operand has type `…`; logical And/Or operands must be `u32` or `bool`",
        corrective_action: "Cast the operand to U32 or Bool.",
    },
    ValidationRule {
        code: "V096",
        phase: ValidationPhase::Type,
        invariant: "binary comparison `…` operands have mismatched types: left=`…`, right=`…`. Comparisons require matching types",
        corrective_action: "Cast both operands to the same type before comparing.",
    },
    ValidationRule {
        code: "V097",
        phase: ValidationPhase::Type,
        invariant: "Subgroup operation used without backend subgroup capability evidence",
        corrective_action: "Validate with ValidationOptions::with_backend(backend) where `backend.supports_subgroup_ops() == true`, or remove the subgroup-dependent operation before lowering.",
    },
    ValidationRule {
        code: "V098",
        phase: ValidationPhase::Type,
        invariant: "Negation operand violates the portable total-arithmetic contract",
        corrective_action: "Use `0 - x` for wrapping i32 negation, cast to U32 before Negate, or guard with Select(i32::MIN, 0, -x).",
    },
    ValidationRule {
        code: "V099",
        phase: ValidationPhase::Type,
        invariant: "unary operation `…` operand has type `…`, but legal set is U32, I32, or F32",
        corrective_action: "Cast or rewrite the operand to U32/I32/F32.",
    },
    ValidationRule {
        code: "V100",
        phase: ValidationPhase::Type,
        invariant: "unary operation `LogicalNot` operand has type `…`; legal set is `u32` or `bool`",
        corrective_action: "Cast or rewrite the operand to produce U32 or Bool.",
    },
    ValidationRule {
        code: "V101",
        phase: ValidationPhase::Type,
        invariant: "unary operation `…` operand has type `…`; legal integer set is `u32`, `i32`, or `u64`",
        corrective_action: "Cast or rewrite the operand to produce U32, I32, or U64.",
    },
    ValidationRule {
        code: "V102",
        phase: ValidationPhase::Type,
        invariant: "unary operation `…` operand has type `…`; legal set for math ops is `f32`",
        corrective_action: "Cast or rewrite the operand to produce F32.",
    },
    ValidationRule {
        code: "V103",
        phase: ValidationPhase::Type,
        invariant: "unary operation `…` operand has type `…`; unpack ops require a 32-bit integer (`u32` or `i32`) word",
        corrective_action: "Cast or rewrite the operand to produce U32 or I32.",
    },
    ValidationRule {
        code: "V104",
        phase: ValidationPhase::Type,
        invariant: "unary operation `…` is not recognized",
        corrective_action: "Use a known UnOp variant from this enum (`Negate`, `LogicalNot`, `BitNot`, `Popcount`, `Clz`, `Ctz`, `ReverseBits`, `Sin`, `Cos`, `Exp`, `Log`, `Log2`, `Exp2`, `Tan`, `Acos`, `Asin`, `Atan`, `Tanh`, `Sinh`, `Cosh`, `Abs`, `Sqrt`, `InverseSqrt`, `Reciprocal`, `Floor`, `Ceil`, `Round`, `Trunc`, `Sign`, `IsNan`, `IsInf`, `IsFinite`, `Unpack4Low`, `Unpack4High`, `Unpack8Low`, `Unpack8High`).",
    },
    ValidationRule {
        code: "V105",
        phase: ValidationPhase::Program,
        invariant: "Program lacks one top-level Region",
        corrective_action: "Construct runnable programs with Program::wrapped or add one top-level Region.",
    },
    ValidationRule {
        code: "V106",
        phase: ValidationPhase::Program,
        invariant: "workgroup_size[…] is 0",
        corrective_action: "All workgroup dimensions must be >= 1.",
    },
    ValidationRule {
        code: "V107",
        phase: ValidationPhase::Program,
        invariant: "duplicate buffer name `…`",
        corrective_action: "Each buffer must have a unique name.",
    },
    ValidationRule {
        code: "V108",
        phase: ValidationPhase::Program,
        invariant: "duplicate binding slot … (buffer `…`)",
        corrective_action: "Each buffer must have a unique binding.",
    },
    ValidationRule {
        code: "V109",
        phase: ValidationPhase::Program,
        invariant: "workgroup buffer `…` has count 0",
        corrective_action: "Declare a positive element count.",
    },
    ValidationRule {
        code: "V110",
        phase: ValidationPhase::Program,
        invariant: "output buffer `…` uses unsupported element type `…`",
        corrective_action: "Output buffers must use fixed-width scalar or vector element types, not Array or Tensor.",
    },
    ValidationRule {
        code: "V111",
        phase: ValidationPhase::Node,
        invariant: "malformed validation frame stream: PopScope without matching PushScope",
        corrective_action: "Rebuild the program through the structured IR builder before validation.",
    },
    ValidationRule {
        code: "V112",
        phase: ValidationPhase::Node,
        invariant: "unreachable statements after `return`",
        corrective_action: "Remove statements after `return` or reorder them.",
    },
    ValidationRule {
        code: "V114",
        phase: ValidationPhase::Node,
        invariant: "malformed validation frame stream: loop variable `…` inserted outside any scope",
        corrective_action: "Rebuild the program through the structured IR builder before validation.",
    },
    ValidationRule {
        code: "V115",
        phase: ValidationPhase::Composition,
        invariant: "region `…` is marked non-composable with itself but appears multiple times in one fused program",
        corrective_action: "Split the parser into separate dispatches, or give each instance distinct scratch storage before fusion.",
    },
    ValidationRule {
        code: "V116",
        phase: ValidationPhase::Composition,
        invariant: "Fused nodes mix non-atomic reads and atomic access without an ordering barrier",
        corrective_action: "Insert `Node::barrier()` between the read path and the atomic path, or rename the buffers before fusion.",
    },
    ValidationRule {
        code: "V118",
        phase: ValidationPhase::Node,
        invariant: "malformed validation frame stream: let binding `…` appeared outside any scope",
        corrective_action: "Rebuild the program through the structured IR builder before validation.",
    },
    ValidationRule {
        code: "V119",
        phase: ValidationPhase::Node,
        invariant: "assignment to buffer `…` requires read-write storage but declared access is `…`",
        corrective_action: "Use a read-write/output buffer or store into a mutable local binding.",
    },
    ValidationRule {
        code: "V120",
        phase: ValidationPhase::Node,
        invariant: "Assignment targets an undeclared variable",
        corrective_action: "Add a declaration before this assignment.",
    },
    ValidationRule {
        code: "V121",
        phase: ValidationPhase::Node,
        invariant: "Store value type does not match the buffer element type",
        corrective_action: "Cast the value to the buffer element type or use a compatible store type.",
    },
    ValidationRule {
        code: "V122",
        phase: ValidationPhase::Node,
        invariant: "Node::Store buffer `…` index has type `…` but must be `u32`",
        corrective_action: "Cast the index to U32 before storing.",
    },
    ValidationRule {
        code: "V123",
        phase: ValidationPhase::Node,
        invariant: "Node::If condition has type `…` but must be `u32` or `bool`",
        corrective_action: "Cast or rewrite the condition expression to produce `u32` or `bool`.",
    },
    ValidationRule {
        code: "V124",
        phase: ValidationPhase::Node,
        invariant: "Node::Loop from-bound has type `…`; legal loop bound type is `u32`",
        corrective_action: "Cast the `from` bound to `u32`.",
    },
    ValidationRule {
        code: "V125",
        phase: ValidationPhase::Node,
        invariant: "Node::Loop to-bound has type `…`; legal loop bound type is `u32`",
        corrective_action: "Cast the `to` bound to `u32`.",
    },
    ValidationRule {
        code: "V126",
        phase: ValidationPhase::Node,
        invariant: "indirect dispatch offset … is not 4-byte aligned",
        corrective_action: "Use an offset aligned to a u32 dispatch count tuple.",
    },
    ValidationRule {
        code: "V127",
        phase: ValidationPhase::Node,
        invariant: "indirect dispatch references unknown buffer `…`",
        corrective_action: "Declare the count buffer before validation.",
    },
    ValidationRule {
        code: "V128",
        phase: ValidationPhase::Node,
        invariant: "async stream tag is empty",
        corrective_action: "Use a stable non-empty tag to pair AsyncLoad and AsyncWait nodes.",
    },
    ValidationRule {
        code: "V129",
        phase: ValidationPhase::Memory,
        invariant: "malformed barrier visitor dispatch",
        corrective_action: "Rebuild the program through the structured IR builder before validation.",
    },
    ValidationRule {
        code: "V130",
        phase: ValidationPhase::Program,
        invariant: "backend-allocated output buffer `…` has no static element count or output byte range",
        corrective_action: "Declare the output with `.with_count(n)`, or use `.with_output_byte_range(0..0)` for a genuinely empty output.",
    },
];

/// Every registered validation rule, ordered by code.
#[must_use]
pub fn rules() -> &'static [ValidationRule] {
    VALIDATION_RULES
}

/// Render the catalog as the generated TOML document.
///
/// The output is the exact bytes `docs/generated/error-codes.toml` must hold.
/// A gate compares the two and a caller with a write flag replaces the file
/// with this string, so neither side has to know the format twice.
#[must_use]
pub fn render_catalog_toml() -> String {
    let mut out = String::with_capacity(VALIDATION_RULES.len() * 256);
    out.push_str("# Generated from vyre_foundation::validate::catalog. Do not hand-edit.\n");
    out.push_str("schema_version = 1\n");
    for rule in VALIDATION_RULES {
        out.push_str("\n[[rule]]\ncode = ");
        push_basic_string(&mut out, rule.code);
        out.push_str("\nphase = ");
        push_basic_string(&mut out, rule.phase.as_str());
        out.push_str("\ninvariant = ");
        push_basic_string(&mut out, rule.invariant);
        out.push_str("\ncorrective_action = ");
        push_basic_string(&mut out, rule.corrective_action);
        out.push('\n');
    }
    out
}

/// Append `value` as a TOML basic string, escaping what the grammar requires.
fn push_basic_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
}
