//! Compile-time expansion of `Expr::Call` composition nodes.
//!
//! Calls are resolved against Category A operation programs and expanded into
//! ordinary IR before backend lowering. No runtime dispatch or GPU-side
//! interpreter is introduced by this pass.

use crate::error::{IrError as Error, IrResult as Result};
use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::expr::Ident;
use crate::ir_inner::model::node::Node;
use crate::ir_inner::model::program::{BufferDecl, Program};
use crate::ir_inner::model::types::{BufferAccess, DataType};
use rustc_hash::FxHashMap as HashMap;

/// Resolve an operation id to the canonical IR program for that operation.
pub type OpResolver = fn(&str) -> Option<Program>;

/// What the inliner does with a call the resolver cannot expand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnresolvedCalls {
    /// Fail with [`Error::InlineUnknownOp`].
    ///
    /// Backends cannot emit a call, so every path that feeds a backend
    /// demands that all calls disappear.
    Reject,
    /// Leave the call node in place.
    ///
    /// The reference interpreter needs only composite ops expanded: an
    /// intrinsic has no composition body to inline and is executed through
    /// its registered CPU function instead. Rejecting those would make the
    /// interpreter refuse every program that uses one.
    Keep,
}

/// Inline all `Expr::Call` nodes in a program using the built-in operation set.
///
/// # Errors
///
/// Returns [`Error::InlineUnknownOp`] when a call cannot be resolved,
/// [`Error::InlineNonInlinable`] when a registered operation must dispatch as a
/// separate kernel, and [`Error::InlineCycle`] when recursive operation
/// composition is detected.
#[inline]
#[must_use]
pub fn inline_calls(program: &Program) -> Result<Program> {
    inline_calls_with_resolver(program, default_resolver)
}

/// Inline all `Expr::Call` nodes with a caller-supplied operation resolver.
///
/// This entry point exists for tests and embedders that provide their own
/// operation registry. The resolver must return complete Category A programs;
/// intrinsic-only operations are not valid inline targets.
///
/// # Errors
///
/// Returns [`Error::InlineUnknownOp`] when a call cannot be resolved,
/// [`Error::InlineNonInlinable`] when a registered operation must dispatch as a
/// separate kernel, and [`Error::InlineCycle`] when recursive operation
/// composition is detected.
#[inline]
#[must_use]
pub fn inline_calls_with_resolver(program: &Program, resolver: OpResolver) -> Result<Program> {
    inline_calls_with_mode(program, resolver, UnresolvedCalls::Reject)
}

/// Expand every composition body registered in the process dialect and leave
/// intrinsic calls in place.
///
/// This is the form the reference interpreter runs before execution. A
/// composite op is defined by its IR body, so executing one means executing
/// that body; without this pass the interpreter reaches for a CPU function
/// the op never registered and silently produces an empty result.
///
/// # Errors
///
/// Returns [`Error::InlineCycle`] when recursive composition is detected, and
/// the structural inline errors when a resolved body is not inlineable.
#[inline]
#[must_use]
pub fn inline_composite_calls(program: &Program) -> Result<Program> {
    inline_calls_with_mode(program, default_resolver, UnresolvedCalls::Keep)
}

/// Inline with an explicit resolver and unresolved-call policy.
///
/// # Errors
///
/// As [`inline_calls_with_resolver`], except that
/// [`UnresolvedCalls::Keep`] suppresses [`Error::InlineUnknownOp`].
#[inline]
#[must_use]
pub fn inline_calls_with_mode(
    program: &Program,
    resolver: OpResolver,
    unresolved: UnresolvedCalls,
) -> Result<Program> {
    // Rebuilding the node tree costs a full traversal plus a fresh allocation
    // of every node. Most programs contain no call at all, and since the
    // reference interpreter now runs this on every execution, paying that on
    // a call-free program would be a per-run tax for nothing. A `Program`
    // clone is an Arc bump.
    if !contains_call(program) {
        return Ok(program.clone());
    }
    let mut ctx = InlineCtx::new_with_mode(resolver, unresolved);
    let entry = ctx.inline_nodes(program.entry())?;
    // Reuse the buffer Arc + buffer_index from the source program instead
    // of re-cloning + re-interning via Program::wrapped.
    Ok(program.with_rewritten_wrapped_entry(entry))
}

/// Resolve inline calls against the process-wide dialect lookup.
///
/// Foundation does not host a dialect registry, so it asks through the
/// `DialectLookup` dependency-inversion boundary that the driver layer
/// installs. An op resolves when it is registered AND carries a
/// composition body; intrinsics have no body to inline and stay
/// unresolved, as does any op id when no provider is installed. Callers
/// that need a different registry pass their own resolver to
/// [`inline_calls_with_resolver`].
///
/// This is what lets a registered op call another registered op and still
/// reach a backend. While this returned `None` unconditionally, the
/// canonical pre-emit pipeline resolved nothing at all, so every
/// `Expr::Call` outside `vyre-aot` failed with [`Error::InlineUnknownOp`].
#[inline]
#[must_use]
pub fn default_resolver(op_id: &str) -> Option<Program> {
    let lookup = crate::dispatch::dialect_lookup::dialect_lookup()?;
    lookup.lookup(lookup.intern_op(op_id))?.program()
}

/// Whether the program contains any `Expr::Call` at all.
#[inline]
fn contains_call(program: &Program) -> bool {
    let mut found = false;
    crate::transform::visit::walk_exprs(program, |expr| {
        found |= matches!(expr, Expr::Call { .. });
    });
    found
}

/// Mutable state for one inline expansion pass.
pub struct InlineCtx {
    /// Operation resolver used for `Expr::Call` targets.
    resolver: OpResolver,
    /// What to do with a call the resolver cannot expand.
    unresolved: UnresolvedCalls,
    /// Active expansion stack used to reject recursive composition.
    stack: Vec<String>,
    /// Monotonic suffix for generated temporary names.
    next_call_id: usize,
}

mod expand;
mod impl_inlinectx;

/// Map a callee's input buffers to the argument expressions from a call site.
#[inline]
pub(crate) fn input_arg_map(callee: &Program, args: Vec<Expr>) -> HashMap<Ident, Expr> {
    let mut inputs = input_buffers(callee);
    inputs.sort_by_key(|buf| buf.binding());
    inputs
        .into_iter()
        .zip(args)
        .map(|(buf, arg)| (Ident::from(buf.name()), arg))
        .collect()
}

/// Return read-only and uniform buffers that receive call arguments.
#[must_use]
#[inline]
pub(crate) fn input_buffers(callee: &Program) -> Vec<&BufferDecl> {
    callee
        .buffers()
        .iter()
        .filter(|buf| matches!(buf.access(), BufferAccess::ReadOnly | BufferAccess::Uniform))
        .collect()
}

/// Return the single output buffer required for an inlineable callee.
///
/// # Errors
///
/// Returns an inline error when the callee has no output buffer or more than
/// one output buffer.
#[inline]
#[must_use]
pub fn output_buffer<'a>(op_id: &str, program: &'a Program) -> Result<&'a BufferDecl> {
    let outputs: Vec<&BufferDecl> = program
        .buffers()
        .iter()
        .filter(|buf| buf.is_output())
        .collect();
    match outputs.as_slice() {
        [output] => Ok(output),
        [] => Err(Error::InlineNoOutput {
            op_id: op_id.to_string(),
        }),
        outputs => Err(Error::InlineOutputCountMismatch {
            op_id: op_id.to_string(),
            got: outputs.len(),
        }),
    }
}

/// Construct the zero literal used when an inline target needs a default value.
#[inline]
#[must_use]
pub fn zero_value(ty: &DataType) -> Expr {
    match ty {
        DataType::I32 => Expr::i32(0),
        DataType::Bool => Expr::LitBool(false),
        DataType::F32 | DataType::F16 | DataType::BF16 | DataType::F64 => Expr::f32(0.0),
        _ => Expr::u32(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_inner::model::expr::Expr;
    use crate::ir_inner::model::node::Node;
    use crate::ir_inner::model::program::BufferDecl;

    fn make_caller() -> Program {
        Program::wrapped(
            vec![
                BufferDecl::storage("A", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
                BufferDecl::output("out", 1, DataType::U32).with_count(1),
            ],
            [1, 1, 1],
            vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::Call {
                    op_id: "add_one".into(),
                    args: vec![Expr::Load {
                        buffer: "A".into(),
                        index: Box::new(Expr::u32(0)),
                    }],
                },
            )],
        )
    }

    fn make_callee() -> Program {
        Program::wrapped(
            vec![
                BufferDecl::storage("x", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
                BufferDecl::output("result", 1, DataType::U32).with_count(1),
            ],
            [1, 1, 1],
            vec![Node::store(
                "result",
                Expr::u32(0),
                Expr::add(
                    Expr::Load {
                        buffer: "x".into(),
                        index: Box::new(Expr::u32(0)),
                    },
                    Expr::u32(1),
                ),
            )],
        )
    }

    fn test_resolver(op_id: &str) -> Option<Program> {
        if op_id == "add_one" {
            Some(make_callee())
        } else {
            None
        }
    }

    #[test]
    fn test_inline_call_success() {
        let caller = make_caller();
        let inlined = inline_calls_with_resolver(&caller, test_resolver).unwrap();

        // The call should be gone
        let nodes = inlined.entry();
        // Since we inline a store, we expect an internal let for the argument or a direct replacement
        // Just verify we don't have Expr::Call anymore
        let mut has_call = false;
        let dump = format!("{nodes:?}");
        if dump.contains("Call {") {
            has_call = true;
        }
        assert!(!has_call, "Expr::Call should be inlined out: {dump}");
    }

    #[test]
    fn test_inline_unknown_op() {
        let caller = make_caller();
        // `make_caller` calls a test-only op id that no dialect registers,
        // so the default resolver cannot find it whether or not a provider
        // is installed in this process.
        let err = inline_calls(&caller).unwrap_err();
        assert!(matches!(err, Error::InlineUnknownOp { .. }));
    }

    /// With no dialect provider installed, the default resolver has nothing
    /// to ask and must say so rather than panicking or blocking.
    #[test]
    fn default_resolver_without_a_provider_resolves_nothing() {
        assert!(default_resolver("vyre-libs::definitely::not::registered").is_none());
    }

    #[test]
    fn test_zero_value() {
        assert_eq!(zero_value(&DataType::I32), Expr::i32(0));
        assert_eq!(zero_value(&DataType::F32), Expr::f32(0.0));
        assert_eq!(zero_value(&DataType::Bool), Expr::LitBool(false));
        assert_eq!(zero_value(&DataType::U32), Expr::u32(0));
    }
}
