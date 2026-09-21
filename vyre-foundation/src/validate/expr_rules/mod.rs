use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::op_signature::DataType;
use crate::ir_inner::model::program::BufferDecl;
use crate::validate::atomic_rules;
use crate::validate::bytes_rejection;
use crate::validate::call_rules::validate_call;
use crate::validate::cast::{cast_is_narrowing, cast_is_valid, cast_target_set};
use crate::validate::depth;
use crate::validate::report::warn;
use crate::validate::typecheck::{self, expr_type, ScopeTypes};
use crate::validate::{err, Binding, ValidationError, ValidationOptions, ValidationReport};
use crate::validate::{ValidationLocation, ValidationPhase};
use rustc_hash::FxHashMap;

#[allow(clippy::too_many_lines)]
#[inline]
pub(crate) fn validate_expr(
    expr: &Expr,
    buffers: &FxHashMap<&str, &BufferDecl>,
    scope: &FxHashMap<crate::ir::Ident, Binding>,
    options: ValidationOptions<'_>,
    report: &mut ValidationReport,
    depth_level: usize,
) {
    if !depth::check_expr_depth(depth_level, &mut report.errors) {
        return;
    }
    match expr {
        Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF32(_) | Expr::LitBool(_) => {}
        Expr::Var(name) => {
            if !scope.contains_key(name.as_str()) {
                report.errors.push(err(
                    "V066",
                    ValidationPhase::Expression,
                    ValidationLocation::Program,
                    format!("reference to undeclared variable `{name}`"),
                    format!("add `let {name} = ...;` before this use."),
                ));
            }
        }
        // A buffer reference names a buffer instead of producing a value, so
        // it has no type and nothing downstream can consume it. It is legal
        // only as a call argument, where the inliner rebinds the callee's
        // parameter onto this buffer. The `Expr::Call` arm below validates
        // arguments in that position; reaching it here means it appeared
        // somewhere that expects a value.
        Expr::BufferRef { buffer } => {
            report.errors.push(err("V051", ValidationPhase::Expression, ValidationLocation::Program, format!(
                "buffer reference `{buffer}` is not a value and is legal only as a call argument"
            ), format!(
                "pass it directly as an argument to a composite op, or use `Expr::Load {{ buffer: {buffer}, index }}` to read an element."
            )));
        }
        Expr::Load { buffer, index } => {
            bytes_rejection::check_load(buffer, buffers, &mut report.errors);
            validate_expr(index, buffers, scope, options, report, depth_level + 1);
        }
        Expr::BufLen { buffer } => {
            if !buffers.contains_key(buffer.as_str()) {
                report.errors.push(err(
                    "V067",
                    ValidationPhase::Expression,
                    ValidationLocation::Program,
                    format!("buflen of unknown buffer `{buffer}`"),
                    "declare it in Program::buffers.".to_string(),
                ));
            }
        }
        Expr::InvocationId { axis }
        | Expr::WorkgroupId { axis }
        | Expr::LocalId { axis }
        | Expr::LogicalIndex { axis }
        | Expr::LogicalTileId { axis }
        | Expr::LogicalWithinTileId { axis } => {
            if *axis > 2 {
                report.errors.push(err(
                    "V068",
                    ValidationPhase::Expression,
                    ValidationLocation::Program,
                    format!("invocation/workgroup ID axis {axis} out of range"),
                    "use 0 (x), 1 (y), or 2 (z).".to_string(),
                ));
            }
        }
        Expr::BinOp { op, left, right } => {
            validate_expr(left, buffers, scope, options, report, depth_level + 1);
            validate_expr(right, buffers, scope, options, report, depth_level + 1);
            typecheck::validate_binop_operands(
                *op,
                left,
                right,
                buffers,
                scope,
                options.requires_subgroup_ops(),
                &mut report.errors,
            );
        }
        Expr::UnOp { op, operand } => {
            validate_expr(operand, buffers, scope, options, report, depth_level + 1);
            typecheck::validate_unop_operand(op, operand, buffers, scope, &mut report.errors);
        }
        Expr::Call { op_id, args } => {
            for arg in args {
                // Argument position is the one place a buffer reference is
                // legal, so skip the value rules that reject it. The
                // signature check below (`validate_buffer_argument`) owns
                // every rule about which buffer is acceptable here.
                if matches!(arg, Expr::BufferRef { .. }) {
                    continue;
                }
                validate_expr(arg, buffers, scope, options, report, depth_level + 1);
            }
            validate_call(op_id.as_str(), args, buffers, scope, &mut report.errors);
        }
        Expr::Fma { a, b, c } => {
            validate_expr(a, buffers, scope, options, report, depth_level + 1);
            validate_expr(b, buffers, scope, options, report, depth_level + 1);
            validate_expr(c, buffers, scope, options, report, depth_level + 1);
            // VAL-002: Fma requires f32 operands on every input. target-text `fma`
            // (and the reference interpreter's Fma path) are defined for
            // floats; integer operands silently become (a * b + c) via
            // u32 arithmetic today, which is NOT what the node promises.
            for (slot, operand) in [("a", a.as_ref()), ("b", b.as_ref()), ("c", c.as_ref())] {
                if let Some(ty) = expr_type(operand, &mut ScopeTypes::new(buffers, scope)) {
                    if ty != DataType::F32 {
                        report.errors.push(err("V028", ValidationPhase::Type, ValidationLocation::Operand {
                                node: 0,
                                operand: match slot {
                                    "a" => 0,
                                    "b" => 1,
                                    _ => 2,
                                },
                            }, format!(
                                "Fma requires three f32 operands. Fma operand `{slot}` has type `{ty}`, must be `f32`"
                            ), "cast the operand to F32 before Fma, or use the integer mul/add form explicitly.".to_string()));
                    }
                }
            }
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            validate_expr(cond, buffers, scope, options, report, depth_level + 1);
            validate_expr(true_val, buffers, scope, options, report, depth_level + 1);
            validate_expr(false_val, buffers, scope, options, report, depth_level + 1);
            // VAL-002: Select requires the two branches to agree on type.
            // Divergent branch types give the node an ambiguous static type
            // and break downstream lowering + reference evaluation.
            let t_ty = expr_type(true_val, &mut ScopeTypes::new(buffers, scope));
            let f_ty = expr_type(false_val, &mut ScopeTypes::new(buffers, scope));
            if let (Some(t), Some(f)) = (&t_ty, &f_ty) {
                if t != f {
                    report.errors.push(err(
                        "V029",
                        ValidationPhase::Expression,
                        ValidationLocation::Program,
                        format!("Select branches have mismatched types: true=`{t}`, false=`{f}`"),
                        "cast both branches to the same type before Select.".to_string(),
                    ));
                }
            }
        }
        Expr::Cast { target, value } => {
            validate_expr(value, buffers, scope, options, report, depth_level + 1);
            if !options.supports_cast_target(target) {
                report.errors.push(err("V034", ValidationPhase::Expression, ValidationLocation::Program, format!(
                    "backend `{}` does not support cast target `{target}`",
                    options.backend_name()
                ), format!("choose a target type this backend supports, or validate against a backend that advertises `{target}` cast support")));
            }
            if let Some(src) = expr_type(value, &mut ScopeTypes::new(buffers, scope)) {
                if target == &DataType::Bytes && src != DataType::Bytes {
                    report.errors.push(err(
                        "V023",
                        ValidationPhase::Expression,
                        ValidationLocation::Program,
                        "cast to Bytes is unsupported in target-text lowering".to_string(),
                        "use buffer load/store directly for byte data.".to_string(),
                    ));
                } else if !cast_is_valid(&src, target) {
                    let legal_targets = cast_target_set(&src);
                    report.errors.push(err("V012", ValidationPhase::Expression, ValidationLocation::Program, format!(
                        "unsupported cast from `{src}` to `{target}`. Source type `{src}` legal targets are {legal_targets}. Choose one of those targets or rewrite this cast expression before validation"
                    ), "rewrite the program to satisfy this validation invariant"));
                } else if cast_is_narrowing(&src, target) {
                    let legal_targets = cast_target_set(&src);
                    report.warnings.push(warn(
                        "V035",
                        ValidationLocation::Program,
                        format!("narrowing cast from `{src}` to `{target}` may truncate high bits"),
                        format!("source type `{src}` legal targets are {legal_targets}; use a non-narrowing target or prove the source value fits before casting"),
                    ));
                }
            }
        }
        Expr::Atomic {
            op,
            buffer,
            index,
            expected,
            value,
            ordering,
        } => {
            atomic_rules::validate_atomic(
                *op,
                buffer,
                index,
                expected.as_deref(),
                value,
                *ordering,
                buffers,
                scope,
                &mut report.errors,
            );
            validate_expr(index, buffers, scope, options, report, depth_level + 1);
            if let Some(expected) = expected {
                validate_expr(expected, buffers, scope, options, report, depth_level + 1);
            }
            validate_expr(value, buffers, scope, options, report, depth_level + 1);
        }
        Expr::SubgroupBallot { cond } => {
            validate_expr(cond, buffers, scope, options, report, depth_level + 1);
            validate_subgroup_expr_support(&mut report.errors, options);
        }
        Expr::SubgroupShuffle { value, lane } => {
            validate_expr(value, buffers, scope, options, report, depth_level + 1);
            validate_expr(lane, buffers, scope, options, report, depth_level + 1);
            validate_subgroup_expr_support(&mut report.errors, options);
        }
        Expr::SubgroupReduce { op, value } => {
            validate_expr(value, buffers, scope, options, report, depth_level + 1);
            validate_subgroup_expr_support(&mut report.errors, options);
            // V047: bitwise subgroup reductions (And/Or/Xor) are undefined over
            // floats, both the target emitters and the reference oracle already fail
            // closed on an f32 operand (`SubgroupReduceOp::combine_f32` returns
            // None for bitwise ops). Reject it here at the type boundary so the
            // failure is uniform across every backend instead of surfacing late
            // (and only on backends whose emit happens to catch it).
            if op.is_bitwise() {
                if let Some(DataType::F32) = expr_type(value, &mut ScopeTypes::new(buffers, scope))
                {
                    report.errors.push(err("V047", ValidationPhase::Expression, ValidationLocation::Program, format!(
                        "subgroup `{op:?}` is a bitwise reduction and rejects f32 operands (its value has type `f32`)"
                    ), "use an integer operand (u32/i32) for And/Or/Xor, or use Add/Mul/Min/Max for a float reduction.".to_string()));
                }
            }
        }
        Expr::SubgroupLocalId | Expr::SubgroupSize => {
            validate_subgroup_expr_support(&mut report.errors, options);
        }
        Expr::Opaque(extension) => {
            validate_expr_extension(extension.as_ref(), &mut report.errors);
        }
    }
}

#[inline]
fn validate_subgroup_expr_support(
    errors: &mut Vec<ValidationError>,
    options: ValidationOptions<'_>,
) {
    if !options.requires_subgroup_ops() {
        errors.push(err("V041", ValidationPhase::Expression, ValidationLocation::Program, "subgroup expressions require backend subgroup-ops support".to_string(), "Validate with ValidationOptions::with_backend(backend) where backend.supports_subgroup_ops() == true.".to_string()));
    }
}

fn validate_expr_extension(
    extension: &dyn crate::ir_inner::model::expr::ExprNode,
    errors: &mut Vec<ValidationError>,
) {
    if extension.extension_kind().is_empty() {
        errors.push(err(
            "V030",
            ValidationPhase::Expression,
            ValidationLocation::Program,
            "opaque expression extension has an empty extension_kind",
            "return a stable non-empty namespace from ExprNode::extension_kind.",
        ));
    }
    if extension.debug_identity().is_empty() {
        errors.push(err(
            "V030",
            ValidationPhase::Expression,
            ValidationLocation::Program,
            format!(
                "opaque expression extension `{}` has an empty debug_identity",
                extension.extension_kind()
            ),
            "return a stable human-readable identity from ExprNode::debug_identity",
        ));
    }
    if extension.result_type().is_none() {
        errors.push(err("V030", ValidationPhase::Expression, ValidationLocation::Program, format!(
            "opaque expression extension `{}`/`{}` has no static result type",
            extension.extension_kind(),
            extension.debug_identity()
        ), "implement ExprNode::result_type so validation, CSE, and backends know the produced DataType"));
    }
    if let Err(message) = extension.validate_extension() {
        errors.push(err(
            "V030",
            ValidationPhase::Expression,
            ValidationLocation::Program,
            format!(
                "opaque expression extension `{}`/`{}` failed validation: {message}",
                extension.extension_kind(),
                extension.debug_identity()
            ),
            "rewrite the program to satisfy this validation invariant",
        ));
    }
}

#[inline]
pub(crate) fn validate_output_markers(buffers: &[BufferDecl], errors: &mut Vec<ValidationError>) {
    let outputs = output_marker_count(buffers);
    if outputs > 1 {
        errors.push(err(
            "V022",
            ValidationPhase::Expression,
            ValidationLocation::Program,
            format!("program declares {outputs} output buffers"),
            "mark at most one result buffer with BufferDecl::output(...).".to_string(),
        ));
    }
}

#[inline]
#[must_use]
pub(crate) fn output_marker_count(buffers: &[BufferDecl]) -> usize {
    buffers.iter().filter(|buf| buf.is_output()).count()
}

#[cfg(test)]
mod tests;
