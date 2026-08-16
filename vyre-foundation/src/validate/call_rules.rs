//! Validation of `Expr::Call` against the registered op signature.
//!
//! A call is the one place the IR reaches outside the program for meaning: the
//! argument list only makes sense against the signature the dialect registered
//! for that op id. These rules resolve the op, check the arity, and check each
//! argument, including the `buffer<T>` parameters that take a whole buffer
//! rather than a value.

use crate::dialect_lookup::Signature;
use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::program::BufferDecl;
use crate::ir_inner::model::op_signature::DataType;
use crate::operation::OperationRegistry;
use crate::validate::typecheck::{expr_type, ScopeTypes};
use crate::validate::{err, Binding, ValidationError};
use crate::validate::{ValidationLocation, ValidationPhase};
use rustc_hash::FxHashMap;

pub(crate) fn validate_call(
    op_id: &str,
    args: &[Expr],
    buffers: &FxHashMap<&str, &BufferDecl>,
    scope: &FxHashMap<crate::ir::Ident, Binding>,
    errors: &mut Vec<ValidationError>,
) {
    let Some(operation) = OperationRegistry::global().get(op_id) else {
        errors.push(err(
            "V016",
            ValidationPhase::Expression,
            ValidationLocation::Program,
            format!("call references unknown op `{op_id}`"),
            format!(
            "submit one canonical OperationRegistration for `{op_id}` or inline/remove this call."
        ),
        ));
        return;
    };
    let Some(signature) = operation.signature else {
        errors.push(err(
            "V016",
            ValidationPhase::Expression,
            ValidationLocation::Program,
            format!("call references operation `{op_id}` without a callable signature"),
            "attach a Signature to its canonical OperationRegistration or inline the composition."
                .to_string(),
        ));
        return;
    };
    validate_call_signature(op_id, signature, args, buffers, scope, errors);
}

fn validate_call_signature(
    op_id: &str,
    signature: &Signature,
    args: &[Expr],
    buffers: &FxHashMap<&str, &BufferDecl>,
    scope: &FxHashMap<crate::ir::Ident, Binding>,
    errors: &mut Vec<ValidationError>,
) {
    let expected = signature.inputs.len();
    if args.len() != expected {
        errors.push(err(
            "V020",
            ValidationPhase::Expression,
            ValidationLocation::Program,
            format!(
                "call `{op_id}` has {} arguments but signature expects {expected}",
                args.len()
            ),
            format!("pass exactly {expected} arguments in the order declared by the op signature"),
        ));
        return;
    }

    for (index, (arg, param)) in args.iter().zip(signature.inputs.iter()).enumerate() {
        if let Some(element) = buffer_element_spelling(param.ty) {
            validate_buffer_argument(op_id, index, param.name, element, arg, buffers, errors);
            continue;
        }
        let Some(expected_ty) = data_type_from_signature_spelling(param.ty) else {
            errors.push(err(
    "V021",
    ValidationPhase::Expression,
    ValidationLocation::Program,
    format!(
                "call `{op_id}` signature input `{}` uses unknown type spelling `{}`",
                param.name,
                param.ty
            ),
    "register a foundation-known scalar/vector type spelling for this parameter or validate it in the dialect layer"
));
            continue;
        };
        let Some(actual_ty) = expr_type(arg, &mut ScopeTypes::new(buffers, scope)) else {
            continue;
        };
        if actual_ty != expected_ty {
            errors.push(err(
    "V022",
    ValidationPhase::Expression,
    ValidationLocation::Program,
    format!(
                "call `{op_id}` argument {index} (`{}`) has type `{actual_ty}` but signature expects `{expected_ty}`",
                param.name
            ),
    "cast or rewrite the argument to match the registered op signature"
));
        }
    }
}

/// Return the element type spelling of a buffer parameter.
///
/// A parameter written `buffer<u32>` takes a whole buffer rather than a
/// value: the caller passes [`Expr::BufferRef`] and the inliner rebinds the
/// callee's own buffer onto it. Every other spelling names a value type.
#[inline]
fn buffer_element_spelling(spelling: &str) -> Option<&str> {
    spelling.strip_prefix("buffer<")?.strip_suffix('>')
}

/// Check one `buffer<T>` argument: it must be a reference to a declared
/// buffer whose element type is `T`.
fn validate_buffer_argument(
    op_id: &str,
    index: usize,
    param_name: &str,
    element: &str,
    arg: &Expr,
    buffers: &FxHashMap<&str, &BufferDecl>,
    errors: &mut Vec<ValidationError>,
) {
    let Some(expected) = data_type_from_signature_spelling(element) else {
        errors.push(err(
    "V021",
    ValidationPhase::Expression,
    ValidationLocation::Program,
    format!(
            "call `{op_id}` signature input `{param_name}` uses unknown buffer element spelling `{element}`"
        ),
    "use a foundation-known scalar/vector type spelling inside `buffer<...>`.".to_string()
));
        return;
    };
    let Expr::BufferRef { buffer } = arg else {
        errors.push(err(
            "V053",
            ValidationPhase::Expression,
            ValidationLocation::Program,
            format!(
                "call `{op_id}` argument {index} (`{param_name}`) is declared `buffer<{element}>` but a value was passed"
            ),
            "pass `Expr::BufferRef { buffer }` naming the buffer this op should read."
                .to_string(),
        ));
        return;
    };
    let Some(decl) = buffers.get(buffer.as_str()) else {
        errors.push(err(
            "V052",
            ValidationPhase::Expression,
            ValidationLocation::Program,
            format!("call to `{op_id}` passes a reference to unknown buffer `{buffer}`"),
            "declare it in Program::buffers.".to_string(),
        ));
        return;
    };
    let actual = decl.element();
    if actual != expected {
        errors.push(err(
    "V054",
    ValidationPhase::Expression,
    ValidationLocation::Program,
    format!(
            "call `{op_id}` argument {index} (`{param_name}`) references buffer `{buffer}` with element type `{actual}` but the signature declares `buffer<{expected}>`"
        ),
    "pass a buffer whose element type matches, or change the op signature.".to_string()
));
    }
}

fn data_type_from_signature_spelling(spelling: &str) -> Option<DataType> {
    match spelling {
        "u8" | "U8" => Some(DataType::U8),
        "u16" | "U16" => Some(DataType::U16),
        "u32" | "U32" => Some(DataType::U32),
        "u64" | "U64" => Some(DataType::U64),
        "i8" | "I8" => Some(DataType::I8),
        "i16" | "I16" => Some(DataType::I16),
        "i32" | "I32" => Some(DataType::I32),
        "i64" | "I64" => Some(DataType::I64),
        "f32" | "F32" => Some(DataType::F32),
        "bool" | "Bool" => Some(DataType::Bool),
        "bytes" | "Bytes" => Some(DataType::Bytes),
        "vec2<u32>" | "Vec2U32" => Some(DataType::Vec2U32),
        "vec4<u32>" | "Vec4U32" => Some(DataType::Vec4U32),
        _ => None,
    }
}
