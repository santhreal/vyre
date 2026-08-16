//! The `Expr::Call` ABI, single-homed for both reference node executors.
//!
//! Callee resolution, arity, argument encoding, the input byte budget, the
//! CPU-reference invocation and the output decode are one decision each. Both
//! the statement evaluator and the hashmap interpreter reach the same registered
//! CPU reference through them, so a change to the ABI cannot land in one arm and
//! miss the other. Each executor owns only how it evaluates an argument
//! expression, and hands that in as a closure.

use rustc_hash::FxHashMap;
use vyre_foundation::cpu_op::CpuFn;
use vyre_foundation::dialect_lookup::{Signature, TypedParam};
use vyre_foundation::ir::{DataType, Expr};
use vyre_foundation::operation::{OperationRegistry, SemanticOperation};

use crate::execution::expr_cast::spec_output_value;
use crate::value::Value;
use crate::ReferenceError;

/// Largest input payload the reference interpreter encodes for one call.
const MAX_CALL_INPUT_BYTES: usize = 64 * 1024 * 1024;

/// Output bytes reserved for a call whose first output declares no fixed width.
const BYTES_OUTPUT_RESERVE: usize = 256;

/// One callee resolved out of the global registry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedCall {
    pub(crate) operation: SemanticOperation,
}

/// Resolved callees keyed by the address of the `Expr::Call` node that named them.
///
/// The address is stable for the lifetime of the `Program` an invocation walks,
/// so it identifies a call site without re-hashing its operation id per step.
pub(crate) type OpCache = FxHashMap<*const Expr, ResolvedCall>;

/// Resolve `op_id` against the global registry, caching the result per call site.
pub(crate) fn resolve_call(
    call_expr: *const Expr,
    op_id: &str,
    cache: &mut OpCache,
) -> Result<ResolvedCall, ReferenceError> {
    if let Some(resolved) = cache.get(&call_expr).copied() {
        return Ok(resolved);
    }
    let operation = OperationRegistry::global().get(op_id).ok_or_else(|| {
        ReferenceError::new(format!(
            "unsupported call `{op_id}`. Fix: submit one canonical OperationRegistration or inline the callee as IR."
        ))
    })?;
    let resolved = ResolvedCall { operation };
    cache.insert(call_expr, resolved);
    Ok(resolved)
}

/// The callable signature of a resolved operation.
pub(crate) fn callable_signature(
    op_id: &str,
    operation: &SemanticOperation,
) -> Result<&'static Signature, ReferenceError> {
    operation.signature.ok_or_else(|| {
        ReferenceError::new(format!(
            "op `{op_id}` has no callable signature. Fix: attach a Signature to its canonical OperationRegistration before reference execution."
        ))
    })
}

/// Encode `args`, run the registered CPU reference, and decode its output.
///
/// `eval_arg` is the caller's expression evaluator. Everything else, the arity
/// check, the per-argument byte widths, the input budget, the reference lookup
/// and the output type, is the ABI and belongs here.
pub(crate) fn invoke_signature<E>(
    op_id: &str,
    signature: &Signature,
    args: &[Expr],
    eval_arg: E,
) -> Result<Value, ReferenceError>
where
    E: FnMut(&Expr) -> Result<Value, ReferenceError>,
{
    validate_arity(op_id, args.len(), signature.inputs.len())?;
    let input = encode_inputs(op_id, args, signature.inputs, eval_arg)?;
    let mut output = Vec::with_capacity(output_reserve(signature.outputs));
    let cpu_ref = crate::reference_fn(op_id).ok_or_else(|| {
        ReferenceError::new(format!(
            "op `{op_id}` has no CPU reference implementation. Fix: register one ReferenceFacet for this canonical operation or inline its composition body."
        ))
    })?;
    invoke_cpu_ref(op_id, cpu_ref, &input, &mut output)?;
    Ok(spec_output_value(
        output_data_type(signature.outputs),
        &output,
    ))
}

pub(crate) fn invoke_cpu_ref(
    op_id: &str,
    cpu_ref: CpuFn,
    input: &[u8],
    output: &mut Vec<u8>,
) -> Result<(), ReferenceError> {
    let original_len = output.len();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cpu_ref(input, output))).map_err(
        |payload| {
            output.truncate(original_len);
            ReferenceError::new(format!(
                "CPU reference for `{op_id}` panicked: {}. Fix: make the primitive reference total over byte inputs and return a structured error before registering it.",
                panic_payload_message(payload.as_ref())
            ))
        },
    )
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn validate_arity(op_id: &str, actual: usize, expected: usize) -> Result<(), ReferenceError> {
    if actual == expected {
        return Ok(());
    }
    Err(ReferenceError::new(format!(
        "call `{op_id}` received {actual} arguments but the primitive signature requires {expected}. Fix: pass exactly {expected} arguments."
    )))
}

fn encode_inputs<E>(
    op_id: &str,
    args: &[Expr],
    inputs: &[TypedParam],
    mut eval_arg: E,
) -> Result<Vec<u8>, ReferenceError>
where
    E: FnMut(&Expr) -> Result<Value, ReferenceError>,
{
    let mut input = Vec::with_capacity(reserve_for(inputs));
    for (arg, param) in args.iter().zip(inputs) {
        let width = declared_width(op_id, param.ty)?;
        eval_arg(arg)?.extend_bytes_width(width, &mut input)?;
        if input.len() > MAX_CALL_INPUT_BYTES {
            return Err(ReferenceError::new(format!(
                "call `{op_id}` requires {} input bytes, exceeding the {MAX_CALL_INPUT_BYTES}-byte reference budget. Fix: reduce call input size.",
                input.len()
            )));
        }
    }
    Ok(input)
}

/// Bytes to allocate up front for a call's input payload.
///
/// A payload parameter contributes an unknown number of bytes, so it reserves
/// nothing and the vector grows when it is appended.
fn reserve_for(inputs: &[TypedParam]) -> usize {
    inputs
        .iter()
        .map(|param| match param_encoding(param.ty) {
            Some(ParamEncoding::Scalar(width)) => width,
            Some(ParamEncoding::Payload) | None => 0,
        })
        .sum()
}

/// How the call ABI encodes one declared parameter spelling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParamEncoding {
    /// A scalar occupying a fixed number of bytes.
    Scalar(usize),
    /// A buffer: the argument contributes its own bytes, unpadded.
    Payload,
}

/// The encoding for a declared parameter spelling, or `None` when the ABI has
/// none.
///
/// `signature_param_spellings_all_have_an_encoding` walks the registry and fails
/// when a registered callable signature uses a spelling this table does not name,
/// so a new parameter type cannot silently reach a wrong width and truncate an
/// oracle input. A buffer spelling is recognized only when its element is itself a
/// scalar the ABI knows, because `buffer<something-else>` is exactly the case that
/// must stay unrecognized.
fn param_encoding(ty: &str) -> Option<ParamEncoding> {
    if let Some(width) = scalar_width(ty) {
        return Some(ParamEncoding::Scalar(width));
    }
    let element = ty.strip_prefix("buffer<")?.strip_suffix('>')?;
    scalar_width(element).map(|_| ParamEncoding::Payload)
}

/// Byte width the call ABI writes for one argument: the scalar's width, or zero
/// for a payload, which `Value::extend_bytes_width` reads as unpadded.
fn declared_width(op_id: &str, ty: &str) -> Result<usize, ReferenceError> {
    match param_encoding(ty) {
        Some(ParamEncoding::Scalar(width)) => Ok(width),
        Some(ParamEncoding::Payload) => Ok(0),
        None => Err(ReferenceError::new(format!(
            "call `{op_id}` declares parameter type `{ty}`, which the reference call ABI cannot encode. Fix: declare a spelling the ABI encodes, or give this one an encoding in `param_encoding`."
        ))),
    }
}

/// Byte width of one fixed-width scalar spelling.
fn scalar_width(ty: &str) -> Option<usize> {
    match ty {
        "u32" | "i32" | "f32" | "vec-count" => Some(4),
        "u64" | "i64" | "f64" => Some(8),
        "u8" | "i8" | "bool" => Some(1),
        _ => None,
    }
}

/// Bytes to reserve for a call's output buffer.
fn output_reserve(outputs: &[TypedParam]) -> usize {
    outputs.first().map_or(BYTES_OUTPUT_RESERVE, |param| {
        scalar_width(param.ty).unwrap_or(BYTES_OUTPUT_RESERVE)
    })
}

/// The `DataType` a call's returned bytes are decoded as.
fn output_data_type(outputs: &[TypedParam]) -> DataType {
    outputs
        .first()
        .map_or(DataType::Bytes, |param| match param.ty {
            "u32" => DataType::U32,
            "i32" => DataType::I32,
            "f32" => DataType::F32,
            _ => DataType::Bytes,
        })
}

// Inline: covers the crate-private `ParamEncoding` and `declared_width`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::{declared_width, param_encoding, ParamEncoding};
    use vyre_foundation::operation::OperationRegistry;

    /// Every parameter spelling a registered callable signature uses must have an
    /// encoding, or an unrecognized one is encoded at a width the ABI guessed and
    /// the reference oracle returns a truncated answer the conform gate then
    /// trusts.
    ///
    /// The expected membership is the registry, read at run time, so registering
    /// an operation whose signature names a new parameter type turns this red
    /// instead of reaching a fallback.
    #[test]
    fn signature_param_spellings_all_have_an_encoding() {
        let mut missing: Vec<(&str, &str, &str)> = Vec::new();
        for operation in OperationRegistry::global().iter() {
            let Some(signature) = operation.signature else {
                continue;
            };
            let params = signature
                .inputs
                .iter()
                .map(|param| ("input", param))
                .chain(signature.outputs.iter().map(|param| ("output", param)));
            for (role, param) in params {
                if param_encoding(param.ty).is_none() {
                    missing.push((operation.id, role, param.ty));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "Fix: give every spelling below an encoding in `param_encoding`, or change \
             the operation to declare a spelling that already has one: {missing:?}"
        );
    }

    /// A buffer parameter carries its whole payload, and a scalar carries its
    /// declared width. Encoding a buffer at a scalar width is the truncation this
    /// distinction exists to prevent.
    #[test]
    fn a_buffer_parameter_is_encoded_unpadded_and_a_scalar_at_its_width() {
        assert_eq!(param_encoding("u32"), Some(ParamEncoding::Scalar(4)));
        assert_eq!(param_encoding("u64"), Some(ParamEncoding::Scalar(8)));
        assert_eq!(param_encoding("buffer<u32>"), Some(ParamEncoding::Payload));
        assert_eq!(param_encoding("buffer<f32>"), Some(ParamEncoding::Payload));
        assert_eq!(declared_width("op", "u32").expect("a scalar encodes"), 4);
        assert_eq!(
            declared_width("op", "buffer<f32>").expect("a buffer encodes"),
            0,
            "a zero width is what makes the append unpadded"
        );
    }

    /// An element spelling the ABI does not know keeps the buffer unrecognized,
    /// and an unrecognized spelling is refused rather than guessed at.
    #[test]
    fn an_unknown_spelling_is_refused_with_the_type_named() {
        assert_eq!(param_encoding("buffer<matrix>"), None);
        assert_eq!(param_encoding("texture<f32>"), None);
        let error = declared_width("demo.op", "buffer<matrix>")
            .expect_err("an unknown spelling has no encoding");
        let message = error.to_string();
        assert!(
            message.contains("buffer<matrix>") && message.contains("demo.op"),
            "the refusal names the operation and the spelling: {message}"
        );
    }
}
