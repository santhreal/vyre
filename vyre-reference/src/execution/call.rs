//! Evaluation of `Expr::Call` for the reference interpreter.

use crate::ReferenceError;
use vyre_foundation::cpu_op::CpuFn;
use vyre_foundation::ir::{DataType, Expr, Program};
use vyre_foundation::operation::OperationRegistry;
use vyre_foundation::TypedParam;

use crate::execution::expr_cast::spec_output_value;
use crate::{
    value::Value,
    workgroup::{Invocation, Memory, ResolvedCall},
};

const MAX_CALL_INPUT_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn eval_call(
    call_expr: *const Expr,
    op_id: &str,
    args: &[Expr],
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<Value, crate::ReferenceError> {
    let ResolvedCall { operation } = resolve_call(call_expr, op_id, invocation)?;
    let signature = operation.signature.ok_or_else(|| {
        ReferenceError::new(format!(
            "op `{op_id}` has no callable signature. Fix: attach a Signature to its canonical OperationRegistration before reference execution."
        ))
    })?;
    {
        validate_arity(op_id, args.len(), signature.inputs.len())?;
        let input = encode_inputs(op_id, args, signature.inputs, invocation, memory, program)?;

        let out_bytes = signature
            .outputs
            .first()
            .map(|p| match p.ty {
                "u32" | "i32" | "f32" | "vec-count" => 4,
                "u64" | "i64" | "f64" => 8,
                "u8" | "i8" | "bool" => 1,
                _ => 256,
            })
            .unwrap_or(256);
        let mut output = Vec::with_capacity(out_bytes);
        let cpu_ref = crate::reference_fn(op_id).ok_or_else(|| {
            ReferenceError::new(format!(
                "op `{op_id}` has no CPU reference implementation. Fix: register one ReferenceFacet for this canonical operation or inline its composition body."
            ))
        })?;
        invoke_cpu_ref(op_id, cpu_ref, &input, &mut output)?;

        let parsed_out_type = signature
            .outputs
            .first()
            .map(|p| match p.ty {
                "u32" => DataType::U32,
                "i32" => DataType::I32,
                "f32" => DataType::F32,
                "u8" => DataType::Bytes,
                "bool" => DataType::Bytes,
                _ => DataType::Bytes,
            })
            .unwrap_or(DataType::Bytes);

        Ok(spec_output_value(parsed_out_type, &output))
    }
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

fn resolve_call(
    call_expr: *const Expr,
    op_id: &str,
    invocation: &mut Invocation<'_>,
) -> Result<ResolvedCall, crate::ReferenceError> {
    if let Some(resolved) = invocation.op_cache.get(&call_expr).copied() {
        return Ok(resolved);
    }
    let operation = OperationRegistry::global().get(op_id).ok_or_else(|| {
        ReferenceError::new(format!(
            "unsupported call `{op_id}`. Fix: submit one canonical OperationRegistration or inline the callee as IR."
        ))
    })?;
    let resolved = ResolvedCall { operation };
    invocation.op_cache.insert(call_expr, resolved);
    Ok(resolved)
}

fn validate_arity(
    op_id: &str,
    actual: usize,
    expected: usize,
) -> Result<(), crate::ReferenceError> {
    if actual == expected {
        return Ok(());
    }
    Err(ReferenceError::new(format!(
        "call `{op_id}` received {actual} arguments but the primitive signature requires {expected}. Fix: pass exactly {expected} arguments."
    )))
}

fn encode_inputs(
    op_id: &str,
    args: &[Expr],
    inputs: &[TypedParam],
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<Vec<u8>, crate::ReferenceError> {
    let mut input = Vec::with_capacity(inputs.iter().map(|param| param_width(param.ty)).sum());
    for (arg, param) in args.iter().zip(inputs) {
        let declared_width = param_width(param.ty);
        let next_len = input.len().checked_add(declared_width).ok_or_else(|| {
            ReferenceError::new(format!(
                "call `{op_id}` input byte size overflows usize. Fix: reduce the argument count or byte payload size."
            ))
        })?;
        if next_len > MAX_CALL_INPUT_BYTES {
            return Err(ReferenceError::new(format!(
                "call `{op_id}` requires {next_len} input bytes, exceeding the {MAX_CALL_INPUT_BYTES}-byte reference budget. Fix: reduce call input size."
            )));
        }
        let value = crate::execution::expr::eval(arg, invocation, memory, program)?;
        value.extend_bytes_width(declared_width, &mut input)?;
    }
    Ok(input)
}

fn param_width(ty: &str) -> usize {
    match ty {
        "u32" | "i32" | "f32" | "vec-count" => 4,
        "u64" | "i64" | "f64" => 8,
        "u8" | "i8" | "bool" => 1,
        _ => 1,
    }
}
