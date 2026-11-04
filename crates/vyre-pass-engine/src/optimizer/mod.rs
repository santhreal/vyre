//! Optimizer passes executed through the semantic compile-and-execute boundary.
//!
//! Each analysis kernel is lifted into a validated single-node graph. The pass
//! engine supplies typed input bytes and applies the returned semantic result to
//! the IR; compiler schedule selection controls launch organization.

mod arena_cursor;
mod arena_kernel;
pub mod canonicalize_via_encoded;
pub mod combined_decode;
pub mod const_fold_via_encoded;
// `cse_via_encoded` is the public path for cross-scope CSE and for the analysis
// Programs it dispatches. These two submodules exist because that file was
// split, so they stay private and the owner re-exports what they hold; a second
// public path is a second name for one item.
mod cse_cross_scope;
mod cse_programs;
pub mod cse_via_encoded;
pub mod dce_program;
pub mod dce_via_encoded;
pub mod encode;
pub mod expr_arena;
pub mod pattern_match_via_encoded;
pub mod pipeline;
mod rewrite_walk;
pub mod validate_via_encoded;

fn build_encoded_analysis_program(
    expr_count: u32,
    output_name: &str,
    per_expr_body: Vec<vyre_foundation::ir::Node>,
) -> vyre_foundation::ir::Program {
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
    let count = expr_count.max(1);
    let mut buffers = arena_kernel::arena_row_buffers(expr_count, 0);
    buffers.push(BufferDecl::output(output_name, 4, DataType::U32).with_count(count));
    let body = vec![
        Node::let_bind("i", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("i"), Expr::u32(expr_count)),
            per_expr_body,
        ),
    ];
    Program::wrapped(buffers, [256, 1, 1], body)
}

fn execute_retained_program(
    executor: &dyn vyre_megakernel::SemanticExecutor,
    policy: &vyre_megakernel::SemanticExecutionPolicy,
    stage: &str,
    program: vyre_foundation::ir::Program,
    input_bytes: &[Vec<u8>],
) -> Result<Vec<Vec<u8>>, vyre_megakernel::SemanticExecutionError> {
    use std::collections::BTreeMap;
    use vyre_foundation::ir::{
        BufferAccess, GraphInput, ProgramGraph, ShapeDim, ValueContract, ValueLifetime,
    };
    use vyre_foundation::logical::LogicalProgramGraph;
    use vyre_megakernel::{
        SemanticExecutionError, SemanticExecutionOutput, SemanticExecutionRequest,
    };

    let mut graph = ProgramGraph::new();
    let mut ports = Vec::new();
    let mut request_inputs = BTreeMap::new();
    let mut byte_iter = input_bytes.iter();
    for buffer in program
        .buffers()
        .iter()
        .filter(|buffer| buffer.access() != BufferAccess::Workgroup)
    {
        let lifetime = if buffer.access() == BufferAccess::ReadWrite {
            ValueLifetime::Retained
        } else {
            ValueLifetime::Invocation
        };
        let contract = ValueContract {
            dtype: buffer.element(),
            shape: vec![ShapeDim::Known(u64::from(buffer.count()))],
            access: buffer.access(),
            lifetime,
        };
        let value = graph
            .add_external_value(format!("{stage}.{}.input", buffer.name()), contract.clone())
            .map_err(|error| {
                SemanticExecutionError::InvalidRequest(format!(
                    "{stage} graph input construction failed: {error}"
                ))
            })?;
        let bytes = byte_iter.next().ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(format!(
                "{stage} requires more canonical input buffers"
            ))
        })?;
        request_inputs.insert(value, bytes.as_slice());
        ports.push((buffer.name().to_string(), value, contract));
    }
    if byte_iter.next().is_some() {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "{stage} received extra canonical input buffers"
        )));
    }

    let inputs = ports
        .iter()
        .map(|(buffer, value, contract)| GraphInput {
            buffer: buffer.clone(),
            value: *value,
            contract: contract.clone(),
        })
        .collect::<Vec<_>>();
    let output_order = ports
        .iter()
        .filter_map(|(_, value, contract)| {
            (contract.access == BufferAccess::ReadWrite).then_some(*value)
        })
        .collect::<Vec<_>>();
    graph
        .add_node(stage, program, inputs, Vec::new())
        .map_err(|error| {
            SemanticExecutionError::InvalidRequest(format!(
                "{stage} graph construction failed: {error}"
            ))
        })?;
    let logical = LogicalProgramGraph::validate(&graph, &policy.external_facts().symbolic_bindings)
        .map_err(|error| {
            SemanticExecutionError::InvalidRequest(format!(
                "{stage} logical graph validation failed: {error}"
            ))
        })?;
    let request = SemanticExecutionRequest::new(&logical, request_inputs, policy.clone())?;
    let SemanticExecutionOutput { mut outputs, .. } = executor.execute(&request)?;
    let mut ordered = Vec::with_capacity(output_order.len());
    for value in output_order {
        ordered.push(outputs.remove(&value).ok_or_else(|| {
            SemanticExecutionError::Backend(format!(
                "{stage} executor omitted canonical output value {}",
                value.0
            ))
        })?);
    }
    if !outputs.is_empty() {
        return Err(SemanticExecutionError::Backend(format!(
            "{stage} executor returned {} undeclared output values",
            outputs.len()
        )));
    }
    Ok(ordered)
}

fn run_encoded_analysis_kernel(
    arena: &expr_arena::ExprArenaEncoding,
    executor: &dyn vyre_megakernel::SemanticExecutor,
    policy: &vyre_megakernel::SemanticExecutionPolicy,
    inputs: &mut Vec<Vec<u8>>,
    output: &mut Vec<u32>,
    build_program: fn(u32) -> vyre_foundation::ir::Program,
    stage: &str,
    output_name: &str,
) -> Result<(), vyre_megakernel::SemanticExecutionError> {
    use vyre_libs::dispatch_buffers::{
        decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes,
    };
    let count = arena.expr_count;
    let words = count as usize;
    ensure_input_slots(inputs, 4);
    write_u32_slice_le_bytes(&mut inputs[0], &arena.kinds);
    write_u32_slice_le_bytes(&mut inputs[1], &arena.arg0);
    write_u32_slice_le_bytes(&mut inputs[2], &arena.arg1);
    write_u32_slice_le_bytes(&mut inputs[3], &arena.arg2);

    let mut execution = vyre_megakernel::execute_single_program(
        executor,
        stage,
        build_program(count),
        inputs,
        policy,
    )?;
    if execution.outputs.len() != 1 {
        return Err(vyre_megakernel::SemanticExecutionError::Backend(format!(
            "{stage} semantic execution expected exactly one {output_name} output, got {}. Fix: return the canonical graph output",
            execution.outputs.len()
        )));
    }
    decode_u32_output_exact(
        &execution.outputs.remove(0),
        words,
        &format!("{stage} {output_name}"),
        output,
    )
    .map_err(|error| {
        vyre_megakernel::SemanticExecutionError::Backend(format!(
            "{stage} semantic output decoding failed: {error}"
        ))
    })
}
