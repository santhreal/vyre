//! Prove `select1_query` computes the same outputs before and after optimization.
//!
//! Both programs run on `vyre-reference`, which is the only interpreter allowed to
//! execute on the CPU, so this is a parity check between two IR shapes rather than
//! a measurement of either one.
use vyre_reference::value::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bits = [0b1011u32, 0x8000_0000, 0xFFFF_0000, 0u32];
    let queries = [1u32, 2, 3, 4, 5];
    let to_bytes = vyre_primitives::wire::pack_u32_slice;
    let inputs = [to_bytes(&bits), to_bytes(&queries), vec![0u8; 5 * 4]];
    let values: Vec<Value> = inputs.into_iter().map(Value::from).collect();

    let program = vyre_primitives::bitset::select::select1_query("bits", "queries", "out", 4, 5);
    let optimized = vyre_foundation::optimizer::optimize(program.clone()).map_err(|error| {
        format!("the registered optimizer did not converge on select1_query: {error}")
    })?;

    let original_outputs = outputs_of(&program, &values, "the unoptimized program")?;
    let optimized_outputs = outputs_of(&optimized, &values, "the optimized program")?;

    println!("Original CPU output: {original_outputs:?}");
    println!("Optimized CPU output: {optimized_outputs:?}");
    if original_outputs != optimized_outputs {
        return Err(
            "optimization changed the outputs of select1_query. Fix: repair the pass that rewrote it, then rerun this example."
                .into(),
        );
    }
    println!("CPU outputs match");
    Ok(())
}

/// Output buffers one program produces on the reference interpreter.
fn outputs_of(
    program: &vyre_foundation::ir::Program,
    values: &[Value],
    which: &str,
) -> Result<Vec<Vec<u8>>, String> {
    let outputs = vyre_reference::reference_eval(program, values)
        .map_err(|error| format!("{which} failed to execute on vyre-reference: {error}"))?;
    Ok(outputs.iter().map(Value::to_bytes).collect())
}
