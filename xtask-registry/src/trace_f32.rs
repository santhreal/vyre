//! `trace-f32` - run the recorded test inputs of every registered operation
//! through the pure-Rust reference interpreter.
//!
//! The reference is the canonical CPU oracle, so an operation whose own
//! recorded inputs the reference rejects has a program the oracle cannot
//! evaluate, and its expected outputs cannot be trusted. Every rejection is a
//! finding. `--op-id ID` narrows the corpus to one operation and reports the
//! byte-identical `expected_output` literal as notes, which is how a new
//! fixture is written:
//!
//! ```text
//! Some(|| vec![
//!     vec![
//!         vec![0xab, 0xcd, ...],   // run 0, output buffer 0
//!     ],
//! ])
//! ```

use vyre::ir::Program;
use vyre_reference::reference_eval;
use vyre_reference::value::Value;
use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

/// Runs recorded operation fixtures through the reference interpreter.
pub struct TraceF32;

impl Gate for TraceF32 {
    fn name(&self) -> &'static str {
        "trace-f32"
    }

    fn help(&self) -> &'static str {
        "Run the recorded test inputs of every registered operation through the reference; --op-id ID narrows to one"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let selected = ctx.flag("--op-id");
        let corpus = corpus(selected);
        if let Some(op_id) = selected {
            if corpus.is_empty() {
                return Err(GateError::new(
                    format!("op id `{op_id}` is not registered, or is registered without test inputs"),
                    "add `test_inputs` to its canonical registration first; this gate then computes the expected outputs",
                ));
            }
        }
        let mut report = Report::clean();
        let mut traced = 0usize;
        for case in &corpus {
            let mut literal = String::from("Some(|| vec![");
            let mut rejected = false;
            for (run_idx, input_set) in case.inputs.iter().enumerate() {
                let values: Vec<Value> = input_set
                    .iter()
                    .map(|bytes| Value::Bytes(bytes.clone().into()))
                    .collect();
                match reference_eval(&case.program, &values) {
                    Ok(out) => {
                        let outputs: Vec<Vec<u8>> =
                            out.into_iter().map(|value| value.to_bytes()).collect();
                        literal.push_str(&render_run(run_idx, &outputs));
                    }
                    Err(error) => {
                        rejected = true;
                        report.find(Finding::new(
                            format!(
                                "the reference rejected recorded input set {run_idx} of `{}`: {error}",
                                case.id
                            ),
                            "repair the operation program, or replace the offending recorded input; the reference is the oracle the fixture is measured against",
                        ));
                    }
                }
            }
            literal.push_str("\n])");
            if !rejected {
                traced += 1;
                if selected.is_some() {
                    report.note(literal);
                }
            }
        }
        report.note(format!(
            "{traced} of {} recorded fixture(s) evaluated by the reference",
            corpus.len()
        ));
        Ok(report)
    }
}

fn render_run(run_idx: usize, outputs: &[Vec<u8>]) -> String {
    let mut text =
        format!("\n    vec![                                           // run {run_idx}");
    for (buf_idx, bytes) in outputs.iter().enumerate() {
        text.push_str("\n        vec![");
        for (i, byte) in bytes.iter().enumerate() {
            if i > 0 && i % 16 == 0 {
                text.push_str("\n             ");
            }
            text.push_str(&format!("0x{byte:02x}, "));
        }
        text.push_str(&format!(
            "],   // output buffer {buf_idx} ({} bytes)",
            bytes.len()
        ));
    }
    text.push_str("\n    ],");
    text
}

/// One registered operation that carries recorded test inputs.
struct Case {
    id: &'static str,
    program: Program,
    inputs: Vec<Vec<Vec<u8>>>,
}

fn corpus(selected: Option<&str>) -> Vec<Case> {
    let mut cases = Vec::new();
    macro_rules! collect {
        ($catalog:path) => {
            for entry in $catalog() {
                if let Some(op_id) = selected {
                    if entry.id != op_id {
                        continue;
                    }
                }
                let Some(inputs) = entry.test_inputs else {
                    continue;
                };
                let Some(program) = entry.program() else {
                    continue;
                };
                cases.push(Case {
                    id: entry.id,
                    program,
                    inputs: (inputs)(),
                });
            }
        };
    }
    collect!(vyre_libs::operation_catalog::all_entries);
    collect!(vyre_primitives::operation_catalog::all_entries);
    cases
}
