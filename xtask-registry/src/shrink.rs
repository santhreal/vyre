//! `shrink` - reduce every registered corpus case that fails its oracle to a
//! minimal reproducer.
//!
//! The oracle is what the case must survive. By default it is the wire round
//! trip: a program encodes, decodes, and comes back with the same entry node
//! count, because a program the compiler cannot re-read is not a program the
//! compiler can be trusted with. `--oracle PATH` replaces it with a command that
//! is given the wire file and whose exit status is the verdict. A case that fails
//! is delta-debugged by dropping top-level entry nodes while the failure
//! survives, and the finding names the smallest program that still fails.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use vyre::ir::{Node, Program};
use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

/// Delta-debugs the registered corpus cases that fail their oracle.
pub struct Shrink;

impl Gate for Shrink {
    fn name(&self) -> &'static str {
        "shrink"
    }

    fn help(&self) -> &'static str {
        "Delta-debug every registered corpus case that fails its oracle down to a minimal reproducer; --program ID narrows to one, --oracle PATH replaces the oracle"
    }

    fn usage(&self) -> &'static [&'static str] {
        &[
            "--program ID narrows the run to one registered corpus case",
            "--oracle PATH replaces the oracle the reproducer is minimised against",
        ]
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let oracle = match ctx.flag("--oracle") {
            Some(path) => Oracle::Command(PathBuf::from(path)),
            None => Oracle::WireRoundTrip,
        };
        let cases = crate::corpus::selected_cases(ctx.flag("--program"), "reduce")?;

        let work_dir = ctx.root.join("target").join("shrink");
        let mut report = Report::clean();
        report.note(format!("{} corpus case(s) checked", cases.len()));
        let mut failing = 0usize;
        for (id, program) in &cases {
            let Some(reason) = oracle.verdict(&work_dir, id, program)? else {
                continue;
            };
            failing += 1;
            let minimal = reduce(&oracle, &work_dir, id, program)?;
            report.find(Finding::new(
                format!(
                    "corpus case `{id}` fails its oracle: {reason}. The smallest program that still fails has {} of {} top-level node(s)",
                    minimal.entry().len(),
                    program.entry().len()
                ),
                "repair the path this reproducer exercises; the reduced program is written under target/shrink",
            ));
        }
        report.note(format!("{failing} case(s) failed the oracle"));
        Ok(report)
    }
}

/// What a corpus case has to survive.
enum Oracle {
    /// Encode, decode, and come back with the same entry node count.
    WireRoundTrip,
    /// A command given the wire file, whose exit status is the verdict.
    Command(PathBuf),
}

impl Oracle {
    /// `None` when the program survives, `Some(reason)` when it does not.
    fn verdict(
        &self,
        work_dir: &std::path::Path,
        id: &str,
        program: &Program,
    ) -> Result<Option<String>, GateError> {
        let wire = match program.to_wire() {
            Ok(wire) => wire,
            Err(error) => return Ok(Some(format!("the program does not encode: {error}"))),
        };
        match self {
            Self::WireRoundTrip => match Program::from_wire(&wire) {
                Ok(decoded) if decoded.entry().len() == program.entry().len() => Ok(None),
                Ok(decoded) => Ok(Some(format!(
                    "the decoded program has {} top-level node(s), not {}",
                    decoded.entry().len(),
                    program.entry().len()
                ))),
                Err(error) => Ok(Some(format!("the program does not decode: {error}"))),
            },
            Self::Command(command) => {
                let path = write_wire(work_dir, id, &wire)?;
                let status = Command::new(command).arg(&path).status().map_err(|error| {
                    GateError::new(
                        format!("failed to run the oracle `{}`: {error}", command.display()),
                        "pass an executable path after --oracle",
                    )
                })?;
                if status.success() {
                    Ok(None)
                } else {
                    Ok(Some(format!(
                        "the oracle `{}` exited with {}",
                        command.display(),
                        status.code().unwrap_or(-1)
                    )))
                }
            }
        }
    }
}

/// Writes one wire blob for the oracle to read.
fn write_wire(work_dir: &std::path::Path, id: &str, wire: &[u8]) -> Result<PathBuf, GateError> {
    fs::create_dir_all(work_dir).map_err(|error| {
        GateError::new(
            format!("failed to create {}: {error}", work_dir.display()),
            "make the target directory writable, then run the gate again",
        )
    })?;
    let path = work_dir.join(format!("{}.vir", id.replace(['/', ':'], "_")));
    fs::write(&path, wire).map_err(|error| {
        GateError::new(
            format!("failed to write {}: {error}", path.display()),
            "make the target directory writable, then run the gate again",
        )
    })?;
    Ok(path)
}

/// Drops top-level entry nodes for as long as the failure survives, and writes
/// the smallest failing program next to the case it came from.
fn reduce(
    oracle: &Oracle,
    work_dir: &std::path::Path,
    id: &str,
    program: &Program,
) -> Result<Program, GateError> {
    let mut smallest = program.clone();
    let mut index = 0usize;
    while index < smallest.entry().len() {
        let mut entry: Vec<Node> = smallest.entry().to_vec();
        entry.remove(index);
        if entry.is_empty() {
            break;
        }
        let candidate = smallest.with_rewritten_entry(entry);
        if oracle.verdict(work_dir, id, &candidate)?.is_some() {
            smallest = candidate;
            continue;
        }
        index += 1;
    }
    if let Ok(wire) = smallest.to_wire() {
        write_wire(work_dir, &format!("{id}.minimal"), &wire)?;
    }
    Ok(smallest)
}
