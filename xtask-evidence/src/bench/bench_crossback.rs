//! Hold the cross-backend comparison tables to their pinned shape.
//!
//! The table records CPU-reference oracle timing only. GPU release performance
//! evidence comes from the dedicated CUDA and WGPU benchmark suites, never from
//! fabricated cross-backend numbers.
//!
//! Timing is measured, so the committed number is not regenerated and compared:
//! two runs of the same tree disagree in the third decimal and every run would
//! report a divergence that is only the clock. What is regenerated and compared
//! is everything the tree determines. The banner, the column header, the row
//! set and the program names must match byte for byte, and each row must carry
//! a parseable positive millisecond value. `--write` re-measures and records
//! the table.

use std::path::PathBuf;
use std::time::Instant;

use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};
use xtask::output_arg::read_text_bounded;

/// Every program the harness knows, with the byte count it runs over.
const PROGRAMS: &[(&str, usize)] = &[("xor-1k", 1024), ("xor-1m", 1024 * 1024)];

/// A table is small. Anything larger than this is not one of ours.
const MAX_TABLE_BYTES: u64 = 262_144;

/// Iterations each timing is amortised over, to reduce clock jitter.
const TIMING_ITERATIONS: usize = 100;

const BANNER: &str = "# cross-backend comparison\n\n\
    Produced by `cargo xtask bench-crossback`. ms values are CPU-reference\n\
    oracle wall-clock per call. GPU release evidence comes from the dedicated\n\
    CUDA and WGPU benchmark suites.\n\n";

const COLUMNS: &str = "| program | wgpu | spirv | secondary_text | native_module | cpu-ref |\n\
    |---------|------|-------|----------------|---------------|---------|\n";

pub struct BenchCrossbackGate;

impl Gate for BenchCrossbackGate {
    fn name(&self) -> &'static str {
        "bench-crossback"
    }

    fn help(&self) -> &'static str {
        "Check every committed cross-backend comparison table against the program list, and \
         re-measure it under --write. `--program NAME` narrows to one program."
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let selected = match select_programs(ctx) {
            Ok(selected) => selected,
            Err(finding) => {
                report.find(finding);
                return Ok(report);
            }
        };
        if std::env::var("VYRE_BENCH_GPU").ok().as_deref() == Some("1") {
            report.find(Finding::new(
                "VYRE_BENCH_GPU=1 is set, and this gate has no measured GPU path to answer it \
                 with",
                "Unset VYRE_BENCH_GPU and run the release CUDA or WGPU benchmark suite for GPU \
                 evidence. This table records CPU-reference timing only.",
            ));
        }
        for (name, size) in selected {
            let relative = table_path(name);
            let absolute = ctx.root.join(&relative);
            if ctx.write {
                let table = render_table(&[measure(name, *size)]);
                xtask::output_arg::create_parent_dir(&absolute);
                std::fs::write(&absolute, &table).map_err(|error| {
                    GateError::new(
                        format!("failed to write `{}`: {error}", absolute.display()),
                        "Check that the docs/perf directory is writable.",
                    )
                })?;
                report.note(format!("wrote {relative}"));
                continue;
            }
            audit_table(&relative, &absolute, name, &mut report);
        }
        Ok(report)
    }
}

/// The programs this invocation covers, or the finding that names the mistake.
///
/// An unknown `--program` used to exit 1 with a message on stderr. It is a
/// finding now, so the sweep counts it like every other one.
fn select_programs(ctx: &GateCtx) -> Result<Vec<&'static (&'static str, usize)>, Finding> {
    let Some(requested) = ctx.flag("--program") else {
        return Ok(PROGRAMS.iter().collect());
    };
    let matched: Vec<_> = PROGRAMS
        .iter()
        .filter(|(name, _)| *name == requested)
        .collect();
    if matched.is_empty() {
        let known: Vec<&str> = PROGRAMS.iter().map(|(name, _)| *name).collect();
        return Err(Finding::new(
            format!("`--program {requested}` names no known program"),
            format!(
                "Pass one of: {}. Omit --program to cover every one.",
                known.join(", ")
            ),
        ));
    }
    Ok(matched)
}

fn table_path(program: &str) -> String {
    format!("docs/perf/cross-backend-{program}.md")
}

/// Read the committed table and hold every tree-determined part of it to shape.
fn audit_table(relative: &str, absolute: &PathBuf, program: &str, report: &mut Report) {
    let text = match read_text_bounded(absolute, MAX_TABLE_BYTES, "cross-backend table") {
        Ok(text) => text,
        Err(error) => {
            report.find(Finding::in_file(
                PathBuf::from(relative),
                format!("cross-backend table for `{program}` is missing or unreadable: {error}"),
                "Run `cargo xtask bench-crossback --write` on a host you are willing to quote \
                 timing from, and commit the table.",
            ));
            return;
        }
    };
    let text = text.replace("\r\n", "\n");
    let expected_prefix = format!("{BANNER}{COLUMNS}");
    if !text.starts_with(&expected_prefix) {
        report.find(Finding::in_file(
            PathBuf::from(relative),
            "cross-backend table does not open with the pinned banner and column header"
                .to_string(),
            "Run `cargo xtask bench-crossback --write` to rewrite it. The header is pinned so a \
             diff between two runs shows a measurement change and nothing else.",
        ));
        return;
    }
    let rows: Vec<&str> = text[expected_prefix.len()..]
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if rows.len() != 1 {
        report.find(Finding::in_file(
            PathBuf::from(relative),
            format!(
                "cross-backend table for `{program}` holds {} rows, and one program records one \
                 row",
                rows.len()
            ),
            "Run `cargo xtask bench-crossback --write` to rewrite it.",
        ));
        return;
    }
    audit_row(relative, program, rows[0], report);
}

/// One row: the program it names, the four unmeasured cells and the timing.
fn audit_row(relative: &str, program: &str, row: &str, report: &mut Report) {
    let cells: Vec<&str> = row.split('|').map(str::trim).collect();
    // A pipe-delimited row splits into an empty leading and trailing cell.
    if cells.len() != 8 {
        report.find(Finding::at(
            PathBuf::from(relative),
            row_line(),
            format!(
                "cross-backend row holds {} cells, and the pinned table has six",
                cells.len().saturating_sub(2)
            ),
            "Run `cargo xtask bench-crossback --write` to rewrite it.",
        ));
        return;
    }
    let named = cells[1].trim_matches('`');
    if named != program {
        report.find(Finding::at(
            PathBuf::from(relative),
            row_line(),
            format!("cross-backend table for `{program}` records a row for `{named}`"),
            "Run `cargo xtask bench-crossback --write` to rewrite it. The file name and the row \
             name the same program.",
        ));
    }
    for (column, cell) in ["wgpu", "spirv", "secondary_text", "native_module"]
        .iter()
        .zip(&cells[2..6])
    {
        if *cell != "n/a" {
            report.find(Finding::at(
                PathBuf::from(relative),
                row_line(),
                format!(
                    "cross-backend row records `{cell}` for column `{column}`, and this harness \
                     measures no backend timing"
                ),
                "Delete the value. GPU timing comes from the release CUDA or WGPU benchmark \
                 suite, and a number recorded here is a number nobody measured.",
            ));
        }
    }
    match cells[6].parse::<f64>() {
        Ok(value) if value > 0.0 && value.is_finite() => {
            report.note(format!("{program} cpu-ref {value:.3} ms per call (recorded)"));
        }
        Ok(value) => report.find(Finding::at(
            PathBuf::from(relative),
            row_line(),
            format!("cross-backend row records a cpu-ref of `{value}` ms"),
            "Re-measure with `cargo xtask bench-crossback --write`. A run takes longer than zero \
             and finishes.",
        )),
        Err(error) => report.find(Finding::at(
            PathBuf::from(relative),
            row_line(),
            format!("cross-backend cpu-ref cell `{}` is not a number: {error}", cells[6]),
            "Re-measure with `cargo xtask bench-crossback --write`.",
        )),
    }
}

/// The pinned table puts the single data row directly under the two header
/// rows, which sit under the four-line banner and its blank separators.
const fn row_line() -> u32 {
    9
}

struct Row {
    program: &'static str,
    cpu_ref_ms: f64,
}

fn measure(program: &'static str, size: usize) -> Row {
    Row {
        program,
        cpu_ref_ms: time_cpu_ref_xor(size),
    }
}

fn render_table(rows: &[Row]) -> String {
    let mut out = String::with_capacity(BANNER.len() + COLUMNS.len() + rows.len() * 64);
    out.push_str(BANNER);
    out.push_str(COLUMNS);
    for row in rows {
        out.push_str(&format!(
            "| `{}` | n/a | n/a | n/a | n/a | {:.3} |\n",
            row.program, row.cpu_ref_ms
        ));
    }
    out
}

/// Reference XOR over bytes, in milliseconds per call.
fn time_cpu_ref_xor(size: usize) -> f64 {
    let input = vec![0u8; size];
    let mut output = vec![0u8; size];
    let start = Instant::now();
    for _ in 0..TIMING_ITERATIONS {
        for (target, source) in output.iter_mut().zip(&input) {
            *target = source ^ 0xA5;
        }
    }
    let elapsed = start.elapsed();
    (elapsed.as_secs_f64() * 1000.0) / (TIMING_ITERATIONS as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(args: &[&str]) -> GateCtx {
        GateCtx::new(
            std::path::PathBuf::from("."),
            args.iter().map(|arg| (*arg).to_string()).collect(),
        )
    }

    /// WHY: `--program` narrowing is the only caller input this gate takes, and
    /// a typo used to exit 1 from inside a child process whose stdout is the
    /// report protocol. It has to arrive as a finding.
    #[test]
    fn an_unknown_program_is_a_finding_and_names_the_known_ones() {
        let error = select_programs(&ctx(&["--program", "xor-9k"])).unwrap_err();
        assert!(error.message.contains("xor-9k"), "{}", error.message);
        assert!(error.fix.contains("xor-1k"), "{}", error.fix);
        assert!(error.fix.contains("xor-1m"), "{}", error.fix);
    }

    #[test]
    fn no_program_flag_covers_every_program() {
        let selected = select_programs(&ctx(&[])).unwrap();
        assert_eq!(selected.len(), PROGRAMS.len());
    }

    #[test]
    fn a_program_flag_narrows_to_one() {
        let selected = select_programs(&ctx(&["--program", "xor-1m"])).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].0, "xor-1m");
    }

    /// WHY: the gate compares the committed table against this renderer, so a
    /// table this renderer produces must audit clean. Anything else means the
    /// two halves disagree and every run reports a divergence that is nobody's
    /// defect.
    #[test]
    fn a_freshly_rendered_table_audits_clean() {
        let table = render_table(&[Row {
            program: "xor-1k",
            cpu_ref_ms: 0.125,
        }]);
        let mut report = Report::clean();
        let prefix = format!("{BANNER}{COLUMNS}");
        assert!(table.starts_with(&prefix), "{table}");
        audit_row(
            "docs/perf/cross-backend-xor-1k.md",
            "xor-1k",
            table[prefix.len()..].trim_end(),
            &mut report,
        );
        assert_eq!(report.findings, Vec::new());
    }

    /// WHY: this is the assertion the old harness carried in prose and never
    /// checked. A fabricated GPU number in a table that says it measures none
    /// is the exact defect the module warns about.
    #[test]
    fn a_fabricated_backend_timing_is_a_finding() {
        let mut report = Report::clean();
        audit_row(
            "docs/perf/cross-backend-xor-1k.md",
            "xor-1k",
            "| `xor-1k` | 0.400 | n/a | n/a | n/a | 0.125 |",
            &mut report,
        );
        assert_eq!(report.findings.len(), 1);
        assert!(
            report.findings[0].message.contains("wgpu"),
            "{}",
            report.findings[0].message
        );
    }

    #[test]
    fn a_row_naming_another_program_is_a_finding() {
        let mut report = Report::clean();
        audit_row(
            "docs/perf/cross-backend-xor-1k.md",
            "xor-1k",
            "| `xor-1m` | n/a | n/a | n/a | n/a | 0.125 |",
            &mut report,
        );
        assert_eq!(report.findings.len(), 1);
        assert!(
            report.findings[0].message.contains("xor-1m"),
            "{}",
            report.findings[0].message
        );
    }

    #[test]
    fn an_unparseable_or_zero_timing_is_a_finding() {
        for cell in ["", "0", "-1", "fast"] {
            let mut report = Report::clean();
            audit_row(
                "docs/perf/cross-backend-xor-1k.md",
                "xor-1k",
                &format!("| `xor-1k` | n/a | n/a | n/a | n/a | {cell} |"),
                &mut report,
            );
            assert_eq!(report.findings.len(), 1, "cell `{cell}` reported nothing");
        }
    }
}
