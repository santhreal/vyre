//! Metal is proved on an Apple GPU, and the proof names every counter the
//! driver publishes.
//!
//! This replaces a 240-line shell script that ran three remote suites and then
//! grepped sixteen counter names out of the JSON they produced. The list in the
//! script was a copy of the driver's counter table, and it had already drifted:
//! the driver publishes seventeen names, the script demanded sixteen, and the
//! one it did not demand is the error bucket that fires when the resident
//! buffer table is poisoned. A copied list is the failure this gate exists to
//! prevent, so the roster is read from the driver at run time and both halves
//! of the gate use the same set.
//!
//! The cheap half runs anywhere: it holds the published roster to the assertions
//! in the driver's own tests and holds the measured cases to the benchmark
//! catalog. The `--host` half runs the suites on the Apple machine over ssh and
//! reads the emitted report, and it fails closed when the host cannot answer,
//! because a Metal verdict from a machine with no Metal device is the defect the
//! script was written for.

use std::collections::BTreeSet;
use std::process::Command;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Crate that owns the Metal backend and publishes its counters.
const METAL_CRATE: &str = "vyre-driver-metal";

/// File that assembles the metric snapshot, relative to the driver crate.
const SNAPSHOT_SOURCE: &str = "src/runtime.rs";

/// Crate whose suite proves conformance against the reference.
const CONFORM_CRATE: &str = "vyre-conform";

/// Feature the conformance suite needs to reach a device.
const CONFORM_FEATURE: &str = "gpu";

/// Crate that measures a backend and writes the report this gate reads.
const BENCH_CRATE: &str = "vyre-bench";

/// Cases the remote run measures.
///
/// The first is the elementwise smoke case every backend carries; the second
/// exercises the resident buffer table, which is the only path that publishes
/// the resident counters.
const MEASURED_CASES: [&str; 2] = [
    "foundation.elementwise.add.1m",
    "dataflow.ifds.skewed.queue_closure.1m",
];

/// Seconds an ssh connection may take to establish.
const CONNECT_TIMEOUT: &str = "8";

/// Metal parity is proved on an Apple GPU, or not at all.
pub struct MetalParity;

impl Gate for MetalParity {
    fn name(&self) -> &'static str {
        "metal-parity"
    }

    fn help(&self) -> &'static str {
        "Hold the Metal counter roster to the driver's own assertions; --host <target> runs the suites on an Apple GPU"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();

        let metal_dir = tree.member_directory(METAL_CRATE)?;
        let snapshot_source = format!("{metal_dir}/{SNAPSHOT_SOURCE}");
        let snapshot = tree.read(&snapshot_source)?;
        let published = published_counters(&snapshot);
        if published.is_empty() {
            report.find(Finding::in_file(
                snapshot_source.as_str(),
                "no Metal counter is published by the metric snapshot, so a Metal report carries no telemetry to judge",
                "publish the counters through METAL_COUNTERS and the resident pushes; a backend that reports nothing cannot be compared",
            ));
        }

        let asserted = asserted_counters(&tree, &metal_dir, &published)?;
        for counter in published.difference(&asserted) {
            report.find(Finding::in_file(
                snapshot_source.as_str(),
                format!(
                    "`{counter}` is published by the Metal snapshot and named by no test under `{METAL_CRATE}`"
                ),
                "assert the counter in the driver's telemetry suite; a counter nothing reads is a name that rots between releases",
            ));
        }

        let bench_dir = tree.member_directory(BENCH_CRATE)?;
        let bench_source = format!("{bench_dir}/src");
        let mut catalog = String::new();
        for path in tree.rust(&[bench_source.as_str()])? {
            catalog.push_str(&tree.read(path)?);
        }
        for case in MEASURED_CASES {
            if !catalog.contains(case) {
                report.find(Finding::in_file(
                    bench_source.as_str(),
                    format!("the Metal run measures `{case}`, which no benchmark case defines"),
                    "measure a case the catalog carries, or restore the case; a run against an unknown case id measures nothing and still exits zero",
                ));
            }
        }

        let conform_dir = tree.member_directory(CONFORM_CRATE)?;
        let conform_manifest = format!("{conform_dir}/Cargo.toml");
        let declares_gpu = tree
            .read_toml(&conform_manifest)?
            .get("features")
            .and_then(toml::Value::as_table)
            .is_some_and(|table| table.contains_key(CONFORM_FEATURE));
        if !declares_gpu {
            report.find(Finding::in_file(
                conform_manifest,
                format!(
                    "`{CONFORM_CRATE}` declares no `{CONFORM_FEATURE}` feature, so the remote conformance run reaches no device"
                ),
                format!("declare the `{CONFORM_FEATURE}` feature the device run selects"),
            ));
        }

        let Some(host) = ctx.flag("--host") else {
            report.note(format!(
                "{} published Metal counter(s), {} measured case(s)",
                published.len(),
                MEASURED_CASES.len()
            ));
            return Ok(report);
        };

        let root = ctx.flag("--remote-root").ok_or_else(|| {
            GateError::new(
                "--host was given without --remote-root",
                "pass --remote-root <path to the vyre checkout on the Apple host>; the run happens in that checkout",
            )
        })?;
        report.note(format!("host: {host}"));
        remote(&host, &root, &format!("\"$runner\" test -p {METAL_CRATE}"))?;
        remote(
            &host,
            &root,
            &format!(
                "VYRE_BACKEND=metal \"$runner\" test -p {CONFORM_CRATE} --features {CONFORM_FEATURE}"
            ),
        )?;
        for case in MEASURED_CASES {
            let emitted = remote(&host, &root, &measure(case))?;
            let keys = report_keys(&emitted).ok_or_else(|| {
                GateError::new(
                    format!("the Metal report for `{case}` is not a JSON object"),
                    "run the benchmark by hand on the Apple host and read what it wrote; a report this gate cannot parse proves nothing",
                )
            })?;
            for counter in &published {
                if !keys.contains(counter.as_str()) {
                    report.find(Finding::new(
                        format!(
                            "the Metal report for `{case}` carries no `{counter}`, which the driver publishes"
                        ),
                        "carry every published counter into the report; a counter dropped in the report is a regression nothing measures",
                    ));
                }
            }
        }
        Ok(report)
    }
}

/// Counter names the metric snapshot publishes, read from the driver.
///
/// One spelling reaches the snapshot in two places: a row in the counter table,
/// and a direct push for the resident buffer table, which is read under a lock
/// and reports an error bucket when that lock is poisoned. Both are a quoted
/// name opening a tuple, so one read covers them.
fn published_counters(source: &str) -> BTreeSet<String> {
    const MARKER: &str = "(\"metal_";
    let mut found = BTreeSet::new();
    let mut rest = source;
    while let Some(at) = rest.find(MARKER) {
        let name = &rest[at + 2..];
        let Some(end) = name.find('"') else {
            break;
        };
        found.insert(name[..end].to_string());
        rest = &name[end + 1..];
    }
    found
}

/// Counters named by a test source in the driver crate.
fn asserted_counters(
    tree: &Tree,
    metal_dir: &str,
    published: &BTreeSet<String>,
) -> Result<BTreeSet<String>, GateError> {
    let mut found = BTreeSet::new();
    for path in tree.rust(&[metal_dir])? {
        let name = path.to_string_lossy();
        if !name.contains("/tests/") && !name.ends_with("/tests.rs") {
            continue;
        }
        let text = tree.read(&path)?;
        for counter in published {
            if text.contains(counter.as_str()) {
                found.insert(counter.clone());
            }
        }
    }
    Ok(found)
}

/// Remote script that measures one case and prints the report it wrote.
fn measure(case: &str) -> String {
    format!(
        "out=\"$(mktemp -d)\"; \
         VYRE_ALLOW_FEW_SAMPLES=1 \"$runner\" run -q -p {BENCH_CRATE} --bin {BENCH_CRATE} -- run \
         --suite smoke --format json --backend metal --case {case} \
         --warmup-samples 0 --measured-samples 3 --sample-timeout-secs 60 \
         --determinism-runs 1 --output \"$out/metal.json\" >/dev/null; \
         cat \"$out/metal.json\"; rm -rf \"$out\""
    )
}

/// Run one command in the checkout on the Apple host.
fn remote(host: &str, root: &str, command: &str) -> Result<String, GateError> {
    let script = format!(
        "set -euo pipefail; cd {}; \
         runner=./cargo_full; [ -x \"$runner\" ] || runner=cargo; {command}",
        quoted(root)
    );
    let output = Command::new("ssh")
        .args([
            "-o",
            "BatchMode=yes",
            "-o",
            &format!("ConnectTimeout={CONNECT_TIMEOUT}"),
            destination(host)?,
            &script,
        ])
        .output()
        .map_err(|error| {
            GateError::new(
                format!("ssh to {host} could not be started: {error}"),
                "install an ssh client and configure the Apple host; this gate proves nothing without it",
            )
        })?;
    if !output.status.success() {
        return Err(GateError::new(
            format!(
                "`{command}` failed on {host}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            "fix the failure on the Apple host and run the gate again; a Metal verdict is only worth what the device run says",
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// One value, quoted for the remote `/bin/sh`.
///
/// The remote script is a string a shell parses, so a checkout path carrying a
/// space, a semicolon or a substitution was read as syntax rather than as a
/// path. Single quotes take everything but a single quote, which is closed,
/// escaped and reopened.
fn quoted(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// The ssh destination, or the reason it cannot be one.
///
/// ssh reads a leading `-` as an option, and `-oProxyCommand=...` runs a command
/// on THIS machine rather than connecting anywhere, so a destination is checked
/// before it reaches the argument list. Quoting cannot help here: the value is
/// an argv entry, not shell text.
fn destination(host: &str) -> Result<&str, GateError> {
    if host.is_empty() || host.starts_with('-') {
        return Err(GateError::new(
            format!("`{host}` is not an ssh destination"),
            "pass --host <user@machine>; a value opening with `-` is read by ssh as an option, and one of those options runs a command locally",
        ));
    }
    Ok(host)
}

/// Every key a benchmark report carries, at any depth.
fn report_keys(text: &str) -> Option<BTreeSet<String>> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    if !value.is_object() {
        return None;
    }
    let mut found = BTreeSet::new();
    collect_keys(&value, &mut found);
    Some(found)
}

fn collect_keys(value: &serde_json::Value, found: &mut BTreeSet<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, nested) in map {
                found.insert(key.clone());
                collect_keys(nested, found);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_keys(item, found);
            }
        }
        _ => {}
    }
}

/// `published_counters`, `report_keys`, `quoted` and `destination` read text the
/// gate never writes to disk, so no integration test can reach them through the
/// CLI.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_counter_row_and_a_direct_push_are_both_published() {
        let source = "const METAL_COUNTERS: [(&str, fn(&M) -> &A); 2] = [\n    (\"metal_pipeline_cache_hits\", |m| &m.hits),\n    (\"metal_output_readback_bytes\", |m| &m.readback),\n];\nmetrics.push((\"metal_resident_buffer_error\", 1_u64));\n";
        let found = published_counters(source);
        assert_eq!(
            found.iter().map(String::as_str).collect::<Vec<_>>(),
            vec![
                "metal_output_readback_bytes",
                "metal_pipeline_cache_hits",
                "metal_resident_buffer_error"
            ]
        );
    }

    #[test]
    fn a_report_key_is_found_at_any_depth() {
        let keys = report_keys("{\"cases\":[{\"metrics\":{\"metal_resident_bytes\":4}}]}")
            .expect("Fix: the fixture report is a JSON object");
        assert!(keys.contains("metal_resident_bytes"));
        assert!(report_keys("[]").is_none());
        assert!(report_keys("not json").is_none());
    }

    /// WHY: the remote root was interpolated into a `/bin/sh` script, so a path
    /// carrying a separator ran whatever followed it on the Apple host.
    #[test]
    fn a_root_carrying_shell_syntax_is_one_word() {
        assert_eq!(quoted("/Users/ci/vyre"), "'/Users/ci/vyre'");
        assert_eq!(
            quoted("/tmp/x; rm -rf ~"),
            "'/tmp/x; rm -rf ~'",
            "a separator inside single quotes is text"
        );
        assert_eq!(quoted("/tmp/it's"), "'/tmp/it'\\''s'");
    }

    /// WHY: `-oProxyCommand=...` is an ssh option that runs a command locally,
    /// so a destination is judged before it becomes an argv entry.
    #[test]
    fn a_destination_that_opens_an_option_is_refused() {
        assert_eq!(destination("ci@apple").expect("a plain host"), "ci@apple");
        assert!(destination("-oProxyCommand=touch /tmp/pwned").is_err());
        assert!(destination("").is_err());
    }
}
