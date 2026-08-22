//! Whether the benchmark device is held by anything but the recording process.
//!
//! The telemetry probe records what the device was doing during a measurement:
//! clocks, power, memory, utilization. Recording a fact is not the same as
//! refusing to publish against it, and a release baseline measured while another
//! process owned the device is a wrong number with correct provenance. A model
//! server holding 22 of 24 GiB and waking on every request moves a kernel's
//! measured time without moving anything the artifact would let a reader
//! question.
//!
//! Exclusivity is decidable: the driver names every process holding the device
//! for compute. A graphics client such as a display server is not one of them, so
//! a desktop session does not read as contention.

use std::process::Command;

/// One process the driver reports as holding the device for compute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeApp {
    /// The operating system process id.
    pub pid: u32,
    /// The process name the driver reports.
    pub name: String,
    /// Device memory the process holds, when the driver reports it.
    pub used_mib: Option<u64>,
}

impl ComputeApp {
    /// One row, as a sentence a refusal can name the process by.
    #[must_use]
    pub fn describe(&self) -> String {
        match self.used_mib {
            Some(mib) => format!("pid {} `{}` holding {mib} MiB", self.pid, self.name),
            None => format!("pid {} `{}`", self.pid, self.name),
        }
    }
}

/// Every compute process on the device other than `own`.
///
/// A probe failure is a configuration failure: an environment that cannot answer
/// the question cannot prove the device was idle, and a release baseline is not
/// recorded against an unprovable device.
pub fn foreign_compute_apps(own: u32) -> anyhow::Result<Vec<ComputeApp>> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|error| {
            anyhow::anyhow!(
                "nvidia-smi compute-process probe failed: {error}. Fix: repair NVIDIA driver \
                 visibility before recording GPU benchmark evidence."
            )
        })?;
    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "nvidia-smi compute-process probe exited with status {}: {}. Fix: repair NVIDIA \
             driver visibility before recording GPU benchmark evidence.",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8(output.stdout).map_err(|error| {
        anyhow::anyhow!("nvidia-smi compute-process output was not UTF-8: {error}")
    })?;
    Ok(parse_compute_apps(&stdout, own))
}

/// Every row of a compute-process query that names a process other than `own`.
///
/// An unparseable pid is kept rather than dropped: a row the reader cannot
/// attribute is a process it cannot rule out, and dropping it would turn an
/// unreadable device into an idle one.
#[must_use]
pub fn parse_compute_apps(text: &str, own: u32) -> Vec<ComputeApp> {
    let mut apps = Vec::new();
    for line in text.lines() {
        let row = line.trim();
        if row.is_empty() || row.starts_with("No running processes found") {
            continue;
        }
        let mut fields = row.split(',').map(str::trim);
        let pid = fields.next().and_then(|field| field.parse::<u32>().ok());
        if pid == Some(own) {
            continue;
        }
        let name = fields.next().unwrap_or("unknown").to_string();
        let used_mib = fields.next().and_then(|field| field.parse::<u64>().ok());
        apps.push(ComputeApp {
            pid: pid.unwrap_or(0),
            name,
            used_mib,
        });
    }
    apps
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: exclusivity is what the refusal is decided on, so the reader has to
    /// separate this process from every other one, keep a row it cannot
    /// attribute, and read the driver's idle text as idle rather than as a
    /// process named "No running processes found". A reader that drops the
    /// unparseable row, or that counts the idle sentence as a process, reports
    /// the opposite verdict in both directions.
    #[test]
    fn every_row_but_this_process_is_contention() {
        let rows = "\
4242, /usr/bin/llama-server, 21980\n\
4243, /usr/local/bin/ollama, [N/A]\n\
7777, /home/user/target/release/vyre-bench, 512\n\
[N/A], /opt/render/worker, 64\n";

        let foreign = parse_compute_apps(rows, 7777);

        assert_eq!(
            foreign
                .iter()
                .map(|app| (app.pid, app.used_mib))
                .collect::<Vec<_>>(),
            vec![(4242, Some(21980)), (4243, None), (0, Some(64))],
            "Fix: every compute process but this one is contention, and an \
             unattributable row is not idle."
        );
        assert_eq!(
            foreign[0].describe(),
            "pid 4242 `/usr/bin/llama-server` holding 21980 MiB"
        );
        assert!(parse_compute_apps("No running processes found\n", 7777).is_empty());
        assert!(parse_compute_apps("", 7777).is_empty());
        assert!(parse_compute_apps("7777, self, 512\n", 7777).is_empty());
    }
}
