//! What the host driver reports about the devices attached to it.
//!
//! Two evidence writers asked the same driver tool the same way and each
//! decided separately what an absent tool, a nonzero exit and an unparseable
//! row mean. Those are one policy: a host with no driver reports no device,
//! and that is a fact about the host rather than an error. Suppressing it
//! anywhere else would turn a device claim on a deviceless host into a clean
//! record, so the decision is made once, here.

use std::process::Command;

/// The driver tool a host exposes its devices through.
const DRIVER_TOOL: &str = "nvidia-smi";

/// What the driver tool answered, or `None` when it did not answer.
///
/// A tool that is not installed and a tool that exited nonzero are the same
/// fact to every caller: this host stated nothing about its devices.
#[must_use]
pub fn query(arguments: &[&str]) -> Option<String> {
    let output = Command::new(DRIVER_TOOL).args(arguments).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// One record per row of a driver answer that `parse` accepts.
///
/// A row the parser rejects is dropped rather than reported. The query fixes
/// the column set, so a row that does not match it came from a driver speaking
/// a different dialect, and inventing a device from it would be worse than
/// listing one fewer.
pub fn rows<T>(answer: Option<&str>, parse: impl Fn(&str) -> Option<T>) -> Vec<T> {
    answer
        .unwrap_or_default()
        .lines()
        .filter_map(parse)
        .collect()
}

/// One record per device the driver lists in answer to `arguments`.
pub fn query_rows<T>(arguments: &[&str], parse: impl Fn(&str) -> Option<T>) -> Vec<T> {
    rows(query(arguments).as_deref(), parse)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: gates run on hosts with no driver, and every device record in this
    /// workspace is built from these rows. A silent host must report no device
    /// rather than one empty device, because an empty device is a claim that a
    /// device was there. A row the parser rejects must disappear for the same
    /// reason: the query fixes the columns, so a row that does not match them
    /// describes something other than the device that was asked about.
    #[test]
    fn nothing_a_parser_rejects_ever_becomes_a_device() {
        let named = |line: &str| {
            let name = line.trim();
            (!name.is_empty() && !name.starts_with('[')).then(|| name.to_string())
        };

        assert!(
            rows(None, named).is_empty(),
            "a host whose driver tool did not answer reports no device"
        );
        assert!(
            rows(Some(""), named).is_empty(),
            "an empty answer is no device, not one unnamed device"
        );
        assert!(
            rows(Some("\n\n"), named).is_empty(),
            "a blank row is no device"
        );
        assert_eq!(
            rows(
                Some("alpha\n[N/A]\nbeta\n"),
                named
            ),
            ["alpha".to_string(), "beta".to_string()],
            "a row in a dialect the parser does not read is dropped, and the rest are kept in order"
        );
    }
}
