//! The provenance every evidence artifact is recorded under.
//!
//! A number under `release/evidence` is read by someone who no longer has the
//! tree, the host or the device that produced it. Three facts make it
//! attributable and nothing else does: which source tree the generator read,
//! which host it ran on, and which device took part. The corpus used to carry
//! the first of the three as a bare string at the head of the object, so 46 of
//! 88 artifacts carried nothing at all, none of the 88 named a host, and the
//! device facts a performance claim requires had no field to live in.
//!
//! The record is one type, produced in one place, and the writer supplies the
//! two halves it owns. [`TreeRecord`] and [`HostRecord`] are facts of the run:
//! the generator does not get to state them, so it cannot state them wrongly.
//! [`MeasurementRecord`] is the fact only the generator knows, and it is a
//! required argument of [`EvidenceArtifact::new`], so a generator that has not
//! decided whether a device took part has nothing to hand the writer.
//!
//! An artifact whose origin this tree cannot establish is recorded as
//! unattributable, with the reason and the command that recaptures it. That is
//! the honest state of a number of unknown origin. Giving one a plausible
//! commit is the failure this module exists to make impossible.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{LazyLock, Mutex};

use serde::{Deserialize, Serialize};

/// Schema version of the provenance block every evidence artifact carries.
///
/// A reader that finds another version is reading a record written against
/// different rules and must not judge it against these ones.
pub const EVIDENCE_PROVENANCE_SCHEMA_VERSION: u32 = 1;

/// The key an evidence artifact names its provenance under, at the head of the
/// object.
pub const PROVENANCE_KEY: &str = "provenance";

/// What tree, host and device one evidence artifact was recorded from.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EvidenceProvenance {
    /// Version of the rules this record was written against.
    pub schema_version: u32,
    /// The source tree the generator read.
    pub tree: TreeRecord,
    /// The host the generator ran on.
    pub host: HostRecord,
    /// What took part in the measurement, if anything did.
    pub measurement: MeasurementRecord,
}

impl EvidenceProvenance {
    /// The provenance of a record taken from `root` on this host right now.
    ///
    /// The tree walk is memoized on `root`, because every artifact a gate owns
    /// is recorded from one tree in one run and a walk per artifact is what
    /// made four gates spend four `git status` walks proving the same fact.
    ///
    /// # Errors
    ///
    /// Returns the sentence a gate reports when git cannot identify the tree.
    /// A recorder that cannot name its source writes nothing rather than one
    /// more generation of unattributable evidence.
    pub fn capture(root: &Path, measurement: MeasurementRecord) -> Result<Self, String> {
        Ok(Self {
            schema_version: EVIDENCE_PROVENANCE_SCHEMA_VERSION,
            tree: TreeRecord::capture(root)?,
            host: HostRecord::capture(),
            measurement,
        })
    }

    /// The record of an artifact nothing in this tree can attribute.
    ///
    /// `reason` states why the origin is unknown and `recapture` is the exact
    /// command that produces a real record. Both are read by someone deciding
    /// whether a number may be quoted, so neither is optional.
    #[must_use]
    pub fn unattributable(reason: impl Into<String>, recapture: impl Into<String>) -> Self {
        let reason = reason.into();
        let recapture = recapture.into();
        Self {
            schema_version: EVIDENCE_PROVENANCE_SCHEMA_VERSION,
            tree: TreeRecord::Unattributable {
                reason: reason.clone(),
                recapture: recapture.clone(),
            },
            host: HostRecord::Unattributable {
                reason: reason.clone(),
                recapture: recapture.clone(),
            },
            measurement: MeasurementRecord::Unattributable { reason, recapture },
        }
    }

    /// Whether any part of this record states that its origin is unknown.
    #[must_use]
    pub fn is_unattributable(&self) -> bool {
        matches!(self.tree, TreeRecord::Unattributable { .. })
            || matches!(self.host, HostRecord::Unattributable { .. })
            || matches!(self.measurement, MeasurementRecord::Unattributable { .. })
    }

    /// Render the record as the one line an artifact carries it on.
    ///
    /// One line at a known place is what makes lifting the stamp back off
    /// exact. A JSON string cannot hold a literal newline, so the first `,\n`
    /// after the value is always the end of it.
    ///
    /// # Errors
    ///
    /// Returns the serializer's message when the record cannot be represented.
    pub fn render_line(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|error| error.to_string())
    }
}

/// The source tree a generator read.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TreeRecord {
    /// Git identified the tree, and these are the facts it stated.
    Attributed {
        /// Branch checked out when the record was taken.
        branch: String,
        /// Commit the record was taken against.
        commit: String,
        /// Unix seconds of `commit`, as git states them.
        commit_timestamp: String,
        /// First parent of `commit`, empty at a root commit.
        parent_commit: String,
        /// Whether uncommitted non-evidence content took part.
        dirty: bool,
        /// The fingerprint a reader recomputes from the carrying commit.
        source_fingerprint: String,
    },
    /// Nothing in this tree can say which source produced the record.
    Unattributable {
        /// Why the origin is unknown.
        reason: String,
        /// The exact command that produces a real record.
        recapture: String,
    },
}

impl TreeRecord {
    /// The tree at `root`, or the sentence saying why there is none.
    ///
    /// # Errors
    ///
    /// Returns the sentence a gate reports when git names no commit, or cannot
    /// state what the worktree differs from it by.
    pub fn capture(root: &Path) -> Result<Self, String> {
        captured_tree(root)
    }

    /// The commit this record pins, when it pins one.
    #[must_use]
    pub fn commit(&self) -> Option<&str> {
        match self {
            Self::Attributed { commit, .. } => Some(commit),
            Self::Unattributable { .. } => None,
        }
    }

    /// The fingerprint string a stale-source check resolves, when there is one.
    #[must_use]
    pub fn source_fingerprint(&self) -> Option<&str> {
        match self {
            Self::Attributed {
                source_fingerprint, ..
            } => Some(source_fingerprint),
            Self::Unattributable { .. } => None,
        }
    }
}

/// The host a generator ran on.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum HostRecord {
    /// The host identified itself, and these are its facts.
    Recorded {
        /// Name the host answers to.
        hostname: String,
        /// Operating system this build targets.
        os: String,
        /// Instruction set architecture this build targets.
        architecture: String,
        /// Processor model string, as the host reports it.
        cpu_model: String,
        /// Processors the run could use.
        cpu_cores: usize,
    },
    /// The host could not be identified.
    Unattributable {
        /// Why the host is unknown.
        reason: String,
        /// The exact command that produces a real record.
        recapture: String,
    },
}

impl HostRecord {
    /// This host, as it reports itself.
    #[must_use]
    pub fn capture() -> Self {
        let Some(hostname) = hostname() else {
            return Self::Unattributable {
                reason: "this host does not report a name, so a record taken here names no machine"
                    .to_string(),
                recapture: "repair the host name resolution and record the artifact again"
                    .to_string(),
            };
        };
        Self::Recorded {
            hostname,
            os: std::env::consts::OS.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            cpu_model: cpu_model().unwrap_or_else(|| "unreported".to_string()),
            cpu_cores: std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        }
    }
}

/// What took part in producing one record, beyond the host reading source.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum MeasurementRecord {
    /// No device took part. The record is a projection of the source tree.
    ///
    /// This is a positive statement, not an absent device field. A projection
    /// of the registry is reproducible from the commit alone, and a reader
    /// needs to know that rather than to find an empty device list.
    HostOnly,
    /// A device produced part of the record, and these are its facts.
    Device {
        /// Every device the run could see, as the driver reports it.
        devices: Vec<DeviceFacts>,
    },
    /// The record carries numbers whose origin this tree cannot establish.
    Unattributable {
        /// Why the origin is unknown.
        reason: String,
        /// The exact command that recaptures the measurement.
        recapture: String,
    },
}

impl MeasurementRecord {
    /// The devices this host reports, as the class of a measured record.
    ///
    /// A host with no visible device still yields `Device` with an empty list,
    /// which is a finding rather than a silent host-only record: a generator
    /// that claims a device result on a host with no device has recorded a
    /// number the device never produced.
    #[must_use]
    pub fn device() -> Self {
        Self::Device {
            devices: DeviceFacts::probe(),
        }
    }
}

/// One device, by the facts a performance claim has to name.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct DeviceFacts {
    /// Marketing name the driver reports.
    pub name: String,
    /// Driver version the device is running.
    pub driver_version: String,
}

impl DeviceFacts {
    /// Every device this host reports, in the order the driver lists them.
    ///
    /// A host with no driver reports none. That is a fact about the host and
    /// the gate judges it; suppressing it here would turn a device claim on a
    /// deviceless host into a clean record.
    #[must_use]
    pub fn probe() -> Vec<Self> {
        let Ok(output) = Command::new("nvidia-smi")
            .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
            .output()
        else {
            return Vec::new();
        };
        if !output.status.success() {
            return Vec::new();
        }
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(Self::parse_csv_row)
            .collect()
    }

    /// One `name, driver_version` row of the canonical device query.
    #[must_use]
    pub fn parse_csv_row(line: &str) -> Option<Self> {
        let (name, driver_version) = line.split_once(',')?;
        let name = name.trim();
        let driver_version = driver_version.trim();
        if name.is_empty() || driver_version.is_empty() {
            return None;
        }
        Some(Self {
            name: name.to_string(),
            driver_version: driver_version.to_string(),
        })
    }
}

/// One evidence artifact and the provenance class it is recorded under.
///
/// Nothing writes an evidence body on its own. The body and the measurement
/// class leave the generator together, so a generator that has not decided
/// whether a device took part cannot construct one and does not compile.
pub struct EvidenceArtifact<'a, T: Serialize + ?Sized> {
    measurement: MeasurementRecord,
    body: &'a T,
}

impl<'a, T: Serialize + ?Sized> EvidenceArtifact<'a, T> {
    /// Pair one body with what took part in producing it.
    pub fn new(measurement: MeasurementRecord, body: &'a T) -> Self {
        Self { measurement, body }
    }

    /// What the generator states took part in producing this record.
    #[must_use]
    pub fn measurement(&self) -> &MeasurementRecord {
        &self.measurement
    }

    /// Render the body as the bytes the artifact holds under its provenance.
    ///
    /// # Errors
    ///
    /// Returns the serializer's message when the body cannot be represented.
    pub fn render_body(&self) -> Result<String, String> {
        crate::output_arg::render_evidence_json(&self.body)
    }
}

/// A way one recorded provenance block fails to attribute its artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProvenanceIssue {
    /// The artifact carries no provenance block at all.
    Absent,
    /// The head of the artifact is not a provenance block this reader parses.
    Unreadable {
        /// What the parser said.
        reason: String,
    },
    /// The record was written against another version of these rules.
    ForeignSchema {
        /// Version the record declares.
        recorded: u32,
    },
    /// The tree carried uncommitted content when the record was taken.
    Dirty {
        /// Commit the record pins.
        commit: String,
    },
    /// The record pins a commit the current branch does not contain.
    NotAnAncestor {
        /// Commit the record pins.
        commit: String,
    },
    /// The record claims a device result and names no device.
    DeviceResultWithoutDeviceFacts,
    /// The record claims a device result from a host it cannot name.
    DeviceResultWithoutHost,
}

impl ProvenanceIssue {
    /// The half-sentence a caller appends to whatever it is naming.
    #[must_use]
    pub fn predicate(&self) -> String {
        match self {
            Self::Absent => {
                "carries no provenance block, so nothing it records is attributable".to_string()
            }
            Self::Unreadable { reason } => {
                format!("carries a provenance block this reader cannot judge: {reason}")
            }
            Self::ForeignSchema { recorded } => format!(
                "carries provenance schema version {recorded}, and this reader judges version {EVIDENCE_PROVENANCE_SCHEMA_VERSION}"
            ),
            Self::Dirty { commit } => format!(
                "was recorded from a worktree whose changes commit {commit} does not carry, so no checkout reproduces it"
            ),
            Self::NotAnAncestor { commit } => format!(
                "pins commit {commit}, which the current branch does not contain, so it describes a tree this branch never had"
            ),
            Self::DeviceResultWithoutDeviceFacts => {
                "reports a device result and names no device, so its numbers name no hardware"
                    .to_string()
            }
            Self::DeviceResultWithoutHost => {
                "reports a device result from a host it cannot name".to_string()
            }
        }
    }
}

/// Name every way `provenance` fails to attribute the artifact carrying it.
///
/// `contains` answers whether a commit is an ancestor of the branch under
/// judgement. It is a closure so the judgement is provable without a checkout,
/// and so one gate run asks git once per distinct commit rather than once per
/// artifact.
///
/// Every field of every variant is named here, with no rest pattern anywhere.
/// Adding a field to the record or a variant to any of its states stops this
/// compiling until someone decides what the judgement of it is, which is the
/// only way a schema extension cannot land unjudged.
///
/// An unattributable record yields nothing. It is a recorded decision about a
/// number of unknown origin, which is the correct state for one, and the caller
/// reports it as awaiting recapture rather than as a defect to be papered over.
#[must_use]
pub fn issues(
    provenance: &EvidenceProvenance,
    contains: &mut impl FnMut(&str) -> bool,
) -> Vec<ProvenanceIssue> {
    let EvidenceProvenance {
        schema_version,
        tree,
        host,
        measurement,
    } = provenance;
    if *schema_version != EVIDENCE_PROVENANCE_SCHEMA_VERSION {
        return vec![ProvenanceIssue::ForeignSchema {
            recorded: *schema_version,
        }];
    }
    let mut found = Vec::new();
    match tree {
        TreeRecord::Attributed {
            branch: _,
            commit,
            commit_timestamp: _,
            parent_commit: _,
            dirty,
            source_fingerprint: _,
        } => {
            if *dirty {
                found.push(ProvenanceIssue::Dirty {
                    commit: commit.clone(),
                });
            }
            if !contains(commit) {
                found.push(ProvenanceIssue::NotAnAncestor {
                    commit: commit.clone(),
                });
            }
        }
        TreeRecord::Unattributable {
            reason: _,
            recapture: _,
        } => return found,
    }
    let host_named = match host {
        HostRecord::Recorded {
            hostname: _,
            os: _,
            architecture: _,
            cpu_model: _,
            cpu_cores: _,
        } => true,
        HostRecord::Unattributable {
            reason: _,
            recapture: _,
        } => false,
    };
    match measurement {
        MeasurementRecord::HostOnly => {}
        MeasurementRecord::Unattributable {
            reason: _,
            recapture: _,
        } => {}
        MeasurementRecord::Device { devices } => {
            if devices.is_empty() {
                found.push(ProvenanceIssue::DeviceResultWithoutDeviceFacts);
            }
            if !host_named {
                found.push(ProvenanceIssue::DeviceResultWithoutHost);
            }
        }
    }
    found
}

/// The provenance block at the head of `committed`, and the body under it.
///
/// The stamp is one line at a known place, so lifting it back off is exact.
/// The body is what the owning gate generates and is the only half a comparison
/// against the tree may look at: the provenance names the tree the body was
/// recorded from, which is a different tree from the one running the gate
/// whenever anything has been committed since.
#[must_use]
pub fn split(committed: &str) -> (Result<EvidenceProvenance, ProvenanceIssue>, String) {
    let head = format!("{{\n  \"{PROVENANCE_KEY}\": ");
    let Some(rest) = committed.strip_prefix(head.as_str()) else {
        return (Err(ProvenanceIssue::Absent), committed.to_string());
    };
    let Some(end) = rest.find(",\n") else {
        return (Err(ProvenanceIssue::Absent), committed.to_string());
    };
    let record = serde_json::from_str::<EvidenceProvenance>(&rest[..end]).map_err(|error| {
        ProvenanceIssue::Unreadable {
            reason: error.to_string(),
        }
    });
    (record, format!("{{\n{}", &rest[end + ",\n".len()..]))
}

/// Put `provenance` at the head of `body`, or say why it has no head to take one.
///
/// # Errors
///
/// Returns the sentence a caller reports when the record cannot be serialized,
/// or when the rendered artifact is not a JSON object.
pub fn stamp(body: &str, provenance: &EvidenceProvenance) -> Result<String, String> {
    let line = provenance.render_line()?;
    let Some(rest) = body.strip_prefix("{\n") else {
        return Err(
            "recorded evidence must be a JSON object so it can name the tree it came from"
                .to_string(),
        );
    };
    Ok(format!("{{\n  \"{PROVENANCE_KEY}\": {line},\n{rest}"))
}

/// The tree at `root`, memoized so one run asks git once.
fn captured_tree(root: &Path) -> Result<TreeRecord, String> {
    static TREES: LazyLock<Mutex<BTreeMap<PathBuf, Result<TreeRecord, String>>>> =
        LazyLock::new(|| Mutex::new(BTreeMap::new()));
    let mut cache = TREES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(record) = cache.get(root) {
        return record.clone();
    }
    let record = read_tree(root);
    cache.insert(root.to_path_buf(), record.clone());
    record
}

/// Ask git for every tree fact one record names.
fn read_tree(root: &Path) -> Result<TreeRecord, String> {
    let source_fingerprint = crate::source_provenance::capture(root)?;
    let commit = crate::source_provenance::recorded_commit(&source_fingerprint)
        .ok_or_else(|| format!("`{source_fingerprint}` names no commit"))?
        .to_string();
    Ok(TreeRecord::Attributed {
        branch: git_text(root, &["rev-parse", "--abbrev-ref", "HEAD"])?,
        commit_timestamp: git_text(root, &["show", "-s", "--format=%ct", &commit])?,
        parent_commit: git_text(
            root,
            &["rev-parse", "--verify", "--quiet", &format!("{commit}^")],
        )
        .unwrap_or_default(),
        dirty: source_fingerprint.contains(":dirty=true"),
        source_fingerprint,
        commit,
    })
}

/// Run one git command in `root` and return its trimmed output.
fn git_text(root: &Path, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(arguments)
        .output()
        .map_err(|error| format!("`git {}` could not run: {error}", arguments.join(" ")))?;
    if !output.status.success() {
        return Err(format!(
            "`git {}` failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// The name this host answers to.
fn hostname() -> Option<String> {
    if let Ok(text) = std::fs::read_to_string("/etc/hostname") {
        let name = text.trim();
        if !name.is_empty() {
            return Some(name.to_string());
        }
    }
    let output = Command::new("hostname").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!name.is_empty()).then_some(name)
}

/// The processor model this host reports.
fn cpu_model() -> Option<String> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    text.lines()
        .find_map(|line| line.strip_prefix("model name"))
        .and_then(|rest| rest.split_once(':'))
        .map(|(_, model)| model.trim().to_string())
        .filter(|model| !model.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The provenance schema is judged field by field, so a new required field
    /// has to be decided about rather than defaulted into silence.
    fn attributed(commit: &str, dirty: bool) -> TreeRecord {
        TreeRecord::Attributed {
            branch: "integration".to_string(),
            commit: commit.to_string(),
            commit_timestamp: "1787487896".to_string(),
            parent_commit: "0".repeat(40),
            dirty,
            source_fingerprint: format!("git:{commit}:dirty={dirty}"),
        }
    }

    fn host() -> HostRecord {
        HostRecord::Recorded {
            hostname: "santhserver".to_string(),
            os: "linux".to_string(),
            architecture: "x86_64".to_string(),
            cpu_model: "AMD Ryzen".to_string(),
            cpu_cores: 16,
        }
    }

    fn device() -> MeasurementRecord {
        MeasurementRecord::Device {
            devices: vec![DeviceFacts {
                name: "NVIDIA GeForce RTX 3080 Ti".to_string(),
                driver_version: "580.173.02".to_string(),
            }],
        }
    }

    fn record(
        tree: TreeRecord,
        host: HostRecord,
        measurement: MeasurementRecord,
    ) -> EvidenceProvenance {
        EvidenceProvenance {
            schema_version: EVIDENCE_PROVENANCE_SCHEMA_VERSION,
            tree,
            host,
            measurement,
        }
    }

    #[test]
    fn a_complete_device_record_on_an_ancestor_commit_is_clean() {
        let provenance = record(attributed("a".repeat(40).as_str(), false), host(), device());
        assert_eq!(issues(&provenance, &mut |_| true), Vec::new());
    }

    #[test]
    fn a_dirty_record_is_reported_even_on_an_ancestor_commit() {
        let provenance = record(attributed("b".repeat(40).as_str(), true), host(), device());
        assert_eq!(
            issues(&provenance, &mut |_| true),
            vec![ProvenanceIssue::Dirty {
                commit: "b".repeat(40)
            }]
        );
    }

    #[test]
    fn a_commit_the_branch_does_not_contain_is_reported() {
        let provenance = record(attributed("c".repeat(40).as_str(), false), host(), device());
        assert_eq!(
            issues(&provenance, &mut |_| false),
            vec![ProvenanceIssue::NotAnAncestor {
                commit: "c".repeat(40)
            }]
        );
    }

    #[test]
    fn a_device_result_naming_no_device_is_reported() {
        let provenance = record(
            attributed("d".repeat(40).as_str(), false),
            host(),
            MeasurementRecord::Device {
                devices: Vec::new(),
            },
        );
        assert_eq!(
            issues(&provenance, &mut |_| true),
            vec![ProvenanceIssue::DeviceResultWithoutDeviceFacts]
        );
    }

    #[test]
    fn a_device_result_from_an_unnamed_host_is_reported() {
        let provenance = record(
            attributed("e".repeat(40).as_str(), false),
            HostRecord::Unattributable {
                reason: "no name".to_string(),
                recapture: "repair the host".to_string(),
            },
            device(),
        );
        assert_eq!(
            issues(&provenance, &mut |_| true),
            vec![ProvenanceIssue::DeviceResultWithoutHost]
        );
    }

    #[test]
    fn a_host_only_record_states_that_no_device_took_part() {
        let provenance = record(
            attributed("f".repeat(40).as_str(), false),
            host(),
            MeasurementRecord::HostOnly,
        );
        assert_eq!(issues(&provenance, &mut |_| true), Vec::new());
        let rendered = provenance.render_line().expect("render");
        assert!(
            rendered.contains("\"measurement\":{\"state\":\"host_only\"}"),
            "a host-only record states its class rather than carrying an empty device field: {rendered}"
        );
    }

    #[test]
    fn an_unattributable_record_is_a_recorded_decision_and_not_a_finding() {
        let provenance = EvidenceProvenance::unattributable(
            "the recorded numbers predate the provenance schema",
            "./cargo_full run --bin xtask -- release-benchmarks --backend cuda --write",
        );
        assert!(provenance.is_unattributable());
        assert_eq!(issues(&provenance, &mut |_| false), Vec::new());
    }

    #[test]
    fn a_record_written_against_another_schema_version_is_reported() {
        let mut provenance = record(attributed("a".repeat(40).as_str(), false), host(), device());
        provenance.schema_version = EVIDENCE_PROVENANCE_SCHEMA_VERSION + 1;
        assert_eq!(
            issues(&provenance, &mut |_| true),
            vec![ProvenanceIssue::ForeignSchema {
                recorded: EVIDENCE_PROVENANCE_SCHEMA_VERSION + 1
            }]
        );
    }

    #[test]
    fn a_stamped_head_lifts_back_off_exactly() {
        let provenance = record(attributed("a".repeat(40).as_str(), false), host(), device());
        let body = "{\n  \"cases\": [],\n  \"schema_version\": 3\n}\n";
        let stamped = stamp(body, &provenance).expect("stamp");
        let (recovered, recovered_body) = split(&stamped);
        assert_eq!(recovered.expect("provenance"), provenance);
        assert_eq!(recovered_body, body);
    }

    #[test]
    fn a_body_that_is_not_an_object_has_no_head_to_stamp() {
        let provenance = record(attributed("a".repeat(40).as_str(), false), host(), device());
        assert!(stamp("[]\n", &provenance).is_err());
    }

    #[test]
    fn an_artifact_with_no_stamp_reads_as_absent_and_keeps_its_body() {
        let body = "{\n  \"cases\": []\n}\n";
        let (record, recovered) = split(body);
        assert_eq!(record.expect_err("absent"), ProvenanceIssue::Absent);
        assert_eq!(recovered, body);
    }

    #[test]
    fn a_device_row_names_both_facts_or_none() {
        assert_eq!(
            DeviceFacts::parse_csv_row("NVIDIA GeForce RTX 3080 Ti, 580.173.02"),
            Some(DeviceFacts {
                name: "NVIDIA GeForce RTX 3080 Ti".to_string(),
                driver_version: "580.173.02".to_string(),
            })
        );
        assert_eq!(
            DeviceFacts::parse_csv_row("NVIDIA GeForce RTX 3080 Ti"),
            None
        );
        assert_eq!(
            DeviceFacts::parse_csv_row("NVIDIA GeForce RTX 3080 Ti, "),
            None
        );
    }
}
