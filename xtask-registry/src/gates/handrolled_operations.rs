//! `cargo xtask handrolled-operations` - an operation rebuilt instead of composed.
//!
//! A registered operation is a building block. A second operation that needs
//! what it does either composes it, which the IR records as an attributed
//! child region, or rebuilds its body inline. The second spelling produces the
//! same result today and drifts tomorrow: the copy does not move when the
//! original is fixed, the optimizer sees two shapes where the registry
//! declares one, and the composition depth every LEGO law measures reads as
//! own work.
//!
//! This gate counts the second spelling and is pinned at zero. It is not
//! pinned at today's count, because a handroll is never the correct state of
//! the tree: the fix is always available, and it is to compose the operation
//! that already exists.
//!
//! ## What it reads
//!
//! Built IR at run time, never source text. A handroll written with different
//! variable names, a different loop bound spelling or a different buffer name
//! is the same handroll, and only the built program says so.
//!
//! The instrument is the fingerprint [`lego_audit`](crate::gates::lego_audit)
//! already owns and `whats-similar` already scores. There is no second
//! fingerprint here, and that matters for more than deduplication: the
//! fingerprint encodes attribution directly. An attributed region collapses to
//! a hash of the child operation's name and its body is never inlined, while
//! an unattributed region inlines the body it holds. So a byte run equal to
//! another operation's whole program can appear inside a host only where the
//! host did not attribute it, and the containment relation needs no separate
//! attribution join.
//!
//! ## What it does not catch
//!
//! A partial copy. The relation reports containment of a whole registered
//! program, so an operation that rebuilds two thirds of another one is not a
//! finding here; `whats-similar` scores that pair by similarity instead.
//!
//! An operation whose fingerprint is shorter than
//! `MIN_COMPARABLE_FINGERPRINT_BYTES`. A byte run that short describes one or
//! two nodes, and finding it inside a larger program is coincidence more often
//! than reuse. That floor is the one `whats-similar` applies to the same
//! fingerprints, read from its owner rather than chosen again here.
//!
//! Two operations that are the same program and nothing else. Equal
//! fingerprints are a duplicate rather than a handroll, and `whats-similar`
//! reports them at score 1.0.

use std::collections::{BTreeSet, HashMap};

use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

use crate::gates::lego_audit::{collect_ops, MIN_COMPARABLE_FINGERPRINT_BYTES};

/// One registered operation reduced to what the containment relation reads.
///
/// The relation is a pure function of identity and fingerprint, so it can be
/// driven from a synthetic set as well as from the live registry. That is what
/// lets the contract be proven red by injection rather than only observed
/// green against whatever the tree currently holds.
pub struct FingerprintedOperation<'a> {
    /// Registered operation id.
    pub id: &'a str,
    /// The operation's whole-program structural fingerprint.
    pub fingerprint: &'a [u8],
}

/// One operation holding another operation's whole program inline.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Handroll<'a> {
    /// The operation that holds the copy.
    pub host: &'a str,
    /// The registered operation whose whole program the host rebuilt.
    pub rebuilt: &'a str,
    /// Byte offset into the host fingerprint where the copy starts.
    pub offset: usize,
}

/// Bytes of a fingerprint used to index candidate matches.
///
/// The relation is quadratic in operations and linear in fingerprint bytes if
/// written as a scan of every pair, which is minutes on a registry of this
/// size. Indexing every candidate by its opening bytes makes it one pass over
/// each host with a bucket lookup per offset. Eight is below
/// [`MIN_COMPARABLE_FINGERPRINT_BYTES`], so every candidate has a key.
const KEY_BYTES: usize = 8;

/// Every operation in `ops` that holds another one's whole program inline.
///
/// Reported once per (host, rebuilt) pair at the lowest offset the copy starts
/// at, ordered by host then by rebuilt, so two runs over the same registry
/// produce the same list.
#[must_use]
pub fn handrolls<'a>(ops: &[FingerprintedOperation<'a>]) -> Vec<Handroll<'a>> {
    let mut by_opening: HashMap<&[u8], Vec<usize>> = HashMap::new();
    for (index, op) in ops.iter().enumerate() {
        if op.fingerprint.len() < MIN_COMPARABLE_FINGERPRINT_BYTES {
            continue;
        }
        by_opening
            .entry(&op.fingerprint[..KEY_BYTES])
            .or_default()
            .push(index);
    }

    let mut found = Vec::new();
    for host in ops {
        if host.fingerprint.len() < KEY_BYTES {
            continue;
        }
        let mut reported: BTreeSet<&str> = BTreeSet::new();
        for offset in 0..=host.fingerprint.len() - KEY_BYTES {
            let Some(candidates) = by_opening.get(&host.fingerprint[offset..offset + KEY_BYTES])
            else {
                continue;
            };
            for &index in candidates {
                let candidate = &ops[index];
                if candidate.id == host.id
                    || candidate.fingerprint.len() >= host.fingerprint.len()
                {
                    continue;
                }
                let end = offset + candidate.fingerprint.len();
                if end > host.fingerprint.len()
                    || &host.fingerprint[offset..end] != candidate.fingerprint
                {
                    continue;
                }
                if reported.insert(candidate.id) {
                    found.push(Handroll {
                        host: host.id,
                        rebuilt: candidate.id,
                        offset,
                    });
                }
            }
        }
    }
    found.sort();
    found
}

/// Reports registered operations that rebuild another registered operation.
pub struct HandrolledOperations;

impl Gate for HandrolledOperations {
    fn name(&self) -> &'static str {
        "handrolled-operations"
    }

    fn help(&self) -> &'static str {
        "Report registered operations that hold another registered operation's whole program inline instead of composing it"
    }

    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        if ops.is_empty() {
            return Err(GateError::new(
                "the live operation registry produced no buildable program, so the containment walk judged nothing",
                "link the crates that submit operations through vyre-registry-link, then run the gate again; a walk over an empty registry reporting zero handrolls is the failure this refuses",
            ));
        }

        let subjects: Vec<FingerprintedOperation<'_>> = ops
            .iter()
            .map(|op| FingerprintedOperation {
                id: op.id.as_str(),
                fingerprint: &op.fingerprint,
            })
            .collect();
        let comparable = subjects
            .iter()
            .filter(|op| op.fingerprint.len() >= MIN_COMPARABLE_FINGERPRINT_BYTES)
            .count();

        for handroll in handrolls(&subjects) {
            report.find(Finding::new(
                format!(
                    "registered operation `{}` holds the whole program of `{}` inline at fingerprint byte {}, with no attribution to it",
                    handroll.host, handroll.rebuilt, handroll.offset
                ),
                format!(
                    "replace the inlined body with a child region naming `{}`, so the selection is an edge to the registered building block rather than a second copy of it",
                    handroll.rebuilt
                ),
            ));
        }

        report.note(format!(
            "handrolled-operations: {} registered operations built, {comparable} above the {MIN_COMPARABLE_FINGERPRINT_BYTES}-byte comparison floor",
            subjects.len()
        ));
        Ok(report)
    }
}
