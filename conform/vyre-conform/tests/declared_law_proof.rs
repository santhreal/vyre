//! Every algebraic law a registration declares is either proved by executing
//! the operation, or recorded as unproven with the payload its statement needs.
//!
//! The registry is the variant space: these tests walk every linked
//! registration rather than a list of operation names, so a new operation
//! declaring a law is judged the moment it registers. A declaration the oracle
//! refutes fails here; a declaration whose statement cannot be exercised fails
//! unless `law-proof-decisions.toml` records why.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use vyre_conform::law_proof::{prove_declared_laws, prove_law, LawVerdict, UnprovenKind};
use vyre_foundation::operation::SemanticOperation;

/// One recorded decision about a law that carries no executable proof.
struct Decision {
    kind: UnprovenKind,
    reason: String,
}

fn decisions_path() -> PathBuf {
    structure_gate::workspace_root().join("conform/vyre-conform/law-proof-decisions.toml")
}

/// The recorded decisions, keyed by `(op id, law)`, and the declared cap.
fn decisions() -> (BTreeMap<(String, String), Decision>, usize) {
    let path = decisions_path();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "Fix: {} must exist and be readable; every law without an executable proof is recorded there ({error})",
            path.display()
        )
    });
    // `Value::from_str` parses a TOML value expression, not a document, so a
    // whole file handed to it fails at line 1 column 1 no matter how valid it is.
    let document: toml::Table = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("Fix: {} must be valid TOML ({error})", path.display()));
    let cap = document
        .get("unproven_cap")
        .and_then(toml::Value::as_integer)
        .and_then(|cap| usize::try_from(cap).ok())
        .unwrap_or_else(|| {
            panic!(
                "Fix: {} must declare `unproven_cap` as a non-negative integer",
                path.display()
            )
        });
    let mut rows = BTreeMap::new();
    let declared = document
        .get("decision")
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| {
            panic!(
                "Fix: {} must declare a `[[decision]]` array",
                path.display()
            )
        });
    for row in declared {
        let op_id = row
            .get("op")
            .and_then(toml::Value::as_str)
            .unwrap_or_else(|| panic!("Fix: every decision row needs a nonblank `op`"))
            .to_string();
        let law = row
            .get("law")
            .and_then(toml::Value::as_str)
            .unwrap_or_else(|| panic!("Fix: every decision row needs a nonblank `law`"))
            .to_string();
        let kind_name = row
            .get("kind")
            .and_then(toml::Value::as_str)
            .unwrap_or_else(|| panic!("Fix: decision row `{op_id}` / `{law}` needs a `kind`"));
        let kind = UnprovenKind::parse(kind_name).unwrap_or_else(|| {
            let known = UnprovenKind::ALL
                .iter()
                .map(|kind| kind.name())
                .collect::<Vec<_>>()
                .join(", ");
            panic!(
                "Fix: decision row `{op_id}` / `{law}` names kind `{kind_name}`; the vocabulary is {known}"
            )
        });
        let reason = row
            .get("reason")
            .and_then(toml::Value::as_str)
            .map(str::trim)
            .filter(|reason| !reason.is_empty())
            .unwrap_or_else(|| {
                panic!("Fix: decision row `{op_id}` / `{law}` needs a nonblank `reason`")
            })
            .to_string();
        assert!(
            rows.insert((op_id.clone(), law.clone()), Decision { kind, reason })
                .is_none(),
            "Fix: decision row `{op_id}` / `{law}` is declared twice; keep one decision per pair"
        );
    }
    (rows, cap)
}

/// Every linked registration, read through the reader that asserts every
/// submitting crate reached the registry.
fn registered() -> Vec<SemanticOperation> {
    vyre_registry_link::operation::live_operation_registry()
        .iter()
        .collect()
}

/// WHY: a declared law used to be a string nothing executed. Every declaration
/// is now proved against the reference oracle or recorded as unproven, and this
/// walks the live registry so a new operation cannot add an unjudged law.
#[test]
fn every_declared_law_is_proved_or_recorded_as_unproven() {
    let (recorded, cap) = decisions();
    let mut refuted = Vec::new();
    let mut unrunnable = Vec::new();
    let mut unproven = BTreeSet::new();
    let mut missing = Vec::new();
    let mut proved = 0usize;
    for entry in registered() {
        for proof in prove_declared_laws(&entry) {
            let key = (proof.op_id.to_string(), proof.law.to_string());
            match &proof.verdict {
                LawVerdict::Holds { cases } => {
                    assert!(
                        *cases > 0,
                        "Fix: `{}` law `{}` reports holding over zero fixture cases",
                        proof.op_id,
                        proof.law
                    );
                    proved += 1;
                    assert!(
                        !recorded.contains_key(&key),
                        "Fix: `{}` law `{}` is proved by witness {:?} and still carries a decision row; delete the row and lower `unproven_cap`",
                        proof.op_id,
                        proof.law,
                        proof.witness
                    );
                }
                LawVerdict::Refuted { case, detail } => refuted.push(format!(
                    "`{}` law `{}` refuted on fixture case {case}: {detail}",
                    proof.op_id, proof.law
                )),
                LawVerdict::Unrunnable { reason } => unrunnable.push(format!(
                    "`{}` law `{}` could not be run: {reason}",
                    proof.op_id, proof.law
                )),
                LawVerdict::Unproven { kind, reason } => {
                    unproven.insert(key.clone());
                    match recorded.get(&key) {
                        None => missing.push(format!(
                            "`{}` law `{}` carries no executable proof, kind `{}` ({reason})",
                            proof.op_id,
                            proof.law,
                            kind.name()
                        )),
                        Some(decision) => {
                            assert_eq!(
                                decision.kind,
                                *kind,
                                "Fix: decision row `{}` / `{}` records kind `{}` and the prover reports `{}` ({}): correct the row",
                                proof.op_id,
                                proof.law,
                                decision.kind.name(),
                                kind.name(),
                                decision.reason
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(
        unrunnable.is_empty(),
        "Fix: {} declared law(s) could not be executed at all; repair the operation or its fixtures:\n  {}",
        unrunnable.len(),
        unrunnable.join("\n  ")
    );
    assert!(
        refuted.is_empty(),
        "Fix: the reference oracle refutes {} declared law(s); the declaration is wrong or needs a guard:\n  {}",
        refuted.len(),
        refuted.join("\n  ")
    );
    assert!(
        missing.is_empty(),
        "Fix: record each of these {} pair(s) in conform/vyre-conform/law-proof-decisions.toml, or make the law provable:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
    let stale: Vec<String> = recorded
        .keys()
        .filter(|key| !unproven.contains(*key))
        .map(|(op, law)| format!("`{op}` law `{law}`"))
        .collect();
    assert!(
        stale.is_empty(),
        "Fix: delete {} decision row(s) naming a pair the registry no longer declares unproven: {}",
        stale.len(),
        stale.join(", ")
    );
    assert_eq!(
        unproven.len(),
        cap,
        "Fix: `unproven_cap` in law-proof-decisions.toml must equal the number of unproven declarations, so recording a law's payload lowers it"
    );
    assert!(
        proved > 0,
        "Fix: no declared law was proved; the prover reaches no registration"
    );
}

/// WHY: a prover that cannot refute proves nothing. `bitset::and_not` computes
/// `lhs & !rhs` over two same-shaped inputs, so exchanging its operands must
/// change the result; if this passes, every commutativity verdict above is
/// vacuous.
#[test]
fn an_asymmetric_operation_is_refuted_when_asked_for_commutativity() {
    let entry = registered()
        .into_iter()
        .find(|entry| entry.id == "vyre-libs::bitset::and_not")
        .expect("Fix: `vyre-libs::bitset::and_not` must stay registered; it is the prover's own control case");
    assert!(
        !entry.laws.contains(&"commutative"),
        "Fix: `and_not` must not declare commutativity; it is the control case that has to be refutable"
    );

    let proof = prove_law(&entry, "commutative");

    assert!(
        proof.is_refuted(),
        "Fix: the prover must refute commutativity of `lhs & !rhs`; verdict was {:?}",
        proof.verdict
    );
}

/// WHY: every proved law reports how many fixture cases carried its witness,
/// and a count of zero would read as a proof while executing nothing. Selection
/// across cases is proved in `vyre_conform::law_proof`, against a declared
/// shape rather than against whichever fixture set a library op ships; this
/// holds the registry to the other half: a verdict of `Holds` names at least
/// one case that ran.
///
/// What this does not catch: a witness that is a weaker statement than the law.
/// The refutation control below covers that direction.
#[test]
fn every_proved_law_names_at_least_one_case_it_ran() {
    let mut proved = 0usize;
    for entry in registered() {
        for proof in prove_declared_laws(&entry) {
            let LawVerdict::Holds { cases } = proof.verdict else {
                continue;
            };
            assert!(
                cases > 0,
                "Fix: `{}` law `{}` reports as proved over zero fixture cases; a proof that ran nothing is a label",
                proof.op_id,
                proof.law
            );
            proved += 1;
        }
    }
    assert!(
        proved > 0,
        "Fix: no declared law was proved; the prover reaches no registration"
    );
}

/// WHY: the roster of decisions is only as strong as the vocabulary it is
/// judged against. Every kind must round-trip through its recorded name, so a
/// kind added to the enum cannot be spelled two ways.
#[test]
fn every_unproven_kind_round_trips_through_its_name() {
    for kind in UnprovenKind::ALL {
        assert_eq!(
            UnprovenKind::parse(kind.name()),
            Some(*kind),
            "Fix: kind `{}` must parse back from its own name",
            kind.name()
        );
    }
    let names: BTreeSet<&str> = UnprovenKind::ALL.iter().map(|kind| kind.name()).collect();
    assert_eq!(
        names.len(),
        UnprovenKind::ALL.len(),
        "Fix: two unproven kinds share a name"
    );
}

/// WHY: an operation declaring no law must produce no proof row, or the roster
/// above would count rows nobody declared.
#[test]
fn an_operation_declaring_no_law_produces_no_proof() {
    let entry = registered()
        .into_iter()
        .find(|entry| entry.laws.is_empty())
        .expect("Fix: the registry must carry at least one operation with no declared law");

    assert!(
        prove_declared_laws(&entry).is_empty(),
        "Fix: `{}` declares no law and must produce no proof row",
        entry.id
    );
}

/// WHY: the variant space is `vyre_spec::law_catalog()`, read at run time. A law
/// added there is either proved by execution or classified by the payload its
/// statement needs; a law in neither set reaches no arm and every declaration of
/// it would report a kind chosen by a fallback.
#[test]
fn every_law_in_the_catalog_is_either_provable_or_classified() {
    let provable: BTreeSet<&str> = vyre_conform::law_proof::PROVABLE_LAWS
        .iter()
        .copied()
        .collect();
    let mut unclassified = Vec::new();
    let mut both = Vec::new();
    for law in vyre_spec::law_catalog() {
        let classified = UnprovenKind::for_law(law).is_some();
        match (provable.contains(law), classified) {
            (false, false) => unclassified.push(*law),
            (true, true) => both.push(*law),
            _ => {}
        }
    }
    assert!(
        unclassified.is_empty(),
        "Fix: law(s) {unclassified:?} are neither in PROVABLE_LAWS nor classified by UnprovenKind::for_law; add a witness or name the payload the statement needs"
    );
    assert!(
        both.is_empty(),
        "Fix: law(s) {both:?} are both provable and classified as unproven; a provable law needs no payload row"
    );
    for law in vyre_conform::law_proof::PROVABLE_LAWS {
        assert!(
            vyre_spec::law_catalog().contains(law),
            "Fix: PROVABLE_LAWS names `{law}`, which the frozen law catalog does not declare"
        );
    }
}
