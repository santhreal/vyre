//! The `domain-vocabulary` gate: consumer-domain vocabulary stays out of published compiler,
//! backend, and runtime surfaces.
//!
//! Enforces that:
//! 1. No item published by a crate outside the composition and verification layers names a
//!    consumer-domain concept. One model family's vocabulary in a lower layer's public API makes
//!    every other consumer a special case of the family that arrived first, and the mechanism the
//!    name stood for stops being nameable on its own.
//! 2. Every term in the contract states the mechanism to publish instead, so a finding names the
//!    replacement and not only the offence.
//!
//! Scope is derived, not listed. Every layer in `docs/CRATE_OWNERSHIP.toml` is checked unless
//! `docs/DOMAIN_VOCABULARY.toml` records it exempt with a reason, so a layer added later is in
//! scope until a decision for it is recorded. A crate named for the container format or domain it
//! adapts is exempt from the terms its own package name contains, derived from that name rather
//! than from a per-crate allowlist.
//!
//! The published surface is the checked surface. Prose, test material, and internal items are out
//! of scope: a comment stating which mechanism replaced a domain name is the documentation this
//! contract wants, and the snapshots under `docs/public-api/` are exactly what a consumer can
//! name.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use toml::Value;

use crate::gate::{Finding, GateBehavior, GateCtx, GateError, Report};
use crate::gates::crate_registry;
use crate::gates::scan::Tree;

/// The vocabulary contract this gate reads instead of holding terms of its own.
const CONTRACT: &str = "docs/DOMAIN_VOCABULARY.toml";

/// Schema version this gate decodes.
const SCHEMA_VERSION: i64 = 1;

/// Directory holding one published-API snapshot per publishing crate.
const SNAPSHOTS: &str = "docs/public-api";

/// Gate that keeps consumer-domain vocabulary out of published non-composition surfaces.
pub struct DomainVocabulary;

/// One term of the contract: the identifier segments naming a domain concept, and the mechanism to
/// publish in its place.
struct Term {
    /// Lowercase identifier segments, matched as a contiguous run.
    segments: Vec<String>,
    /// What to publish instead, stated as the finding's corrective action.
    mechanism: String,
}

impl Term {
    /// The term as it reads in a finding.
    fn name(&self) -> String {
        self.segments.join("_")
    }

    /// Whether split identifier segments contain this term as a contiguous run.
    fn occurs_in(&self, segments: &[String]) -> bool {
        segments
            .windows(self.segments.len())
            .any(|window| window == self.segments.as_slice())
    }
}

impl GateBehavior for DomainVocabulary {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let tree = Tree::open(&ctx.root)?;
        let contract = tree.read_toml(CONTRACT)?;

        if contract.get("schema_version").and_then(Value::as_integer) != Some(SCHEMA_VERSION) {
            report.find(Finding::in_file(
                CONTRACT,
                format!("the contract does not declare schema_version = {SCHEMA_VERSION}"),
                "declare the schema version this gate decodes",
            ));
            return Ok(report);
        }

        let exempt_layers = decode_exempt_layers(&contract, &mut report);
        let terms = decode_terms(&contract, &mut report);
        if terms.is_empty() {
            report.find(Finding::in_file(
                CONTRACT,
                "the contract declares no [[term]] rows, so every published name passes",
                "record the domain terms that stay out of published non-composition surfaces",
            ));
            return Ok(report);
        }

        let crates = crate_registry::load_registry(&tree, &mut report)?;
        let mut published_items = 0usize;
        let mut checked_crates = 0usize;

        for record in &crates {
            if exempt_layers.contains(&record.layer) {
                continue;
            }
            let snapshot = PathBuf::from(SNAPSHOTS).join(format!("{}.txt", record.package));
            let Ok(text) = std::fs::read_to_string(ctx.root.join(&snapshot)) else {
                continue;
            };
            checked_crates += 1;
            published_items += inspect_snapshot(
                &snapshot,
                &record.package,
                &record.layer,
                &text,
                &terms,
                &mut report,
            );
            inspect_mechanism_owners(&snapshot, &text, &mut report);
        }

        if checked_crates == 0 {
            report.find(Finding::in_file(
                CONTRACT,
                "no published snapshot was checked, so the contract proves nothing",
                "keep a snapshot under docs/public-api for every publishing crate",
            ));
            return Ok(report);
        }

        report.cover_complete(
            format!("published items across {checked_crates} non-composition crates"),
            published_items,
        );
        Ok(report)
    }
}

/// Check one crate's published snapshot against every term, returning the items inspected.
///
/// The self-name exemption is computed from `package`: a crate named for what it adapts may
/// publish the terms its own name contains, and no others.
fn inspect_snapshot(
    snapshot: &Path,
    package: &str,
    layer: &str,
    text: &str,
    terms: &[Term],
    report: &mut Report,
) -> usize {
    let self_named: BTreeSet<String> = identifier_segments(package).into_iter().collect();
    let mut inspected = 0usize;

    for (line_index, line) in text.lines().enumerate() {
        let item = line.trim();
        if item.is_empty() {
            continue;
        }
        inspected += 1;
        let segments = identifier_segments(item);
        for term in terms {
            if term.segments.iter().any(|part| self_named.contains(part)) {
                continue;
            }
            if !term.occurs_in(&segments) {
                continue;
            }
            report.find(Finding::in_file(
                snapshot.to_path_buf(),
                format!(
                    "line {}: the `{layer}` layer publishes the consumer-domain term `{}` in `{item}`",
                    line_index + 1,
                    term.name(),
                ),
                term.mechanism.clone(),
            ));
        }
    }
    inspected
}

/// Report a mechanism one crate publishes an owner for from two different modules.
///
/// A type name is the mechanism's name. Two modules declaring it means a consumer reads two
/// facts under one name and no import says which owns the mechanism, which is the shape a
/// model-specific queue beside the generic one takes: `expert_scheduling::WorkQueue` beside
/// `resident_work_queue::WorkQueue` publishes one name over two schedulers.
///
/// The owner set is derived from the snapshot, so the rule needs no roster of mechanisms.
/// Associated items are excluded: a parent segment starting uppercase names a type rather than a
/// module, so `substrate::Query::Output` is one trait's associated type and not a second owner of
/// `Output`.
fn inspect_mechanism_owners(snapshot: &Path, text: &str, report: &mut Report) {
    let mut owners: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for line in text.lines() {
        let Some(path) = declared_type_path(line.trim()) else {
            continue;
        };
        let segments: Vec<&str> = path.split("::").collect();
        let Some((name, modules)) = segments.split_last() else {
            continue;
        };
        // The first segment is the crate, and the rest must all be modules.
        let Some((_, modules)) = modules.split_first() else {
            continue;
        };
        if modules
            .iter()
            .any(|segment| !segment.starts_with(|first: char| first.is_ascii_lowercase()))
        {
            continue;
        }
        let module = if modules.is_empty() {
            "the crate root".to_string()
        } else {
            modules.join("::")
        };
        owners.entry(name.to_string()).or_default().insert(module);
    }

    for (mechanism, modules) in owners {
        if modules.len() < 2 {
            continue;
        }
        let named: Vec<&str> = modules.iter().map(String::as_str).collect();
        report.find(Finding::in_file(
            snapshot.to_path_buf(),
            format!(
                "`{mechanism}` is published by {} modules: {}",
                modules.len(),
                named.join(", ")
            ),
            "keep one owner for the mechanism and have the other module consume it, or name the \
             two mechanisms apart",
        ));
    }
}

/// The published path of a type declaration, or `None` for any other published item.
///
/// Only a nominal type declares a mechanism. A function, constant, or impl line names one that
/// some module already declared, so counting it would report the owner twice.
fn declared_type_path(item: &str) -> Option<&str> {
    let declaration = item.strip_prefix("#[non_exhaustive] ").unwrap_or(item);
    let rest = ["pub struct ", "pub enum ", "pub trait ", "pub union "]
        .into_iter()
        .find_map(|keyword| declaration.strip_prefix(keyword))?;
    let path = rest
        .split_once('<')
        .map_or(rest, |(head, _)| head)
        .split_whitespace()
        .next()?
        .trim_end_matches(&[';', '{', ','][..]);
    path.contains("::").then_some(path)
}

/// Decode the layers the contract records as composition or verification layers.
fn decode_exempt_layers(contract: &toml::Table, report: &mut Report) -> BTreeSet<String> {
    let mut exempt = BTreeSet::new();
    let Some(rows) = contract.get("exempt_layer").and_then(Value::as_array) else {
        return exempt;
    };
    for row in rows {
        let name = row.get("name").and_then(Value::as_str);
        let reason = row
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match name {
            Some(name) if !reason.trim().is_empty() => {
                exempt.insert(name.to_string());
            }
            Some(name) => report.find(Finding::in_file(
                CONTRACT,
                format!("exempt layer `{name}` records no reason"),
                "state why the layer composes or verifies domain behavior",
            )),
            None => report.find(Finding::in_file(
                CONTRACT,
                "an [[exempt_layer]] row declares no name",
                "name the layer the row exempts",
            )),
        }
    }
    exempt
}

/// Decode the contract's terms, reporting a row that names no segments or states no mechanism.
fn decode_terms(contract: &toml::Table, report: &mut Report) -> Vec<Term> {
    let mut terms = Vec::new();
    let Some(rows) = contract.get("term").and_then(Value::as_array) else {
        return terms;
    };
    for row in rows {
        let segments: Vec<String> = row
            .get("segments")
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_lowercase)
                    .collect()
            })
            .unwrap_or_default();
        let mechanism = row
            .get("mechanism")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();

        if segments.is_empty() {
            report.find(Finding::in_file(
                CONTRACT,
                "a [[term]] row declares no segments",
                "state the identifier segments that name the domain concept",
            ));
            continue;
        }
        if mechanism.is_empty() {
            report.find(Finding::in_file(
                CONTRACT,
                format!(
                    "term `{}` states no mechanism to publish instead",
                    segments.join("_")
                ),
                "state the mechanism the term stands for, so a finding names the replacement",
            ));
            continue;
        }
        terms.push(Term {
            segments,
            mechanism: mechanism.to_string(),
        });
    }
    terms
}

/// Split text into lowercase identifier segments, breaking on non-alphanumeric characters and on
/// camel-case boundaries, so `ExpertQueue`, `expert_queue`, and `route-expert` all yield
/// `expert`. An acronym run ends where the next word begins, so `KVCache` yields `kv` and
/// `cache` rather than one unmatchable segment.
fn identifier_segments(text: &str) -> Vec<String> {
    let characters: Vec<char> = text.chars().collect();
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut previous_lower_or_digit = false;

    for (index, &character) in characters.iter().enumerate() {
        if !character.is_ascii_alphanumeric() {
            if !current.is_empty() {
                segments.push(std::mem::take(&mut current));
            }
            previous_lower_or_digit = false;
            continue;
        }
        if character.is_ascii_uppercase() && !current.is_empty() {
            let starts_a_word = characters
                .get(index + 1)
                .is_some_and(char::is_ascii_lowercase);
            if previous_lower_or_digit || starts_a_word {
                segments.push(std::mem::take(&mut current));
            }
        }
        current.push(character.to_ascii_lowercase());
        previous_lower_or_digit = character.is_ascii_lowercase() || character.is_ascii_digit();
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract as the workspace ships it, so a test measures the shipped terms rather than a
    /// copy that goes stale beside them.
    fn shipped_contract() -> toml::Table {
        let path = crate::checkout::checkout_root().join(CONTRACT);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        toml::from_str(&text).expect("the shipped contract parses")
    }

    /// The shipped terms, proven to decode without a finding.
    fn shipped_terms() -> Vec<Term> {
        let mut report = Report::clean();
        let terms = decode_terms(&shipped_contract(), &mut report);
        assert_eq!(
            report.count(),
            0,
            "the shipped contract has malformed term rows: {}",
            report.finding_messages()
        );
        assert!(!terms.is_empty(), "the shipped contract declares no terms");
        terms
    }

    /// One published name carrying a model-family term is reported, with the mechanism as the fix.
    ///
    /// This is the reintroduction the gate exists for: the runtime once published
    /// `expert_scheduling`, and deleting the module is what a later change undoes.
    #[test]
    fn a_routed_shard_renamed_for_one_model_family_is_a_finding() {
        let mut report = Report::clean();
        let inspected = inspect_snapshot(
            Path::new("docs/public-api/vyre-runtime.txt"),
            "vyre-runtime",
            "runtime",
            "pub fn vyre_runtime::expert_scheduling::ExpertQueue::submit(&self)\n",
            &shipped_terms(),
            &mut report,
        );
        assert_eq!(inspected, 1);
        assert_eq!(report.count(), 1, "{}", report.finding_messages());
        let finding = &report.findings[0];
        assert!(finding.message.contains("`expert`"), "{}", finding.message);
        assert!(finding.message.contains("runtime"), "{}", finding.message);
        assert!(
            finding.fix.contains("routed shard"),
            "the finding states no replacement: {}",
            finding.fix
        );
    }

    /// A model-specific scheduler published beside the generic one is a second owner.
    ///
    /// This is the other half of the reintroduction: a term the contract does not list still
    /// duplicates ownership when the second module publishes the same mechanism name.
    #[test]
    fn a_second_module_publishing_one_mechanism_is_a_finding() {
        let mut report = Report::clean();
        inspect_mechanism_owners(
            Path::new("docs/public-api/vyre-runtime.txt"),
            "pub struct vyre_runtime::resident_work_queue::WorkQueue\n\
             pub struct vyre_runtime::routed_shard_scheduling::WorkQueue\n",
            &mut report,
        );
        assert_eq!(report.count(), 1, "{}", report.finding_messages());
        let finding = &report.findings[0];
        assert!(
            finding.message.contains("`WorkQueue`"),
            "{}",
            finding.message
        );
        assert!(
            finding.message.contains("resident_work_queue")
                && finding.message.contains("routed_shard_scheduling"),
            "the finding names neither owner: {}",
            finding.message
        );
    }

    /// Distinct mechanism names in sibling modules are not a second owner.
    ///
    /// The runtime publishes a resident queue, a routed queue, and a replay queue on purpose, so
    /// a rule that counted the noun `Queue` would reject the shape the contract asks for.
    #[test]
    fn sibling_modules_publishing_distinct_mechanisms_are_not_a_finding() {
        let mut report = Report::clean();
        inspect_mechanism_owners(
            Path::new("docs/public-api/vyre-runtime.txt"),
            "pub struct vyre_runtime::resident_work_queue::ResidentWorkQueue\n\
             pub struct vyre_runtime::routed_work_queue::RoutedWorkQueue\n\
             pub struct vyre_runtime::replay::ReplayQueue\n\
             pub struct vyre_runtime::pipeline_cache::PipelineCache\n\
             pub struct vyre_runtime::retained_page_cache::RetainedPageCache\n",
            &mut report,
        );
        assert_eq!(report.count(), 0, "{}", report.finding_messages());
    }

    /// An associated item is not a second owner of its own name.
    ///
    /// `vyre-foundation` publishes `substrate::Query::Output` and `lower::ProgramEffects::Output`.
    /// Both are associated types of one trait each, and a rule that read the parent segment as a
    /// module would report `Output` as a duplicated mechanism in every trait that has one.
    #[test]
    fn an_associated_item_is_not_a_second_owner() {
        let mut report = Report::clean();
        inspect_mechanism_owners(
            Path::new("docs/public-api/vyre-foundation.txt"),
            "pub type vyre_foundation::substrate::Query::Output\n\
             pub type vyre_foundation::lower::ProgramEffects::Output\n\
             pub fn vyre_foundation::visit::NodeVisitor::Break(&self)\n",
            &mut report,
        );
        assert_eq!(report.count(), 0, "{}", report.finding_messages());
    }

    /// A generic parameter list is not part of the mechanism's name.
    #[test]
    fn a_generic_parameter_list_is_not_part_of_the_name() {
        assert_eq!(
            declared_type_path("pub struct vyre_runtime::tenant::Quota<'a, T>"),
            Some("vyre_runtime::tenant::Quota")
        );
        assert_eq!(
            declared_type_path("#[non_exhaustive] pub enum vyre_runtime::PipelineError"),
            Some("vyre_runtime::PipelineError")
        );
        assert_eq!(
            declared_type_path("pub fn vyre_runtime::tenant::Quota::new() -> Self"),
            None
        );
    }

    /// Every term the contract ships is caught in a published name built from that same term.
    ///
    /// The variant space is the contract read at run time, so adding a `[[term]]` row the matcher
    /// cannot catch turns this red instead of passing unexamined.
    #[test]
    fn every_contract_term_is_caught_in_a_published_name() {
        for term in shipped_terms() {
            let camel: String = term
                .segments
                .iter()
                .map(|part| {
                    let mut chars = part.chars();
                    match chars.next() {
                        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                        None => String::new(),
                    }
                })
                .collect();
            let snake = term.segments.join("_");

            for published in [
                format!("pub struct vyre_runtime::{snake}::{camel}Owner"),
                format!("pub fn vyre_driver::transfer::submit_{snake}(&self)"),
            ] {
                let mut report = Report::clean();
                inspect_snapshot(
                    Path::new("docs/public-api/vyre-runtime.txt"),
                    "vyre-runtime",
                    "runtime",
                    &published,
                    std::slice::from_ref(&term),
                    &mut report,
                );
                assert!(
                    report.count() >= 1,
                    "term `{}` escapes in `{published}`",
                    term.name()
                );
            }
        }
    }

    /// A crate named for the container format it adapts may publish that format's own name.
    ///
    /// `vyre-safetensors` maps a container format into transfer descriptors; forbidding the word
    /// in its own API would leave the crate unable to name what it reads. The exemption is the
    /// package name, so it covers no other crate.
    #[test]
    fn a_crate_named_for_what_it_adapts_may_publish_its_own_name() {
        let terms = vec![Term {
            segments: vec!["safetensors".to_string()],
            mechanism: "publish the container-format adapter".to_string(),
        }];
        let published = "pub struct vyre_safetensors::SafetensorIndex\n";

        let mut exempt = Report::clean();
        inspect_snapshot(
            Path::new("docs/public-api/vyre-safetensors.txt"),
            "vyre-safetensors",
            "runtime",
            published,
            &terms,
            &mut exempt,
        );
        assert_eq!(exempt.count(), 0, "{}", exempt.finding_messages());

        let mut other = Report::clean();
        inspect_snapshot(
            Path::new("docs/public-api/vyre-runtime.txt"),
            "vyre-runtime",
            "runtime",
            published,
            &terms,
            &mut other,
        );
        assert_eq!(other.count(), 1, "{}", other.finding_messages());
    }

    /// A mechanism name that happens to contain a term's letters is not a finding.
    ///
    /// Segment matching is what separates `ExpertQueue` from `expertise` and a cancellation token
    /// from a text token, so the gate needs no per-item exemption to stay quiet on either.
    #[test]
    fn a_neutral_published_name_is_not_a_finding() {
        let terms = shipped_terms();
        let published = concat!(
            "pub struct vyre_runtime::structured_concurrency::CancellationToken\n",
            "pub fn vyre_runtime::routed_work_queue::RoutedWorkQueue::submit(&self)\n",
            "pub struct vyre_runtime::retained_page_cache::RetainedPageCache\n",
            "pub fn vyre_foundation::optimizer::expertise_rank(&self) -> u32\n",
        );
        let mut report = Report::clean();
        let inspected = inspect_snapshot(
            Path::new("docs/public-api/vyre-runtime.txt"),
            "vyre-runtime",
            "runtime",
            published,
            &terms,
            &mut report,
        );
        assert_eq!(inspected, 4);
        assert_eq!(report.count(), 0, "{}", report.finding_messages());
    }

    /// A term row that states no mechanism is dropped and reported, because a finding without a
    /// replacement tells an author to rename and nothing more.
    #[test]
    fn a_term_stating_no_mechanism_is_reported_and_not_enforced() {
        let contract: toml::Table = toml::from_str(
            "[[term]]\nsegments = [\"expert\"]\nmechanism = \"  \"\n\
             [[term]]\nsegments = []\nmechanism = \"unused\"\n",
        )
        .expect("fixture parses");
        let mut report = Report::clean();
        let terms = decode_terms(&contract, &mut report);
        assert!(terms.is_empty());
        assert_eq!(report.count(), 2, "{}", report.finding_messages());
        assert!(report.finding_messages().contains("no mechanism"));
        assert!(report.finding_messages().contains("no segments"));
    }

    /// An exempt layer without a reason exempts nothing, so silence has to be argued for.
    #[test]
    fn an_exempt_layer_without_a_reason_exempts_nothing() {
        let contract: toml::Table =
            toml::from_str("[[exempt_layer]]\nname = \"runtime\"\n").expect("fixture parses");
        let mut report = Report::clean();
        let exempt = decode_exempt_layers(&contract, &mut report);
        assert!(exempt.is_empty());
        assert_eq!(report.count(), 1, "{}", report.finding_messages());
    }

    /// Casing and separators reach the same segments, so a rename cannot hide a term.
    #[test]
    fn casing_and_separators_reach_the_same_segments() {
        let expected = ["expert", "queue"];
        for spelling in [
            "ExpertQueue",
            "expert_queue",
            "expert-queue",
            "EXPERT_QUEUE",
            "expert::queue",
        ] {
            assert_eq!(
                identifier_segments(spelling),
                expected,
                "spelling `{spelling}` split differently"
            );
        }
        assert_eq!(identifier_segments("KVCache"), ["kv", "cache"]);
        assert_eq!(identifier_segments("expertise"), ["expertise"]);
    }
}
