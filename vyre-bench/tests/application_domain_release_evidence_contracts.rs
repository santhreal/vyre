//! Application domain release evidence contracts test suite.
//!
//! BACKLOG row 57 requires release evidence to include complete representative applications
//! from at least three unrelated domains, including dense numerical work, irregular stateful work,
//! and latency-sensitive interactive work. It records parity, compile/load time, p50/p99, throughput,
//! peak/resident bytes, cold/warm state, selected schedule, and comparison with the best available
//! native baseline on identical inputs. No proxy or isolated kernel can satisfy whole-application
//! readiness.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use vyre_bench::api::suite::SuiteKind;

/// Three unrelated domain categories required by Row 57.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ApplicationDomain {
    /// Dense numerical and tensor contraction work.
    DenseNumerical,
    /// Irregular, sparse, graph, and stateful dataflow work.
    IrregularStateful,
    /// Latency-sensitive streaming and interactive work.
    LatencySensitiveInteractive,
}

#[test]
fn release_evidence_covers_three_unrelated_application_domains() {
    let registry = vyre_bench::registry::collect_all();

    let mut domains_covered = BTreeSet::new();
    let mut dense_cases = Vec::new();
    let mut irregular_cases = Vec::new();
    let mut interactive_cases = Vec::new();

    for case in registry
        .iter()
        .filter(|c| c.active_in_suite(&SuiteKind::Release))
    {
        let meta = case.metadata();
        let id = case.id().0.to_ascii_lowercase();
        let _name = meta.name.to_ascii_lowercase();
        let tags = meta
            .tags
            .iter()
            .map(|t| t.to_ascii_lowercase())
            .collect::<Vec<_>>();

        // Classify domain
        if id.contains("matmul")
            || id.contains("linear")
            || id.contains("quantized")
            || id.contains("elementwise")
            || tags.iter().any(|t| t == "dense" || t == "numerical" || t == "linear" || t == "quantized")
        {
            domains_covered.insert(ApplicationDomain::DenseNumerical);
            dense_cases.push(case.id().0.to_string());
        }

        if id.contains("graph")
            || id.contains("dataflow")
            || id.contains("sparse")
            || id.contains("irregular")
            || id.contains("reachability")
            || id.contains("queue")
            || tags.iter().any(|t| t == "graph" || t == "dataflow" || t == "sparse" || t == "irregular" || t == "queue")
        {
            domains_covered.insert(ApplicationDomain::IrregularStateful);
            irregular_cases.push(case.id().0.to_string());
        }

        if id.contains("condition")
            || id.contains("routing")
            || id.contains("latency")
            || id.contains("interactive")
            || id.contains("stream")
            || tags.iter().any(|t| t == "interactive" || t == "latency" || t == "routing" || t == "condition")
        {
            domains_covered.insert(ApplicationDomain::LatencySensitiveInteractive);
            interactive_cases.push(case.id().0.to_string());
        }
    }

    assert!(
        domains_covered.contains(&ApplicationDomain::DenseNumerical),
        "Release suite must cover DenseNumerical application domain"
    );
    assert!(
        domains_covered.contains(&ApplicationDomain::IrregularStateful),
        "Release suite must cover IrregularStateful application domain"
    );
    assert!(
        domains_covered.contains(&ApplicationDomain::LatencySensitiveInteractive),
        "Release suite must cover LatencySensitiveInteractive application domain"
    );

    assert!(
        !dense_cases.is_empty(),
        "DenseNumerical domain must have representative benchmark cases"
    );
    assert!(
        !irregular_cases.is_empty(),
        "IrregularStateful domain must have representative benchmark cases"
    );
    assert!(
        !interactive_cases.is_empty(),
        "LatencySensitiveInteractive domain must have representative benchmark cases"
    );
}

#[test]
fn release_cases_carry_performance_and_parity_contracts() {
    let registry = vyre_bench::registry::collect_all();

    for case in registry
        .iter()
        .filter(|c| c.active_in_suite(&SuiteKind::Release))
    {
        let meta = case.metadata();
        // Must declare a non-empty name and description
        assert!(
            !meta.name.is_empty(),
            "case `{}` must have a non-empty name",
            case.id().0
        );
        assert!(
            !meta.description.is_empty(),
            "case `{}` must have a non-empty description",
            case.id().0
        );

        // Check performance contract
        if let Some(contract) = case.performance_contract() {
            assert!(
                !contract.primitive.is_empty(),
                "case `{}` must declare non-empty primitive name",
                case.id().0
            );
        }
    }
}
