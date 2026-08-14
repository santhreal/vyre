#![allow(clippy::module_name_repetitions)]

use crate::pass::{
    MetadataEnum, PassArgs, PASS_BOUNDARY_CLASS, PASS_COST_MODEL_FAMILY, PASS_PHASE,
};
use quote::quote;
use syn::LitStr;

#[test]
fn pass_args_parse_full_metadata_contract() {
    let args = syn::parse2::<PassArgs>(quote! {
        name = "canonical_fold",
        requires = ["domtree", "alias"],
        invalidates = ["cfg"],
        phase = "dataflow",
        boundary_class = "backend_aware",
        requires_caps = ["resident_buffers"],
        preserves_abi = false,
        cost_model_family = "megakernel",
        analyze = "always",
    })
    .expect("Fix: full pass metadata should parse");

    assert_eq!(args.name.value(), "canonical_fold");
    assert_eq!(
        args.requires.iter().map(LitStr::value).collect::<Vec<_>>(),
        vec!["domtree", "alias"]
    );
    assert_eq!(
        args.invalidates
            .iter()
            .map(LitStr::value)
            .collect::<Vec<_>>(),
        vec!["cfg"]
    );
    assert_eq!(
        args.requires_caps
            .iter()
            .map(LitStr::value)
            .collect::<Vec<_>>(),
        vec!["resident_buffers"]
    );
    assert_eq!(
        args.phase.as_ref().map(LitStr::value),
        Some("dataflow".to_string())
    );
    assert_eq!(
        args.boundary_class.as_ref().map(LitStr::value),
        Some("backend_aware".to_string())
    );
    assert_eq!(
        args.cost_model_family.as_ref().map(LitStr::value),
        Some("megakernel".to_string())
    );
    assert_eq!(args.preserves_abi.map(|lit| lit.value), Some(false));
    assert!(args.analyze_always);
}

#[test]
fn pass_args_reject_unknown_argument_with_actionable_fix() {
    let err = syn::parse2::<PassArgs>(quote! {
        name = "bad",
        scheduler = "late",
    })
    .err()
    .expect("Fix: unknown pass argument must fail at macro parse time");

    assert!(err.to_string().contains("unsupported vyre_pass argument"));
    assert!(err.to_string().contains("Fix:"));
}

#[test]
fn pass_args_reject_duplicate_top_level_argument() {
    let err = syn::parse2::<PassArgs>(quote! {
        name = "bad",
        requires = [],
        requires = ["late_override"],
        invalidates = [],
    })
    .err()
    .expect("Fix: vyre_pass must reject duplicate top-level arguments");

    assert!(err
        .to_string()
        .contains("duplicate macro argument `requires`"));
    assert!(err.to_string().contains("Fix:"));
}

#[test]
fn pass_args_reject_non_string_metadata_arrays() {
    let err = syn::parse2::<PassArgs>(quote! {
        name = "bad",
        requires = [123],
    })
    .err()
    .expect("Fix: metadata arrays must accept only string literals");

    assert!(err.to_string().contains("only string literals"));
    assert!(err.to_string().contains("Fix:"));
}

#[test]
fn pass_phase_rejects_consumer_prefixed_phase_names() {
    let phase = LitStr::new("consumer-dataflow", proc_macro2::Span::call_site());
    let err = PASS_PHASE
        .tokens(Some(&phase))
        .expect_err("Fix: platform pass phases must remain consumer neutral");

    assert!(err.to_string().contains("unsupported pass phase"));
    assert!(err.to_string().contains("Fix:"));
}

/// Every metadata enum the attribute lowers into, so a new one is covered by
/// the tests below the moment it is declared rather than when someone
/// remembers to extend a case list.
const METADATA_ENUMS: &[&MetadataEnum] =
    &[&PASS_PHASE, &PASS_BOUNDARY_CLASS, &PASS_COST_MODEL_FAMILY];

#[test]
fn generated_pass_args_matrix_covers_every_metadata_enum_combination() {
    let mut visited = std::collections::BTreeSet::new();
    for (phase, phase_variant) in PASS_PHASE.rows {
        for (boundary, boundary_variant) in PASS_BOUNDARY_CLASS.rows {
            for (cost, cost_variant) in PASS_COST_MODEL_FAMILY.rows {
                for analyze_always in [false, true] {
                    let analyze = if analyze_always {
                        quote! { , analyze = "always" }
                    } else {
                        quote! {}
                    };
                    let tokens = quote! {
                        name = "generated_parse_case",
                        requires = ["domtree", "alias"],
                        invalidates = ["cfg"],
                        phase = #phase,
                        boundary_class = #boundary,
                        requires_caps = ["cuda", "resident"],
                        preserves_abi = false,
                        cost_model_family = #cost
                        #analyze
                    };
                    let args = syn::parse2::<PassArgs>(tokens)
                        .expect("Fix: generated pass metadata parser case should parse");

                    assert_eq!(args.name.value(), "generated_parse_case");
                    assert_eq!(args.requires.len(), 2);
                    assert_eq!(args.invalidates.len(), 1);
                    assert_eq!(args.requires_caps.len(), 2);
                    assert_eq!(args.preserves_abi.map(|value| value.value), Some(false));
                    assert_eq!(
                        args.phase.as_ref().map(LitStr::value).as_deref(),
                        Some(*phase)
                    );
                    assert_eq!(
                        PASS_PHASE
                            .tokens(args.phase.as_ref())
                            .expect("Fix: generated phase must lower")
                            .to_string(),
                        format!(":: vyre :: optimizer :: PassPhase :: {phase_variant}")
                    );
                    assert_eq!(
                        PASS_BOUNDARY_CLASS
                            .tokens(args.boundary_class.as_ref())
                            .expect("Fix: generated boundary must lower")
                            .to_string(),
                        format!(":: vyre :: optimizer :: PassBoundaryClass :: {boundary_variant}")
                    );
                    assert_eq!(
                        PASS_COST_MODEL_FAMILY
                            .tokens(args.cost_model_family.as_ref())
                            .expect("Fix: generated cost family must lower")
                            .to_string(),
                        format!(":: vyre :: optimizer :: CostModelFamily :: {cost_variant}")
                    );
                    assert_eq!(args.analyze_always, analyze_always);
                    visited.insert((*phase, *boundary, *cost, analyze_always));
                }
            }
        }
    }

    assert_eq!(
        visited.len(),
        PASS_PHASE.rows.len()
            * PASS_BOUNDARY_CLASS.rows.len()
            * PASS_COST_MODEL_FAMILY.rows.len()
            * 2
    );
}

#[test]
fn omitted_metadata_argument_lowers_to_the_first_declared_row() {
    for enumeration in METADATA_ENUMS {
        let (_, default_variant) = enumeration.rows[0];
        assert_eq!(
            enumeration
                .tokens(None)
                .expect("Fix: an omitted metadata argument must lower to a default variant")
                .to_string(),
            format!(
                ":: vyre :: optimizer :: {} :: {default_variant}",
                enumeration.type_name
            ),
            "{}",
            enumeration.argument
        );
    }
}

#[test]
fn every_rejected_metadata_string_names_every_accepted_string() {
    for enumeration in METADATA_ENUMS {
        let literal = LitStr::new("no_such_value", proc_macro2::Span::call_site());
        let message = enumeration
            .tokens(Some(&literal))
            .expect_err("Fix: an unknown metadata string must be rejected")
            .to_string();

        assert!(
            message.contains(&format!("unsupported pass {}", enumeration.argument)),
            "{message}"
        );
        assert!(message.contains("Fix:"), "{message}");
        for (accepted, _) in enumeration.rows {
            assert!(
                message.contains(accepted),
                "{} diagnostic omits `{accepted}`: {message}",
                enumeration.argument
            );
        }
    }
}

#[test]
fn every_metadata_row_declares_a_distinct_string_and_variant() {
    for enumeration in METADATA_ENUMS {
        let strings: std::collections::BTreeSet<_> =
            enumeration.rows.iter().map(|(text, _)| *text).collect();
        let variants: std::collections::BTreeSet<_> = enumeration
            .rows
            .iter()
            .map(|(_, variant)| *variant)
            .collect();

        assert_eq!(
            strings.len(),
            enumeration.rows.len(),
            "{}",
            enumeration.argument
        );
        assert_eq!(
            variants.len(),
            enumeration.rows.len(),
            "{}",
            enumeration.argument
        );
    }
}
