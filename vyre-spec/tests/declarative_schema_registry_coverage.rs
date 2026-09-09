//! Tests for the declarative schema registry, invariant enforcement,
//! runtime derivation, and documentation/fuzz generation.
//!
//! WHY: closes the class "schema registry entries omit bounds, identity fields, or domain separators".

use vyre_spec::schema_registry::{DefaultsPolicy, FieldType, SchemaId, SchemaRegistry};

#[test]
fn runtime_derived_schema_registry_covers_all_schema_ids() {
    let all_ids = SchemaId::ALL;
    assert_eq!(
        all_ids.len(),
        22,
        "Fix: all 22 persisted/transmitted schema IDs must be enumerated in ALL"
    );

    let all_schemas = SchemaRegistry::all();
    assert_eq!(
        all_schemas.len(),
        all_ids.len(),
        "Fix: every SchemaId variant must have an entry in CANONICAL_SCHEMA_REGISTRY"
    );

    for &id in all_ids {
        let def = SchemaRegistry::lookup(id)
            .unwrap_or_else(|| panic!("Fix: missing SchemaDefinition for {id:?}"));
        assert_eq!(def.id, id);
        assert!(
            def.validate_invariants(),
            "Fix: schema definition for {id:?} failed structural invariant validation"
        );
        assert!(
            !def.domain_separator.is_empty(),
            "Fix: domain separator must not be empty for {id:?}"
        );
        assert!(
            def.bounds.max_bytes > 0,
            "Fix: max_bytes must be positive for {id:?}"
        );
        assert!(
            def.bounds.max_depth > 0,
            "Fix: max_depth must be positive for {id:?}"
        );
        assert!(
            def.bounds.max_elements > 0,
            "Fix: max_elements must be positive for {id:?}"
        );
        assert!(
            !def.owning_package.is_empty(),
            "Fix: owning_package must not be empty for {id:?}"
        );
        assert!(
            !def.stale_fixtures.is_empty(),
            "Fix: stale_fixtures must declare known previous versions for {id:?}"
        );

        // Verify identity fields are present
        let identity_field_count = def.fields.iter().filter(|f| f.is_identity).count();
        assert!(
            identity_field_count > 0,
            "Fix: schema {id:?} must define at least one identity field"
        );

        // Verify documentation generation produces non-empty markdown
        let doc = def.generate_documentation();
        assert!(
            doc.contains("# Schema:"),
            "Fix: generated doc must contain title"
        );
        assert!(
            doc.contains(def.id.as_str()),
            "Fix: generated doc must contain schema id string"
        );

        // Verify fuzz grammar generation
        let grammar = def.fuzz_grammar();
        assert!(
            grammar.contains("<fields_"),
            "Fix: fuzz grammar must define field rules"
        );
    }
}

#[test]
fn schema_field_numbers_are_strictly_ascending() {
    for def in SchemaRegistry::all() {
        let mut prev = 0;
        for field in def.fields {
            assert!(
                field.number > prev,
                "Fix: field number {} not strictly greater than previous {prev} in {:?}",
                field.number,
                def.id
            );
            prev = field.number;
        }
    }
}

#[test]
fn domain_separators_are_globally_unique() {
    let mut seen = std::collections::BTreeSet::new();
    for def in SchemaRegistry::all() {
        assert!(
            seen.insert(def.domain_separator),
            "Fix: duplicate domain separator `{}` found for {:?}",
            def.domain_separator,
            def.id
        );
    }
}

#[test]
fn schema_id_strings_are_globally_unique() {
    let mut seen = std::collections::BTreeSet::new();
    for &id in SchemaId::ALL {
        assert!(
            seen.insert(id.as_str()),
            "Fix: duplicate schema id string `{}` found for {id:?}",
            id.as_str()
        );
    }
}

#[test]
fn defaults_policy_is_explicit_for_every_entry() {
    for def in SchemaRegistry::all() {
        match def.defaults_policy {
            DefaultsPolicy::NoDefaults
            | DefaultsPolicy::ExplicitDefaultOnly
            | DefaultsPolicy::PreserveUnknownSafe => {}
        }
    }
}
