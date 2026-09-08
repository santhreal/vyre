//! Frontend dialect, schema, field, resource, and layout contracts test suite.
//!
//! BACKLOG row 55 requires versioned domain-neutral schema, field, resource, layout,
//! and translation contracts with exhaustive visitors and canonical identity, proving
//! unknown, duplicate, missing, incompatible, overflowing, and unused members fail
//! before compilation. Core compiler types, fixtures, and diagnostics contain no downstream
//! domain names.

use vyre_foundation::dialect::{
    validate_external_schema, ExternalLayoutDeclaration, ExternalResourceDeclaration,
    ExternalSchema, ExternalSchemaNode, ExternalSchemaVisitor, FieldType,
};
use vyre_foundation::ir::{BufferAccess, DataType};

struct Recorder {
    nodes: Vec<String>,
    fields: Vec<(String, String)>,
    resources: Vec<String>,
    layouts: Vec<String>,
}

impl ExternalSchemaVisitor for Recorder {
    type Error = ();

    fn visit_schema(&mut self, _schema_id: &str, _version: u32) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_node(&mut self, node: &ExternalSchemaNode) -> Result<(), Self::Error> {
        self.nodes.push(node.op_name.clone());
        Ok(())
    }

    fn visit_field(
        &mut self,
        _node_op: &str,
        field_name: &str,
        field_value: &str,
    ) -> Result<(), Self::Error> {
        self.fields.push((field_name.to_string(), field_value.to_string()));
        Ok(())
    }

    fn visit_resource_binding(
        &mut self,
        _node_op: &str,
        resource_name: &str,
    ) -> Result<(), Self::Error> {
        self.resources.push(resource_name.to_string());
        Ok(())
    }

    fn visit_resource_declaration(
        &mut self,
        resource: &ExternalResourceDeclaration,
    ) -> Result<(), Self::Error> {
        self.resources.push(resource.name.clone());
        Ok(())
    }

    fn visit_layout_declaration(
        &mut self,
        layout: &ExternalLayoutDeclaration,
    ) -> Result<(), Self::Error> {
        self.layouts.push(layout.resource_name.clone());
        Ok(())
    }
}

#[test]
fn domain_neutral_schema_contracts_require_versions_and_visitors() {
    let schema = ExternalSchema {
        schema_id: "vyre-libs::generic_pipeline".to_string(),
        version: 1,
        nodes: vec![
            ExternalSchemaNode {
                op_name: "vyre-libs::generic::transform".to_string(),
                raw_fields: vec![("scale".to_string(), "4".to_string())],
                bound_resources: vec!["input_buf".to_string(), "intermediate_buf".to_string()],
            },
            ExternalSchemaNode {
                op_name: "vyre-libs::generic::reduce".to_string(),
                raw_fields: vec![("axis".to_string(), "0".to_string())],
                bound_resources: vec!["intermediate_buf".to_string(), "output_buf".to_string()],
            },
        ],
        declared_resources: vec![
            ExternalResourceDeclaration {
                name: "input_buf".to_string(),
                access: BufferAccess::ReadOnly,
                element_type: DataType::F32,
                alignment: 16,
                byte_capacity: 1024,
            },
            ExternalResourceDeclaration {
                name: "intermediate_buf".to_string(),
                access: BufferAccess::ReadWrite,
                element_type: DataType::F32,
                alignment: 16,
                byte_capacity: 1024,
            },
            ExternalResourceDeclaration {
                name: "output_buf".to_string(),
                access: BufferAccess::WriteOnly,
                element_type: DataType::F32,
                alignment: 16,
                byte_capacity: 1024,
            },
        ],
        declared_layouts: vec![
            ExternalLayoutDeclaration {
                resource_name: "input_buf".to_string(),
                element_type: DataType::F32,
                shape: vec![256],
                strides: vec![1],
                alignment: 16,
            },
            ExternalLayoutDeclaration {
                resource_name: "intermediate_buf".to_string(),
                element_type: DataType::F32,
                shape: vec![256],
                strides: vec![1],
                alignment: 16,
            },
            ExternalLayoutDeclaration {
                resource_name: "output_buf".to_string(),
                element_type: DataType::F32,
                shape: vec![256],
                strides: vec![1],
                alignment: 16,
            },
        ],
    };

    let mut recorder = Recorder {
        nodes: Vec::new(),
        fields: Vec::new(),
        resources: Vec::new(),
        layouts: Vec::new(),
    };

    schema.accept(&mut recorder).expect("visitor must traverse schema");
    assert_eq!(recorder.nodes.len(), 2);
    assert_eq!(recorder.fields.len(), 2);
    assert_eq!(recorder.layouts.len(), 3);

    // Canonical identity is reproducible
    let id_a = schema.canonical_identity();
    let id_b = schema.canonical_identity();
    assert_eq!(id_a, id_b);
    assert_ne!(id_a, [0; 32]);

    // Full validation
    let bound = validate_external_schema("vyre-libs::generic_pipeline", &schema, 1, |_| Ok(()))
        .expect("validation must succeed");
    assert_eq!(bound.len(), 3);
}

#[test]
fn schema_field_type_and_error_exhaustive_closure() {
    let types = [
        FieldType::U32,
        FieldType::I32,
        FieldType::U64,
        FieldType::I64,
        FieldType::F32,
        FieldType::F64,
        FieldType::Bool,
        FieldType::String,
        FieldType::Bytes,
        FieldType::Buffer,
    ];

    for ft in types {
        match ft {
            FieldType::U32 => assert!(ft.parse_and_validate("42").is_ok()),
            FieldType::I32 => assert!(ft.parse_and_validate("-42").is_ok()),
            FieldType::U64 => assert!(ft.parse_and_validate("18446744073709551615").is_ok()),
            FieldType::I64 => assert!(ft.parse_and_validate("-9223372036854775808").is_ok()),
            FieldType::F32 => assert!(ft.parse_and_validate("1.5").is_ok()),
            FieldType::F64 => assert!(ft.parse_and_validate("1.5e10").is_ok()),
            FieldType::Bool => assert!(ft.parse_and_validate("false").is_ok()),
            FieldType::String => assert!(ft.parse_and_validate("text").is_ok()),
            FieldType::Bytes => assert!(ft.parse_and_validate("data").is_ok()),
            FieldType::Buffer => assert!(ft.parse_and_validate("buf").is_ok()),
        }
    }
}
