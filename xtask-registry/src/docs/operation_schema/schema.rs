//! Wire types of `docs/generated/OP_SCHEMA.json` and the version that pins them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Wire version of `docs/generated/OP_SCHEMA.json`.
///
/// `scripts/architecture_docs.py` reads the same file and pins the same
/// number. It cannot import this constant, so
/// `the_python_contract_pins_the_same_operation_schema_version` fails when the
/// two drift; that drift shipped once already, with the generator on 3 and the
/// script still demanding 2.
pub(crate) const SCHEMA_VERSION: u32 = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct OperationSchema {
    pub(crate) schema_version: u32,
    pub(crate) authority: String,
    pub(crate) operation_count: usize,
    pub(crate) tier_counts: BTreeMap<String, usize>,
    pub(crate) category_counts: BTreeMap<String, usize>,
    pub(crate) operations: Vec<OperationRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct OperationRecord {
    pub(crate) id: String,
    pub(crate) tier: String,
    pub(crate) category: String,
    pub(crate) signature: OperationSignature,
    pub(crate) features: Vec<String>,
    pub(crate) oracle: OracleContract,
    pub(crate) backend_support: BTreeMap<String, BackendSupport>,
    pub(crate) target_facets: Vec<String>,
    pub(crate) laws: Vec<String>,
    pub(crate) composition_chain: Vec<CompositionStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct OperationSignature {
    pub(crate) kind: String,
    pub(crate) buffers: Vec<BufferSignature>,
    pub(crate) inputs: Vec<TypedParameter>,
    pub(crate) outputs: Vec<TypedParameter>,
    pub(crate) attributes: Vec<String>,
    pub(crate) bytes_extraction: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TypedParameter {
    pub(crate) name: String,
    pub(crate) data_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct BufferSignature {
    pub(crate) binding: u32,
    pub(crate) name: String,
    pub(crate) access: String,
    pub(crate) memory: String,
    pub(crate) element: String,
    pub(crate) count: u32,
    pub(crate) pipeline_live_out: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct OracleContract {
    pub(crate) reference_eval: bool,
    pub(crate) flat_reference_facet: bool,
    pub(crate) fixture_inputs: bool,
    pub(crate) expected_output: bool,
    pub(crate) tolerance_ulp: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct BackendSupport {
    pub(crate) status: String,
    pub(crate) test_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct CompositionStep {
    pub(crate) depth: usize,
    pub(crate) operation: String,
    pub(crate) registered: bool,
}
