//! Declarative dialect definitions generating builders, visitors, wire schema,
//! reference obligations, property generators, metamorphic tests, documentation,
//! and closure joins.

use crate::ir::Program;

/// Visitor over semantic operation operands and graph dependencies.
pub trait OperationVisitor {
    /// Visit a buffer or value operand name.
    fn visit_operand(&mut self, name: &str);
    /// Visit a memory effect classification.
    fn visit_effect(&mut self, effect: &str);
}

/// Metadata and generated projections for a declarative dialect operation.
pub trait DialectOperationSpec: Send + Sync + 'static {
    /// Stable operation identity.
    fn id(&self) -> &'static str;
    /// Documentation string in markdown/plain text.
    fn documentation(&self) -> &'static str;
    /// Reference obligation summary.
    fn reference_obligation(&self) -> &'static str;
    /// Build the canonical program.
    fn build_program(&self) -> Program;
    /// Run metamorphic validation over test inputs.
    fn metamorphic_test(&self, inputs: &[Vec<u8>]) -> bool;
    /// Generate a deterministic property test input payload from a pseudo-random seed.
    fn generate_property_input(&self, seed: u64) -> Vec<u8>;
    /// Generate the canonical semantic contract record.
    fn contract_record(&self) -> vyre_spec::SemanticContractRecord;
}

/// Declarative dialect operation definition macro generating single-source builders,
/// visitors, wire schema, reference obligations, property generators, metamorphic tests,
/// documentation, and the four closure joins.
#[macro_export]
macro_rules! declare_dialect_op {
    (
        id: $id:expr,
        tier: $tier:expr,
        category: $cat:expr,
        doc: $doc:expr,
        reference_obligation: $ref_ob:expr,
        $(semantic_version: $sem_ver:expr,)?
        $(signature: $sig:expr,)?
        $(laws: $laws:expr,)?
        $(absence: $absence:expr,)?
        $(numeric: $num:expr,)?
        $(geometry_requirements: $geom:expr,)?
        $(explicit_effects: $eff:expr,)?
        $(explicit_capabilities: $caps:expr,)?
        builder: $builder:expr,
        $(metamorphic_check: $metamorphic:expr,)?
        $(test_inputs: $inputs:expr,)?
        $(expected_output: $expected:expr)?
    ) => {
        const _: () = {
            $crate::inventory::submit! {
                $crate::operation::SemanticDescriptor {
                    id: $id,
                    semantic_version: 1 $( - 1 + $sem_ver )?,
                    signature: None $(.or(Some($sig)))?,
                    tier: $tier,
                    category: Some($cat),
                    laws: {
                        #[allow(unused_mut, unused_assignments)]
                        let mut val: &'static [&'static str] = &[];
                        $(val = $laws;)?
                        val
                    },
                    numeric: {
                        #[allow(unused_mut, unused_assignments)]
                        let mut val = $crate::numeric::NumericContract::EXACT;
                        $(val = $num;)?
                        val
                    },
                    geometry_requirements: {
                        #[allow(unused_mut, unused_assignments)]
                        let mut val = $crate::geometry::GeometryRequirements::agnostic();
                        $(val = $geom;)?
                        val
                    },
                    explicit_effects: None $(.or(Some($eff)))?,
                    explicit_capabilities: None $(.or(Some($caps)))?,
                    absence: None $(.or(Some($absence)))?,
                }
            }
            $crate::inventory::submit! {
                $crate::operation::LoweringProvider {
                    id: $id,
                    build: Some($builder),
                }
            }
            $crate::inventory::submit! {
                $crate::operation::ConformanceProvider {
                    id: $id,
                    test_inputs: None $(.or(Some($inputs)))?,
                    expected_output: None $(.or(Some($expected)))?,
                }
            }
            $crate::inventory::submit! {
                $crate::operation::ContractProvider {
                    id: $id,
                    contract: Some(|| {
                        let desc = $crate::operation::SemanticDescriptor {
                            id: $id,
                            semantic_version: 1 $( - 1 + $sem_ver )?,
                            signature: None $(.or(Some($sig)))?,
                            tier: $tier,
                            category: Some($cat),
                            laws: {
                                #[allow(unused_mut, unused_assignments)]
                                let mut val: &'static [&'static str] = &[];
                                $(val = $laws;)?
                                val
                            },
                            numeric: {
                                #[allow(unused_mut, unused_assignments)]
                                let mut val = $crate::numeric::NumericContract::EXACT;
                                $(val = $num;)?
                                val
                            },
                            geometry_requirements: {
                                #[allow(unused_mut, unused_assignments)]
                                let mut val = $crate::geometry::GeometryRequirements::agnostic();
                                $(val = $geom;)?
                                val
                            },
                            explicit_effects: None $(.or(Some($eff)))?,
                            explicit_capabilities: None $(.or(Some($caps)))?,
                            absence: None $(.or(Some($absence)))?,
                        };
                        let op = $crate::operation::SemanticOperation {
                            id: desc.id,
                            semantic_version: desc.semantic_version,
                            signature: desc.signature,
                            tier: desc.tier,
                            category: desc.category,
                            build: Some($builder),
                            test_inputs: None $(.or(Some($inputs)))?,
                            expected_output: None $(.or(Some($expected)))?,
                            laws: desc.laws,
                            numeric: desc.numeric,
                            geometry_requirements: desc.geometry_requirements,
                            source_file: file!(),
                            explicit_effects: desc.explicit_effects,
                            explicit_capabilities: desc.explicit_capabilities,
                            absence: desc.absence,
                        };
                        op.contract_record()
                    }),
                }
            }
        };
    };
}
