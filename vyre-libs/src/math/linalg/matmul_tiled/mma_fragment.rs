//! Tensor-core MMA fragment primitive for M16N8K16.
//!
//! Emits the exact 4-FMA sequence that B6 (`matmul_promote`) detects
//! and collapses into `KernelOpKind::MatrixMma`.

use vyre_foundation::ir::{Expr, Node};

use super::tensor_core_policy::MatmulKernelPath;

/// One descriptor-level MMA instruction shape and operand precision.
///
/// The name states the shape and the precision because that is the whole
/// property a caller gates on. Which target lowers it is the driver's
/// question, and the driver answers it by filling in [`MmaCapabilityRecord`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum MmaInstructionClass {
    DescriptorMmaM16N8K16F16,
    DescriptorMmaM16N8K16Bf16,
    DescriptorMmaM16N8K4Tf32,
}

/// What a target can do with a matrix-multiply-accumulate fragment.
///
/// `descriptor_mma` is the coarse gate: whether the target lowers a
/// descriptor-level MMA op at all, as opposed to expanding it back into
/// scalar FMAs. The remaining fields are per shape and precision, because a
/// target can lower F16 M16N8K16 and still have no TF32 form.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct MmaCapabilityRecord {
    pub(crate) descriptor_mma: bool,
    pub(crate) f16_m16n8k16: bool,
    pub(crate) bf16_m16n8k16: bool,
    pub(crate) tf32_m16n8k4: bool,
}

impl MmaCapabilityRecord {
    /// A target that lowers descriptor MMA at every shape this module emits.
    pub(crate) const fn all_descriptor_mma_shapes() -> Self {
        Self {
            descriptor_mma: true,
            f16_m16n8k16: true,
            bf16_m16n8k16: true,
            tf32_m16n8k4: true,
        }
    }

    /// A target with no descriptor-level MMA. Every tensor-core path on such a
    /// target lowers to the cooperative body instead.
    pub(crate) const fn no_descriptor_mma() -> Self {
        Self {
            descriptor_mma: false,
            f16_m16n8k16: false,
            bf16_m16n8k16: false,
            tf32_m16n8k4: false,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum MmaFallbackDiagnostic {
    CooperativePath,
    TargetDoesNotLowerDescriptorMma,
    MissingInstructionClass { path: MatmulKernelPath },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct MmaCapabilityGate {
    pub(crate) selected_path: MatmulKernelPath,
    pub(crate) instruction_class: Option<MmaInstructionClass>,
    pub(crate) fallback_diagnostic: Option<MmaFallbackDiagnostic>,
}

pub(crate) fn gate_mma_path(
    path: MatmulKernelPath,
    capabilities: MmaCapabilityRecord,
) -> MmaCapabilityGate {
    let Some(instruction_class) = instruction_class_for(path, capabilities) else {
        return MmaCapabilityGate {
            selected_path: MatmulKernelPath::Cooperative,
            instruction_class: None,
            fallback_diagnostic: Some(if path == MatmulKernelPath::Cooperative {
                MmaFallbackDiagnostic::CooperativePath
            } else if capabilities.descriptor_mma {
                MmaFallbackDiagnostic::MissingInstructionClass { path }
            } else {
                MmaFallbackDiagnostic::TargetDoesNotLowerDescriptorMma
            }),
        };
    };

    MmaCapabilityGate {
        selected_path: path,
        instruction_class: Some(instruction_class),
        fallback_diagnostic: None,
    }
}

fn instruction_class_for(
    path: MatmulKernelPath,
    capabilities: MmaCapabilityRecord,
) -> Option<MmaInstructionClass> {
    if !capabilities.descriptor_mma {
        return None;
    }

    match path {
        MatmulKernelPath::TensorCoreF16M16N8K16 if capabilities.f16_m16n8k16 => {
            Some(MmaInstructionClass::DescriptorMmaM16N8K16F16)
        }
        MatmulKernelPath::TensorCoreBf16M16N8K16 if capabilities.bf16_m16n8k16 => {
            Some(MmaInstructionClass::DescriptorMmaM16N8K16Bf16)
        }
        MatmulKernelPath::TensorCoreTf32M16N8K4 if capabilities.tf32_m16n8k4 => {
            Some(MmaInstructionClass::DescriptorMmaM16N8K4Tf32)
        }
        _ => None,
    }
}

/// Build the Program IR for one M16N8K16 matrix-multiply-accumulate
/// fragment using the exact FMA sequence that B6 promotes to
/// `KernelOpKind::MatrixMma`.
///
/// The operand cycling matches the promotable pattern:
///
/// ```text
/// c0' = fma(a0, b0, c0)
/// c1' = fma(a1, b1, c1)
/// c2' = fma(a2, b0, c2)
/// c3' = fma(a3, b1, c3)
///
/// # Returns
///
/// A `Vec<Node>` containing four `Node::Let` bindings in order:
/// `mma_c0`, `mma_c1`, `mma_c2`, `mma_c3`.  When this sequence is
/// lowered to a `KernelDescriptor` and run through `matmul_promote`,
/// the four contiguous `Fma` ops collapse into a single
/// `MatrixMma { M16N8K16, RowMajor, ColMajor, F16, F16, F32 }`.
#[must_use]
pub(crate) fn matmul_mma_fragment(
    a0: Expr,
    a1: Expr,
    a2: Expr,
    a3: Expr,
    b0: Expr,
    b1: Expr,
    c0: Expr,
    c1: Expr,
    c2: Expr,
    c3: Expr,
) -> Vec<Node> {
    vec![
        Node::let_bind("mma_c0", Expr::fma(a0, b0.clone(), c0)),
        Node::let_bind("mma_c1", Expr::fma(a1, b1.clone(), c1)),
        Node::let_bind("mma_c2", Expr::fma(a2, b0, c2)),
        Node::let_bind("mma_c3", Expr::fma(a3, b1, c3)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{Expr, Node};

    #[test]
    fn matmul_mma_fragment_builds_four_fma_nodes() {
        let nodes = matmul_mma_fragment(
            Expr::f32(1.0),
            Expr::f32(2.0),
            Expr::f32(3.0),
            Expr::f32(4.0),
            Expr::f32(5.0),
            Expr::f32(6.0),
            Expr::f32(7.0),
            Expr::f32(8.0),
            Expr::f32(9.0),
            Expr::f32(10.0),
        );
        assert_eq!(nodes.len(), 4);
        for node in &nodes {
            assert!(
                matches!(
                    node,
                    Node::Let {
                        value: Expr::Fma { .. },
                        ..
                    }
                ),
                "each node must be a Let binding an Expr::fma"
            );
        }
    }

    #[test]
    fn matmul_mma_fragment_operand_cycling_matches_b6_contract() {
        // Verify the exact operand pattern:
        // fma(a0, b0, c0), fma(a1, b1, c1), fma(a2, b0, c2), fma(a3, b1, c3)
        let nodes = matmul_mma_fragment(
            Expr::var("a0"),
            Expr::var("a1"),
            Expr::var("a2"),
            Expr::var("a3"),
            Expr::var("b0"),
            Expr::var("b1"),
            Expr::var("c0"),
            Expr::var("c1"),
            Expr::var("c2"),
            Expr::var("c3"),
        );

        let extract_operands = |node: &Node| -> (String, String, String) {
            match node {
                Node::Let {
                    value: Expr::Fma { a, b, c },
                    ..
                } => (format!("{a:?}"), format!("{b:?}"), format!("{c:?}")),
                _ => panic!("expected Let binding an Fma"),
            }
        };

        let op0 = extract_operands(&nodes[0]);
        let op1 = extract_operands(&nodes[1]);
        let op2 = extract_operands(&nodes[2]);
        let op3 = extract_operands(&nodes[3]);

        assert!(op0.0.contains("a0") && op0.1.contains("b0") && op0.2.contains("c0"));
        assert!(op1.0.contains("a1") && op1.1.contains("b1") && op1.2.contains("c1"));
        assert!(op2.0.contains("a2") && op2.1.contains("b0") && op2.2.contains("c2"));
        assert!(op3.0.contains("a3") && op3.1.contains("b1") && op3.2.contains("c3"));
    }

    #[test]
    fn mma_capability_gate_accepts_descriptor_mma_and_reports_instruction_class() {
        let gate = gate_mma_path(
            MatmulKernelPath::TensorCoreF16M16N8K16,
            MmaCapabilityRecord::all_descriptor_mma_shapes(),
        );

        assert_eq!(gate.selected_path, MatmulKernelPath::TensorCoreF16M16N8K16);
        assert_eq!(
            gate.instruction_class,
            Some(MmaInstructionClass::DescriptorMmaM16N8K16F16)
        );
        assert_eq!(gate.fallback_diagnostic, None);
    }

    #[test]
    fn mma_capability_gate_falls_back_without_descriptor_mma() {
        let gate = gate_mma_path(
            MatmulKernelPath::TensorCoreF16M16N8K16,
            MmaCapabilityRecord::no_descriptor_mma(),
        );

        assert_eq!(gate.selected_path, MatmulKernelPath::Cooperative);
        assert_eq!(gate.instruction_class, None);
        assert_eq!(
            gate.fallback_diagnostic,
            Some(MmaFallbackDiagnostic::TargetDoesNotLowerDescriptorMma)
        );
    }

    /// The boundary the removed backend enum could not express: a target that
    /// lowers descriptor MMA but has no form for the requested precision. The
    /// old gate keyed on the backend identity, so it reported "this backend
    /// does not lower descriptor MMA" for a target that plainly does.
    #[test]
    fn descriptor_mma_without_the_requested_precision_reports_the_missing_class() {
        let partial = MmaCapabilityRecord {
            descriptor_mma: true,
            f16_m16n8k16: true,
            bf16_m16n8k16: false,
            tf32_m16n8k4: false,
        };

        let gate = gate_mma_path(MatmulKernelPath::TensorCoreBf16M16N8K16, partial);

        assert_eq!(gate.selected_path, MatmulKernelPath::Cooperative);
        assert_eq!(gate.instruction_class, None);
        assert_eq!(
            gate.fallback_diagnostic,
            Some(MmaFallbackDiagnostic::MissingInstructionClass {
                path: MatmulKernelPath::TensorCoreBf16M16N8K16
            })
        );

        assert_eq!(
            gate_mma_path(MatmulKernelPath::TensorCoreF16M16N8K16, partial).instruction_class,
            Some(MmaInstructionClass::DescriptorMmaM16N8K16F16),
            "Fix: a per-shape gap must not disable the shapes the target does lower."
        );
    }
}
