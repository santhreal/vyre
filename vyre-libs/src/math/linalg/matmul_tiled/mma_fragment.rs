//! Tensor-core MMA fragment primitive for M16N8K16.
//!
//! Emits the exact 4-FMA sequence that B6 (`matmul_promote`) detects
//! and collapses into `KernelOpKind::MatrixMma`.

use vyre_foundation::ir::{Expr, Node};

use super::tensor_core_policy::MatmulKernelPath;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum MmaBackendKind {
    Ptx,
    Metal,
    Wgpu,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum MmaInstructionClass {
    PtxMmaSyncAlignedM16N8K16F16,
    PtxMmaSyncAlignedM16N8K16Bf16,
    PtxMmaSyncAlignedM16N8K4Tf32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct MmaCapabilityRecord {
    pub(crate) backend: MmaBackendKind,
    pub(crate) descriptor_mma: bool,
    pub(crate) f16_m16n8k16: bool,
    pub(crate) bf16_m16n8k16: bool,
    pub(crate) tf32_m16n8k4: bool,
}

impl MmaCapabilityRecord {
    pub(crate) const fn ptx_sm80() -> Self {
        Self {
            backend: MmaBackendKind::Ptx,
            descriptor_mma: true,
            f16_m16n8k16: true,
            bf16_m16n8k16: true,
            tf32_m16n8k4: true,
        }
    }

    pub(crate) const fn metal() -> Self {
        Self {
            backend: MmaBackendKind::Metal,
            descriptor_mma: false,
            f16_m16n8k16: false,
            bf16_m16n8k16: false,
            tf32_m16n8k4: false,
        }
    }

    pub(crate) const fn wgpu() -> Self {
        Self {
            backend: MmaBackendKind::Wgpu,
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
    BackendDoesNotLowerDescriptorMma {
        backend: MmaBackendKind,
    },
    MissingInstructionClass {
        backend: MmaBackendKind,
        path: MatmulKernelPath,
    },
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
            } else if !capabilities.descriptor_mma {
                MmaFallbackDiagnostic::BackendDoesNotLowerDescriptorMma {
                    backend: capabilities.backend,
                }
            } else {
                MmaFallbackDiagnostic::MissingInstructionClass {
                    backend: capabilities.backend,
                    path,
                }
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
    if capabilities.backend != MmaBackendKind::Ptx || !capabilities.descriptor_mma {
        return None;
    }

    match path {
        MatmulKernelPath::TensorCoreF16M16N8K16 if capabilities.f16_m16n8k16 => {
            Some(MmaInstructionClass::PtxMmaSyncAlignedM16N8K16F16)
        }
        MatmulKernelPath::TensorCoreBf16M16N8K16 if capabilities.bf16_m16n8k16 => {
            Some(MmaInstructionClass::PtxMmaSyncAlignedM16N8K16Bf16)
        }
        MatmulKernelPath::TensorCoreTf32M16N8K4 if capabilities.tf32_m16n8k4 => {
            Some(MmaInstructionClass::PtxMmaSyncAlignedM16N8K4Tf32)
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
    fn mma_capability_gate_accepts_ptx_and_reports_instruction_class() {
        let gate = gate_mma_path(
            MatmulKernelPath::TensorCoreF16M16N8K16,
            MmaCapabilityRecord::ptx_sm80(),
        );

        assert_eq!(gate.selected_path, MatmulKernelPath::TensorCoreF16M16N8K16);
        assert_eq!(
            gate.instruction_class,
            Some(MmaInstructionClass::PtxMmaSyncAlignedM16N8K16F16)
        );
        assert_eq!(gate.fallback_diagnostic, None);
    }

    #[test]
    fn mma_capability_gate_falls_back_on_unsupported_backends() {
        for capabilities in [MmaCapabilityRecord::metal(), MmaCapabilityRecord::wgpu()] {
            let gate = gate_mma_path(MatmulKernelPath::TensorCoreF16M16N8K16, capabilities);

            assert_eq!(gate.selected_path, MatmulKernelPath::Cooperative);
            assert_eq!(gate.instruction_class, None);
            assert_eq!(
                gate.fallback_diagnostic,
                Some(MmaFallbackDiagnostic::BackendDoesNotLowerDescriptorMma {
                    backend: capabilities.backend,
                })
            );
        }
    }
}
