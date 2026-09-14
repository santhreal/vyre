//! Contraction candidate detection and strategy planning.
//!
//! Logical contraction IR lowers into scalar, SIMD/SIMT tiled, and target
//! matrix-instruction candidates without domain ownership. Every candidate is
//! derived from the contraction site the descriptor states: its element types,
//! its declared workgroup geometry, and the matrix tile and fragment element
//! types a `MatrixMma` op in the body declares. This analysis never invents a
//! tile extent, an element type, or a ranking a site does not state.
//!
//! Candidates carry counted work, not a measured time. `operand_loads_per_fma`
//! is the operand elements one multiply-accumulate reads under the strategy,
//! which is what the reuse a tile buys is a ratio of. Selection among the
//! candidates is a measured decision made above this analysis; nothing here
//! claims a device outcome.

use super::plan::{
    ContractionCandidate, ContractionPlan, ContractionStrategy, MatrixInstructionSource,
};
use crate::descriptor::{
    BindingVisibility, KernelBody, KernelDescriptor, KernelOpKind, MatrixMmaElement, MatrixMmaSpec,
    MatrixTileShape,
};
use std::collections::BTreeSet;
use vyre_foundation::ir::{BinOp, DataType};

/// Facts a descriptor states about the contraction it carries.
struct ContractionSite {
    /// The matrix specification the body declares, when it declares one.
    declared: Option<MatrixMmaSpec>,
    /// Multiply-accumulate present as an explicit `Fma` op.
    has_fma: bool,
    /// A multiply whose result is summed, which is one multiply-accumulate of
    /// a contraction the lowering unrolled or carried in a loop.
    has_multiply_accumulate: bool,
}

impl ContractionSite {
    fn is_contraction(&self) -> bool {
        self.declared.is_some() || self.has_fma || self.has_multiply_accumulate
    }
}

/// Analyze a kernel descriptor and surface contraction candidate execution strategies.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> ContractionPlan {
    let site = scan_body(&desc.body);
    let mut candidates = Vec::new();

    if site.is_contraction() {
        let dtypes = bound_element_types(desc);

        // The scalar baseline reads both operand elements of every
        // multiply-accumulate from memory: no reuse, one output per
        // invocation.
        candidates.push(ContractionCandidate {
            contraction_id: format!("{}_scalar", desc.id),
            strategy: ContractionStrategy::Scalar,
            operand_loads_per_fma: 2.0,
            supported_dtypes: dtypes.clone(),
            derivation: "scalar baseline reads both operands per multiply-accumulate".to_owned(),
        });

        if let Some(tiled) = simt_tiled_candidate(desc, &dtypes) {
            candidates.push(tiled);
        }

        if let Some(matrix) = matrix_instruction_candidate(desc, &site, &dtypes) {
            candidates.push(matrix);
        }
    }

    ContractionPlan {
        kernel_id: desc.id.clone(),
        candidates,
    }
}

/// Tile the declared workgroup geometry cooperates over.
///
/// A workgroup of one invocation cannot stage a shared tile, so it states no
/// tiled candidate. The tile is the workgroup's own x and y extents: those are
/// the invocations that exist to load it, and a tile wider than the workgroup
/// would be loaded by invocations the dispatch does not have.
fn simt_tiled_candidate(
    desc: &KernelDescriptor,
    dtypes: &[DataType],
) -> Option<ContractionCandidate> {
    let [x, y, z] = desc.dispatch.workgroup_size;
    let invocations = u64::from(x) * u64::from(y) * u64::from(z);
    if invocations < 2 {
        return None;
    }

    let tile_m = x.max(1);
    let tile_n = y.max(1);
    // The staged reduction step is the tile the same invocations can fill in
    // one cooperative load, which is the invocation count spread over the
    // tile's rows.
    let tile_k = u32::try_from(invocations / u64::from(tile_m))
        .unwrap_or(1)
        .max(1);

    // Each staged operand element is read once from memory and reused by the
    // other tile extent: `tile_n` reuses of a left element, `tile_m` of a
    // right element.
    let operand_loads_per_fma = 1.0 / f64::from(tile_n) as f32 + 1.0 / f64::from(tile_m) as f32;

    Some(ContractionCandidate {
        contraction_id: format!("{}_simt_tiled_{tile_m}x{tile_n}x{tile_k}", desc.id),
        strategy: ContractionStrategy::SimtTiled {
            tile_m,
            tile_n,
            tile_k,
            workgroup_size: desc.dispatch.workgroup_size,
        },
        operand_loads_per_fma,
        supported_dtypes: dtypes.to_vec(),
        derivation: format!(
            "tile extents are the declared workgroup geometry {x}x{y}x{z}; each staged operand element is reused {} and {} times",
            tile_n, tile_m
        ),
    })
}

/// Matrix-instruction candidate, stated only by a site that declares one.
///
/// The tile extents, fragment layouts and fragment element types are the ones
/// the declared `MatrixMma` carries. A site that declares no matrix operation
/// states no matrix candidate: the extents a target would need are not
/// derivable from a scalar reduction loop, and inventing them would claim a
/// packing the descriptor never expressed.
fn matrix_instruction_candidate(
    desc: &KernelDescriptor,
    site: &ContractionSite,
    dtypes: &[DataType],
) -> Option<ContractionCandidate> {
    let spec = site.declared?;
    let MatrixTileShape { m, n, k } = spec.tile;
    if m == 0 || n == 0 || k == 0 {
        return None;
    }

    // The fragment element types the declared operation accepts, intersected
    // with the element types the kernel actually binds. A dtype the site never
    // binds is not supported by this candidate.
    let fragment_dtypes: Vec<DataType> = dtypes
        .iter()
        .filter(|dtype| matrix_element_of(dtype) == Some(spec.left.element))
        .cloned()
        .collect();
    let supported_dtypes = if fragment_dtypes.is_empty() {
        matrix_element_dtype(spec.left.element)
            .into_iter()
            .collect()
    } else {
        fragment_dtypes
    };

    // One fragment instruction issues `m * n * k` multiply-accumulates and
    // reads `m * k` left plus `k * n` right elements once each.
    let fmas = f32::from(m) * f32::from(n) * f32::from(k);
    let loads = f32::from(m) * f32::from(k) + f32::from(k) * f32::from(n);

    Some(ContractionCandidate {
        contraction_id: format!("{}_mma_m{m}n{n}k{k}", desc.id),
        strategy: ContractionStrategy::MatrixInstruction {
            tile: spec.tile,
            left_layout: spec.left.layout,
            right_layout: spec.right.layout,
            left_element: spec.left.element,
            right_element: spec.right.element,
            acc_element: spec.accumulator.element,
            source: MatrixInstructionSource::DeclaredByDescriptor,
        },
        operand_loads_per_fma: loads / fmas,
        supported_dtypes,
        derivation: format!(
            "extents m{m} n{n} k{k} and fragment element types are the ones the descriptor's matrix operation declares"
        ),
    })
}

/// Element types the kernel's readable bindings carry, in slot order, once each.
fn bound_element_types(desc: &KernelDescriptor) -> Vec<DataType> {
    let mut out: Vec<DataType> = Vec::new();
    for slot in &desc.bindings.slots {
        if slot.visibility == BindingVisibility::WriteOnly {
            continue;
        }
        if !out.contains(&slot.element_type) {
            out.push(slot.element_type.clone());
        }
    }
    out
}

fn matrix_element_of(dtype: &DataType) -> Option<MatrixMmaElement> {
    match dtype {
        DataType::F16 => Some(MatrixMmaElement::F16),
        DataType::BF16 => Some(MatrixMmaElement::BF16),
        DataType::F32 => Some(MatrixMmaElement::F32),
        _ => None,
    }
}

fn matrix_element_dtype(element: MatrixMmaElement) -> Option<DataType> {
    match element {
        MatrixMmaElement::F16 => Some(DataType::F16),
        MatrixMmaElement::BF16 => Some(DataType::BF16),
        MatrixMmaElement::F32 | MatrixMmaElement::TF32 => Some(DataType::F32),
    }
}

/// Read the contraction facts out of a lowered body.
///
/// A contraction is a sum of products, so the detection is that dataflow and
/// not a proxy for it: an add whose operand is the result of a multiply. A
/// lowering that unrolls the reduction, carries it in a loop, or issues it as
/// one `Fma` all state the same site, and a kernel that merely binds a large
/// buffer states none.
fn scan_body(body: &KernelBody) -> ContractionSite {
    let mut site = ContractionSite {
        declared: None,
        has_fma: false,
        has_multiply_accumulate: false,
    };
    let mut products = BTreeSet::new();
    scan_into(body, &mut site, &mut products);
    site
}

fn scan_into(body: &KernelBody, site: &mut ContractionSite, products: &mut BTreeSet<u32>) {
    for op in &body.ops {
        match &op.kind {
            KernelOpKind::MatrixMma(spec) => {
                if site.declared.is_none() {
                    site.declared = Some(**spec);
                }
            }
            KernelOpKind::Fma => site.has_fma = true,
            KernelOpKind::BinOpKind(BinOp::Mul) => {
                if let Some(result) = op.result {
                    products.insert(result);
                }
            }
            KernelOpKind::BinOpKind(BinOp::Add) => {
                if op.operands.iter().any(|id| products.contains(id)) {
                    site.has_multiply_accumulate = true;
                }
            }
            _ => {}
        }
    }

    for child in &body.child_bodies {
        scan_into(child, site, products);
    }
}
