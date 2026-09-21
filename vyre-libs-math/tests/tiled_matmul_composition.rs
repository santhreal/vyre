//! One tiled matmul names the same registered child for every element type it
//! accepts.
//!
//! `matmul_tiled` and `matmul_bias_tiled` each assemble two kernel bodies under
//! one operation id. The tensor-core body is assembled in the operation's own
//! module and names `semiring_gemm` as the child region that carries the
//! accumulation. The cooperative body comes from the shared contraction
//! composer, which closes with a single region carrying no source, so one
//! operation attributed its contraction to the registered primitive on one path
//! and claimed every node as own work on the other. The composition audit reads
//! the second path as a fully uncomposed operation, and its own composed
//! fraction reports a regression that neither path explains on its own.
//!
//! The invariant below is selection-independent: whatever kernel body the
//! element type selects, the set of attributed child regions the program names
//! is the same set, and that set names the semiring GEMM. The element types
//! come from `ScalarFormat::ALL`, a fixed-length array, so a format added to
//! the enum fails to compile until it is listed there and then enters this
//! sweep.
//!
//! What this does not cover: which kernel body a given element type selects.
//! `MmaCapabilityRecord::current_codegen()` admits the tensor-core body only
//! for the capabilities the build was compiled for, so this sweep cannot force
//! that body to be chosen and does not assert that it was. It asserts that no
//! element type reaches a body without the attribution, which is the condition
//! the composer path violated. It also does not cover whether the child region
//! computes what `semiring_gemm` computes; `vyre-libs/tests/cpu_witnesses.rs`
//! pins the value contract of both operations against the reference evaluator.
#![cfg(feature = "math-linalg")]

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::ir::{Node, Program};
use vyre_foundation::numeric::ScalarFormat;
use vyre_foundation::visit::child_bodies;
use vyre_libs_builder::plumbing::operand::tensor_ref::TensorRef;
use vyre_libs_math::math::linalg::{MatmulBiasTiled, MatmulTiled};
use vyre_libs_math::math::semiring_gemm::OP_ID as SEMIRING_GEMM_OP_ID;

/// Square side used for every program built here.
///
/// A multiple of sixteen on all three axes is what leaves the tensor-core body
/// selectable, so a build whose codegen capabilities admit it takes that body
/// through the same sweep.
const SIDE: u32 = 32;

/// Generators of every attributed region in `nodes`, at any depth.
///
/// `source_region` is what states the region carries another operation's work,
/// so it is the whole test for whether a node is attributed or own.
fn child_generators(nodes: &[Node], out: &mut BTreeSet<String>) {
    for node in nodes {
        if let Node::Region {
            generator,
            source_region: Some(_),
            ..
        } = node
        {
            out.insert(generator.as_str().to_string());
        }
        for body in child_bodies(node) {
            child_generators(body, out);
        }
    }
}

fn attributed_children(program: &Program) -> BTreeSet<String> {
    let mut generators = BTreeSet::new();
    child_generators(program.entry(), &mut generators);
    generators
}

fn tensor(name: &str, format: ScalarFormat, shape: Vec<u32>) -> TensorRef {
    TensorRef::new(name, format.data_type(), shape)
}

/// Every element type the plain builder accepts, mapped to the children it names.
fn plain_children_by_format() -> BTreeMap<ScalarFormat, BTreeSet<String>> {
    let mut built = BTreeMap::new();
    for format in ScalarFormat::ALL {
        let program = MatmulTiled::auto(
            tensor("a", format, vec![SIDE, SIDE]),
            tensor("b", format, vec![SIDE, SIDE]),
            tensor("out", format, vec![SIDE, SIDE]),
        )
        .build();
        if let Ok(program) = program {
            built.insert(format, attributed_children(&program));
        }
    }
    built
}

/// Every element type the bias builder accepts, mapped to the children it names.
fn bias_children_by_format() -> BTreeMap<ScalarFormat, BTreeSet<String>> {
    let mut built = BTreeMap::new();
    for format in ScalarFormat::ALL {
        let program = MatmulBiasTiled::auto(
            tensor("a", format, vec![SIDE, SIDE]),
            tensor("b", format, vec![SIDE, SIDE]),
            tensor("bias", format, vec![SIDE]),
            tensor("out", format, vec![SIDE, SIDE]),
        )
        .build();
        if let Ok(program) = program {
            built.insert(format, attributed_children(&program));
        }
    }
    built
}

fn assert_one_attribution_across_formats(
    builder: &str,
    built: &BTreeMap<ScalarFormat, BTreeSet<String>>,
) {
    assert!(
        built.len() > 1,
        "Fix: `{builder}` accepted {} of the {} scalar formats, so this sweep compares nothing. \
         Widen the accepted element types or the shape the sweep builds.",
        built.len(),
        ScalarFormat::ALL.len()
    );
    let (first_format, expected) = built
        .iter()
        .next()
        .expect("a non-empty map yields a first entry");
    for (format, children) in built {
        assert_eq!(
            children, expected,
            "Fix: `{builder}` names child regions {children:?} for {format:?} and {expected:?} for \
             {first_format:?}. One operation must attribute its work to the same registered \
             children whichever kernel body its element type selects."
        );
        assert!(
            children.contains(SEMIRING_GEMM_OP_ID),
            "Fix: `{builder}` does not name `{SEMIRING_GEMM_OP_ID}` as a child region for \
             {format:?}; it names {children:?}. The tiled matmul computes a semiring contraction \
             on every path, so every path states that child."
        );
    }
}

/// WHY: this closes the class where one operation with more than one kernel body
/// attributes its work to a registered child on the body its author had in mind
/// and claims every node as own work on the others. Asserting equality across
/// the whole element-type space, rather than checking the one path that
/// regressed, is what makes a third body fail here instead of passing quietly.
#[test]
fn plain_tiled_matmul_names_the_semiring_gemm_on_every_kernel_path() {
    assert_one_attribution_across_formats("matmul_tiled", &plain_children_by_format());
}

/// WHY: `matmul_bias_tiled` shares `MatmulTiledCore` with the plain operation
/// and therefore shares its path selection, but it is a separate registered
/// operation with its own baseline and its own buffer signature. Covering one
/// and not the other is how the sibling of a fixed defect survives.
#[test]
fn bias_tiled_matmul_names_the_semiring_gemm_on_every_kernel_path() {
    assert_one_attribution_across_formats("matmul_bias_tiled", &bias_children_by_format());
}

/// WHY: the attribution above is a region wrapped around a body that was
/// already correct, so it must not move a single node of that body. A rewrite
/// that reorders or drops work while adding the child region would satisfy both
/// tests above and change what the operation computes.
#[test]
fn attribution_leaves_the_kernel_body_unchanged() {
    for format in ScalarFormat::ALL {
        let Ok(program) = MatmulTiled::auto(
            tensor("a", format, vec![SIDE, SIDE]),
            tensor("b", format, vec![SIDE, SIDE]),
            tensor("out", format, vec![SIDE, SIDE]),
        )
        .build() else {
            continue;
        };
        let entry = program.entry();
        assert_eq!(
            entry.len(),
            1,
            "Fix: a tiled matmul program has one entry region for {format:?}, found {}",
            entry.len()
        );
        let Node::Region {
            source_region,
            body,
            ..
        } = &entry[0]
        else {
            panic!("Fix: the entry of a tiled matmul program is a region, for {format:?}");
        };
        assert!(
            source_region.is_none(),
            "Fix: the entry region of a tiled matmul is the operation's own boundary and carries \
             no source, for {format:?}"
        );
        assert_eq!(
            body.len(),
            1,
            "Fix: the entry region of a tiled matmul holds exactly the attributed child, for \
             {format:?}, found {} nodes",
            body.len()
        );
        let Node::Region {
            generator,
            source_region: Some(_),
            ..
        } = &body[0]
        else {
            panic!(
                "Fix: the single node under the entry region of a tiled matmul is the attributed \
                 child region, for {format:?}"
            );
        };
        assert_eq!(
            generator.as_str(),
            SEMIRING_GEMM_OP_ID,
            "Fix: the attributed child of a tiled matmul is `{SEMIRING_GEMM_OP_ID}`, for {format:?}"
        );
    }
}
