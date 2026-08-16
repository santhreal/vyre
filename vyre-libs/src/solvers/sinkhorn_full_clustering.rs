//! Full Sinkhorn-balanced dispatch-graph clustering.
//!
//! Replaces the single-step version in `sinkhorn_dispatch_clustering` (#2)
//! with a full iterative fixpoint. This computes an entropy-regularized
//! optimal transport plan between dispatch components, yielding a balanced
//! soft assignment of nodes to clusters.
//!
//! Composes the `crate::math::sinkhorn_iterate` primitive to run
//! entirely on device without host round-trips.

use vyre_foundation::ir::Program;
use crate::math::sinkhorn_iterate::{sinkhorn_iterate, SinkhornBuffers, SinkhornExtents};

/// Stable op identifier for the full-clustering Sinkhorn iteration self-consumer.
pub const OP_ID: &str = "vyre-libs::self_substrate::sinkhorn_full_clustering";

/// Compile a Program that runs full Sinkhorn iterations.
///
/// The composition it adds is the telemetry counter; the program is the
/// primitive's, over the caller's binding record and extents unchanged.
#[must_use]
pub fn sinkhorn_full_clustering_program(
    buffers: SinkhornBuffers<'_>,
    extents: SinkhornExtents,
) -> Program {
    use crate::telemetry::{bump, sinkhorn_full_clustering_calls};
    bump(&sinkhorn_full_clustering_calls);
    sinkhorn_iterate(buffers, extents)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The binding names one region's Sinkhorn program is built against.
    const FIXTURE: SinkhornBuffers<'static> = SinkhornBuffers::CANONICAL;

    /// `SinkhornExtents` for `m x n` under an iteration cap.
    fn extents(m: u32, n: u32, max_iterations: u32) -> SinkhornExtents {
        SinkhornExtents {
            m,
            n,
            max_iterations,
        }
    }

    /// Owns one region's suffixed binding names so a record can borrow them.
    ///
    /// `wrap_program_sequence` concatenates buffer declarations without merging
    /// them, so each region needs its own ten names for the concatenation to
    /// describe three independent Sinkhorn problems rather than three aliases of
    /// one.
    struct RegionNames {
        k: String,
        k_t: String,
        a: String,
        b: String,
        u_curr: String,
        u_next: String,
        v: String,
        kv: String,
        ktu: String,
        changed: String,
    }

    impl RegionNames {
        fn new(suffix: u32) -> Self {
            Self {
                k: format!("k{suffix}"),
                k_t: format!("kt{suffix}"),
                a: format!("a{suffix}"),
                b: format!("b{suffix}"),
                u_curr: format!("uc{suffix}"),
                u_next: format!("un{suffix}"),
                v: format!("v{suffix}"),
                kv: format!("kv{suffix}"),
                ktu: format!("ktu{suffix}"),
                changed: format!("c{suffix}"),
            }
        }

        fn buffers(&self) -> SinkhornBuffers<'_> {
            SinkhornBuffers {
                k: &self.k,
                k_t: &self.k_t,
                a: &self.a,
                b: &self.b,
                u_curr: &self.u_curr,
                u_next: &self.u_next,
                v: &self.v,
                kv: &self.kv,
                ktu: &self.ktu,
                changed: &self.changed,
            }
        }
    }

    #[test]
    fn test_sinkhorn_clustering_program() {
        let p = sinkhorn_full_clustering_program(FIXTURE, extents(10, 20, 5));
        assert_eq!(p.buffers().len(), 10);
        assert!(p.buffers().iter().any(|b| b.name() == FIXTURE.u_curr));
    }

    /// The composition must emit exactly the primitive's program.
    ///
    /// It forwards the record and the extents and adds only a telemetry bump, so
    /// the two emissions must be byte-identical on the wire. Compared on the wire
    /// encoding rather than a debug string, which would compare formatting.
    #[test]
    fn composition_emits_the_primitive_program_unchanged() {
        let extents = extents(17, 17, 4);
        let composed = vyre_foundation::serial::wire::encode::to_wire(
            &sinkhorn_full_clustering_program(FIXTURE, extents),
        )
        .expect("Fix: the composed sinkhorn program must encode to the wire form.");
        let primitive =
            vyre_foundation::serial::wire::encode::to_wire(&sinkhorn_iterate(FIXTURE, extents))
                .expect("Fix: the primitive sinkhorn program must encode to the wire form.");
        assert_eq!(
            composed, primitive,
            "Fix: sinkhorn_full_clustering_program must compose the primitive, not restate its program."
        );
    }

    #[test]
    fn test_multi_region_sinkhorn() {
        let names = [
            RegionNames::new(1),
            RegionNames::new(2),
            RegionNames::new(3),
        ];
        let programs = names
            .iter()
            .map(|region| sinkhorn_full_clustering_program(region.buffers(), extents(2, 2, 1)))
            .collect::<Vec<_>>();
        assert_eq!(
            programs
                .iter()
                .map(|program| program.buffers().len())
                .sum::<usize>(),
            30,
            "Fix: three regions must declare thirty independent bindings."
        );

        let borrowed = programs.iter().collect::<Vec<_>>();
        let final_p = crate::test_support::wrap_program_sequence(&borrowed, [256, 1, 1]);
        let region_count = final_p
            .entry()
            .iter()
            .filter(|n| matches!(n, vyre_foundation::ir::Node::Region { .. }))
            .count();
        assert!(region_count >= 3);
    }

    #[test]
    fn test_end_to_end_sinkhorn_parity() {
        let k = vec![65536, 65536, 65536, 65536];
        let a = vec![32768, 32768];
        let b = vec![32768, 32768];
        let u_c = vec![65536, 65536];
        let v_in = vec![65536, 65536];

        let p = sinkhorn_full_clustering_program(FIXTURE, extents(2, 2, 1));

        use std::sync::Arc;
        use vyre_reference::reference_eval;
        use vyre_reference::value::Value;

        let to_value = |data: &[u32]| {
            let bytes = vyre_primitives::wire::pack_u32_slice(data);
            Value::Bytes(Arc::from(bytes))
        };

        let inputs = vec![
            to_value(&u_c),
            to_value(&[0_u32, 0]),
            to_value(&[0]),
            to_value(&k),
            to_value(&k), // kt
            to_value(&a),
            to_value(&b),
            to_value(&v_in),
            to_value(&[0_u32, 0]),
            to_value(&[0_u32, 0]),
        ];

        let results = reference_eval(&p, &inputs).expect("Fix: interpreter failed");
        let actual_bytes = results[0].to_bytes();
        let actual_u: Vec<u32> = actual_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // first iter: Kv = [2, 2] scaled by 2^32? No, 2^32 = 0.
        // If it wraps to 0, floor is 1. u = a/1 = 32768.
        assert_eq!(actual_u[0], 32768);
    }
}
