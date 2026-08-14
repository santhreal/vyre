//! Every builder that delegates to a second call convention must build the same
//! program as the convention it delegates to.
//!
//! WHY: five builders in this crate exist only as an alternate way to call a
//! builder that the registry already sweeps. Two take an explicit op id so a
//! composition crate can keep its own registry identity while reusing the
//! primitive's IR; three take a params struct so a fourteen-argument builder has
//! one named-field call site. Neither form is registered, and every call to them
//! is in production code, so nothing compared the two arms: a wrapper that stops
//! forwarding a field, or a params struct whose field lands in the wrong slot,
//! produces a program that still builds, still dispatches, and reads a different
//! buffer. Each case here builds both arms and compares them structurally, then
//! perturbs one input to show the comparison can fail.
//!
//! What this does not catch: whether either arm computes the right answer. The
//! per-domain parity targets own that; this one owns the claim that the two arms
//! are the same program.

#![forbid(unsafe_code)]

#[cfg(feature = "math")]
mod op_id_forms_math {
    use vyre_primitives::math::prefix_scan::{
        prefix_scan, prefix_scan_with_op_id, ScanKind, OP_ID_EXCLUSIVE_SUM, OP_ID_INCLUSIVE_SUM,
    };

    #[test]
    fn prefix_scan_delegates_to_the_op_id_form_for_both_kinds() {
        for (kind, op_id) in [
            (ScanKind::InclusiveSum, OP_ID_INCLUSIVE_SUM),
            (ScanKind::ExclusiveSum, OP_ID_EXCLUSIVE_SUM),
        ] {
            let wrapper = prefix_scan("in", "out", 8, kind);
            let explicit = prefix_scan_with_op_id("in", "out", 8, kind, op_id);
            assert!(
                wrapper.structural_eq(&explicit),
                "Fix: prefix_scan must build the same program as prefix_scan_with_op_id for \
                 {kind:?} with op id {op_id}. The wrapper only selects the id."
            );
        }
    }

    #[test]
    fn the_op_id_reaches_the_program_identity() {
        let inclusive =
            prefix_scan_with_op_id("in", "out", 8, ScanKind::InclusiveSum, "caller::id");
        let canonical = prefix_scan("in", "out", 8, ScanKind::InclusiveSum);
        assert!(
            !inclusive.structural_eq(&canonical),
            "Fix: the op id a composition crate supplies must reach the program identity, \
             otherwise the op-id form has no purpose and the equality above proves nothing."
        );
    }
}

#[cfg(feature = "decode")]
mod op_id_forms_decode {
    use vyre_primitives::decode::ziftsieve::{
        ziftsieve_literal_copy, ziftsieve_literal_copy_with_op_id, OP_ID,
    };

    #[test]
    fn ziftsieve_literal_copy_delegates_to_the_op_id_form() {
        let wrapper = ziftsieve_literal_copy("in", "out", "start", "len", "offset", 256, 4, 512);
        let explicit = ziftsieve_literal_copy_with_op_id(
            OP_ID, "in", "out", "start", "len", "offset", 256, 4, 512,
        );
        assert!(
            wrapper.structural_eq(&explicit),
            "Fix: ziftsieve_literal_copy must build the same program as \
             ziftsieve_literal_copy_with_op_id under the canonical op id."
        );
    }

    #[test]
    fn the_op_id_reaches_the_program_identity() {
        let caller_id = ziftsieve_literal_copy_with_op_id(
            "vyre-libs::decode::ziftsieve",
            "in",
            "out",
            "start",
            "len",
            "offset",
            256,
            4,
            512,
        );
        let canonical = ziftsieve_literal_copy("in", "out", "start", "len", "offset", 256, 4, 512);
        assert!(
            !caller_id.structural_eq(&canonical),
            "Fix: the op id a composition crate supplies must reach the program identity, \
             otherwise the op-id form has no purpose and the equality above proves nothing."
        );
    }
}

#[cfg(feature = "graph")]
mod params_struct_forms {
    use vyre_primitives::graph::csr_queue_delta::{
        csr_queue_delta_enqueue, csr_queue_delta_enqueue_with, csr_queue_delta_strided_enqueue,
        csr_queue_delta_strided_enqueue_with, CsrQueueDeltaEnqueueParams,
    };
    use vyre_primitives::graph::csr_queue_split::{
        csr_queue_split_low_forward_traverse, csr_queue_split_low_forward_traverse_with,
        CsrQueueSplitLowForwardParams,
    };

    const NODES: u32 = 64;
    const EDGES: u32 = 128;
    const QUEUE_CAP: u32 = 32;
    const NEXT_CAP: u32 = 48;
    const ALLOW: u32 = 0b11;

    fn delta_params() -> CsrQueueDeltaEnqueueParams<'static> {
        CsrQueueDeltaEnqueueParams {
            active_queue: "active_queue",
            active_len: "active_len",
            edge_offsets: "edge_offsets",
            edge_targets: "edge_targets",
            edge_kind_mask: "edge_kind_mask",
            accumulator: "accumulator",
            next_queue: "next_queue",
            next_len: "next_len",
            node_count: NODES,
            edge_count: EDGES,
            active_queue_capacity: QUEUE_CAP,
            next_queue_capacity: NEXT_CAP,
            allow_mask: ALLOW,
        }
    }

    fn split_params() -> CsrQueueSplitLowForwardParams<'static> {
        CsrQueueSplitLowForwardParams {
            active_queue: "active_queue",
            queue_len: "queue_len",
            edge_offsets: "edge_offsets",
            edge_targets: "edge_targets",
            edge_kind_mask: "edge_kind_mask",
            frontier_out: "frontier_out",
            high_queue: "high_queue",
            high_len: "high_len",
            node_count: NODES,
            edge_count: EDGES,
            queue_capacity: QUEUE_CAP,
            high_queue_capacity: NEXT_CAP,
            high_degree_threshold: 32,
            allow_mask: ALLOW,
        }
    }

    #[test]
    fn csr_queue_delta_enqueue_delegates_to_the_params_form() {
        let params = delta_params();
        let wrapper = csr_queue_delta_enqueue(
            params.active_queue,
            params.active_len,
            params.edge_offsets,
            params.edge_targets,
            params.edge_kind_mask,
            params.accumulator,
            params.next_queue,
            params.next_len,
            params.node_count,
            params.edge_count,
            params.active_queue_capacity,
            params.next_queue_capacity,
            params.allow_mask,
        );
        assert!(
            wrapper.structural_eq(&csr_queue_delta_enqueue_with(params)),
            "Fix: csr_queue_delta_enqueue must build the same program as \
             csr_queue_delta_enqueue_with over the same inputs."
        );
    }

    #[test]
    fn csr_queue_delta_strided_enqueue_delegates_to_the_params_form() {
        let params = delta_params();
        let wrapper = csr_queue_delta_strided_enqueue(
            params.active_queue,
            params.active_len,
            params.edge_offsets,
            params.edge_targets,
            params.edge_kind_mask,
            params.accumulator,
            params.next_queue,
            params.next_len,
            params.node_count,
            params.edge_count,
            params.active_queue_capacity,
            params.next_queue_capacity,
            params.allow_mask,
        );
        assert!(
            wrapper.structural_eq(&csr_queue_delta_strided_enqueue_with(params)),
            "Fix: csr_queue_delta_strided_enqueue must build the same program as \
             csr_queue_delta_strided_enqueue_with over the same inputs."
        );
    }

    #[test]
    fn csr_queue_split_low_forward_traverse_delegates_to_the_params_form() {
        let params = split_params();
        let wrapper = csr_queue_split_low_forward_traverse(
            params.active_queue,
            params.queue_len,
            params.edge_offsets,
            params.edge_targets,
            params.edge_kind_mask,
            params.frontier_out,
            params.high_queue,
            params.high_len,
            params.node_count,
            params.edge_count,
            params.queue_capacity,
            params.high_queue_capacity,
            params.high_degree_threshold,
            params.allow_mask,
        );
        assert!(
            wrapper.structural_eq(&csr_queue_split_low_forward_traverse_with(params)),
            "Fix: csr_queue_split_low_forward_traverse must build the same program as \
             csr_queue_split_low_forward_traverse_with over the same inputs."
        );
    }

    #[test]
    fn a_params_field_reaches_the_program_it_names() {
        let canonical = csr_queue_delta_enqueue_with(delta_params());
        let swapped = csr_queue_delta_enqueue_with(CsrQueueDeltaEnqueueParams {
            accumulator: "next_queue",
            next_queue: "accumulator",
            ..delta_params()
        });
        assert!(
            !canonical.structural_eq(&swapped),
            "Fix: swapping the accumulator and next_queue names must change the program. If it \
             does not, the structural comparisons above would hold for a params struct whose \
             fields land in the wrong slots."
        );

        let canonical_split = csr_queue_split_low_forward_traverse_with(split_params());
        let swapped_split =
            csr_queue_split_low_forward_traverse_with(CsrQueueSplitLowForwardParams {
                frontier_out: "high_queue",
                high_queue: "frontier_out",
                ..split_params()
            });
        assert!(
            !canonical_split.structural_eq(&swapped_split),
            "Fix: swapping the frontier_out and high_queue names must change the program, for the \
             same reason."
        );
    }
}
