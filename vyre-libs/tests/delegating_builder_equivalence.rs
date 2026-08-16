//! Every builder that delegates to a second call convention must build the same
//! program as the convention it delegates to.
//!
//! WHY: some builders in this crate exist only as an alternate way to call a
//! builder that the registry already sweeps. One kind takes an explicit op id so
//! a composition crate can keep its own registry identity while reusing the
//! primitive's IR; the other takes a params struct so a fourteen-argument
//! builder has one named-field call site. Neither form is registered, and every
//! call to them is in production code, so nothing compared the two arms: a
//! wrapper that stops forwarding a field, or a params struct whose field lands
//! in the wrong slot, produces a program that still builds, still dispatches,
//! and reads a different buffer. Each case here builds both arms and compares
//! them structurally, then perturbs one input to show the comparison can fail.
//!
//! The member set is derived from the source in [`closure`] rather than written
//! down here, because this header said "five builders" while the crate published
//! eight and the other three were compared by nothing.
//!
//! What this does not catch: whether either arm computes the right answer. The
//! per-domain parity targets own that; this one owns the claim that the two arms
//! are the same program.

#![forbid(unsafe_code)]

#[cfg(feature = "math")]
mod op_id_forms_math {
    use vyre_libs::math::prefix_scan::{
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

#[cfg(feature = "graph")]
mod params_struct_forms {
    use vyre_foundation::ir::Program;
    use vyre_libs::graph::csr_frontier_queue::{
        csr_queue_forward_traverse, csr_queue_forward_traverse_with, CsrQueueForwardTraverseParams,
    };
    use vyre_libs::graph::csr_queue_delta::{
        csr_queue_delta_enqueue, csr_queue_delta_enqueue_with, csr_queue_delta_strided_enqueue,
        csr_queue_delta_strided_enqueue_with, CsrQueueDeltaEnqueueParams,
    };
    use vyre_libs::graph::csr_queue_split::{
        csr_queue_split_low_forward_traverse, csr_queue_split_low_forward_traverse_with,
        CsrQueueSplitLowForwardParams,
    };
    use vyre_libs::graph::csr_queue_strided::{
        csr_queue_strided_forward_traverse, csr_queue_strided_forward_traverse_with,
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

    fn traverse_params() -> CsrQueueForwardTraverseParams<'static> {
        CsrQueueForwardTraverseParams {
            active_queue: "active_queue",
            queue_len: "queue_len",
            edge_offsets: "edge_offsets",
            edge_targets: "edge_targets",
            edge_kind_mask: "edge_kind_mask",
            frontier_out: "frontier_out",
            node_count: NODES,
            edge_count: EDGES,
            queue_capacity: QUEUE_CAP,
            allow_mask: ALLOW,
        }
    }

    /// Both lane strategies share one params struct and one positional entry
    /// shape, so they are compared from one table rather than twice by hand.
    #[allow(clippy::type_complexity)]
    fn traverse_pairs() -> [(
        &'static str,
        fn(&str, &str, &str, &str, &str, &str, u32, u32, u32, u32) -> Program,
        fn(CsrQueueForwardTraverseParams<'_>) -> Program,
    ); 2] {
        [
            (
                "csr_queue_forward_traverse",
                csr_queue_forward_traverse,
                csr_queue_forward_traverse_with,
            ),
            (
                "csr_queue_strided_forward_traverse",
                csr_queue_strided_forward_traverse,
                csr_queue_strided_forward_traverse_with,
            ),
        ]
    }

    #[test]
    fn every_queued_row_traverse_delegates_to_the_params_form() {
        for (name, positional, with_params) in traverse_pairs() {
            let params = traverse_params();
            let wrapper = positional(
                params.active_queue,
                params.queue_len,
                params.edge_offsets,
                params.edge_targets,
                params.edge_kind_mask,
                params.frontier_out,
                params.node_count,
                params.edge_count,
                params.queue_capacity,
                params.allow_mask,
            );
            assert!(
                wrapper.structural_eq(&with_params(traverse_params())),
                "Fix: {name} must build the same program as {name}_with over the same inputs."
            );
            let swapped = with_params(CsrQueueForwardTraverseParams {
                edge_offsets: "edge_targets",
                edge_targets: "edge_offsets",
                ..traverse_params()
            });
            assert!(
                !wrapper.structural_eq(&swapped),
                "Fix: swapping the edge_offsets and edge_targets names must change the program \
                 built by {name}_with, or the equality above would hold for a params struct whose \
                 fields land in the wrong slots."
            );
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
/// Every delegating form this crate publishes must be compared above.
///
/// The member set is the `pub fn` declarations whose name ends in `_with` or
/// `_with_op_id`, read out of `src` at run time, and the covered set is the
/// names this file mentions. The header of this file once said "five builders"
/// while the crate published eight, and the three that arrived later were
/// compared by nothing.
mod closure {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    use vyre_test_support::collect_rust_files;
    use vyre_test_support::monorepo::vyre_crate_directory;

    /// Names of every `pub fn` in one file that ends in a delegating suffix.
    fn delegating_forms(text: &str) -> Vec<String> {
        text.lines()
            .filter_map(|line| line.trim().strip_prefix("pub fn "))
            .filter_map(|rest| rest.split('(').next())
            .filter(|name| name.ends_with("_with") || name.ends_with("_with_op_id"))
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn every_published_delegating_form_is_compared_here() {
        let crate_dir: PathBuf = vyre_crate_directory(env!("CARGO_PKG_NAME"));
        let mut sources = Vec::new();
        collect_rust_files(&crate_dir.join("src"), &mut sources);
        let declared: BTreeSet<String> = sources
            .iter()
            .flat_map(|path: &PathBuf| {
                let text = std::fs::read_to_string(path)
                    .unwrap_or_else(|error| panic!("{path:?} must be readable: {error}"));
                delegating_forms(&text)
            })
            .collect();
        assert!(
            !declared.is_empty(),
            "the declaration scan found no delegating form under src, so it proves nothing"
        );

        let own_source = Path::new(file!()).file_name().expect("test file name");
        let this_file = std::fs::read_to_string(crate_dir.join("tests").join(own_source))
            .expect("this test's own source must be readable");
        let missing: Vec<&String> = declared
            .iter()
            .filter(|name| !this_file.contains(name.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "these delegating forms are published and compared by nothing: {missing:?}. \
             Fix: add a case that builds both arms, do not edit this assertion."
        );
    }
}
