//! The queue-driven CSR traversal sequence every case in this target drives,
//! owned once.
//!
//! Two lifecycles show up across the sibling contract modules. The manual one
//! allocates seven resident handles and drives the queue-build and
//! queue-consume programs itself, which is what pins the kernel and fence
//! counts. The API one goes through `run_resident_csr_queue_query_into` over a
//! reusable graph and scratch, which is what pins that the graph stays
//! resident. Both are sequences, not assertions: each returns the readbacks and
//! the telemetry snapshot so the calling case keeps the counts that are its
//! reason to exist.

use super::*;

/// A CSR graph as the queue parity cases feed it: node count plus the three
/// edge arrays, which always travel together.
pub(super) struct QueueGraph<'a> {
    pub(super) node_count: u32,
    pub(super) edge_offsets: &'a [u32],
    pub(super) edge_targets: &'a [u32],
    pub(super) edge_kind_mask: &'a [u32],
}

impl QueueGraph<'_> {
    /// The one correct `frontier_out`, and the queue length that produced it,
    /// from the CPU reference path.
    ///
    /// Every case checks the device against this, so computing it in one place
    /// keeps a device result from being compared to a differently-parameterized
    /// reference.
    pub(super) fn expected_traverse(
        &self,
        frontier: &[u32],
        queue_capacity: u32,
    ) -> (Vec<u32>, u32) {
        let (expected_queue, expected_len) =
            frontier_to_queue_cpu(frontier, self.node_count, queue_capacity as usize);
        let expected_out = csr_queue_forward_traverse_cpu(
            &expected_queue,
            expected_len,
            self.edge_offsets,
            self.edge_targets,
            self.edge_kind_mask,
            self.node_count,
            1,
        );
        (expected_out, expected_len)
    }

    /// Upload this graph through the resident CSR queue API.
    pub(super) fn upload_resident(
        &self,
        dispatcher: &CudaProgramDispatcher<'_>,
        context: &str,
    ) -> ResidentCsrQueueGraph {
        upload_resident_csr_queue_graph(
            dispatcher,
            self.node_count,
            self.edge_offsets,
            self.edge_targets,
            self.edge_kind_mask,
        )
        .unwrap_or_else(|error| panic!("Fix: {context} resident CSR queue graph upload failed: {error:?}"))
    }
}

/// Which queue-build program the sequence drives.
///
/// The serial and word-parallel builds must produce the same queue, so the
/// cases differ only in this choice and in the frontier density that makes the
/// difference observable.
pub(super) enum QueueBuild {
    /// One lane per node.
    Serial,
    /// Word-parallel atomic scan, for large sparse frontiers.
    Parallel,
}

/// Which compact ranges the sequence reads back.
pub(super) enum QueueReadback {
    /// `frontier_out` only. Pins that the selector count stays device-side.
    FrontierOut,
    /// `frontier_out` and `queue_len`.
    FrontierOutAndLen,
}

/// Whether the fenced dispatch call also carries the CSR graph bytes.
pub(super) enum GraphUpload {
    /// Upload the graph inside the same fenced call as the kernels, which is
    /// what lets a case pin one fence for uploads, kernels and readbacks.
    Fused,
    /// The graph is already resident, so only the frontier and the scratch and
    /// output buffers are refreshed.
    AlreadyResident,
}

/// What one manual queue sequence observed.
pub(super) struct QueueSequenceOutcome {
    /// Decoded `frontier_out` words.
    pub(super) frontier_out: Vec<u32>,
    /// Decoded `queue_len`, present only under [`QueueReadback::FrontierOutAndLen`].
    pub(super) queue_len: Option<u32>,
    /// Telemetry for the fenced call alone: the sequence resets it first.
    pub(super) telemetry: CudaTelemetrySnapshot,
    /// Data bytes the fenced call uploaded, excluding cached parameter uploads.
    pub(super) uploaded_bytes: u64,
}

/// The seven resident handles a manual queue-driven traversal needs.
///
/// Held as a set because every case allocates all seven, passes the same two
/// subsets to the two programs, and frees all seven; splitting them would let a
/// case leak one.
pub(super) struct ManualQueueHandles {
    frontier: u64,
    queue: u64,
    queue_len: u64,
    edge_offsets: u64,
    edge_targets: u64,
    edge_kind_mask: u64,
    frontier_out: u64,
    frontier_words: usize,
    queue_capacity: u32,
}

impl ManualQueueHandles {
    /// Allocate the handle set for `graph` at `queue_capacity`.
    pub(super) fn alloc(
        dispatcher: &CudaProgramDispatcher<'_>,
        graph: &QueueGraph<'_>,
        queue_capacity: u32,
    ) -> Self {
        let word = std::mem::size_of::<u32>();
        let frontier_words = bitset_words(graph.node_count) as usize;
        let alloc = |bytes: usize, what: &str| {
            dispatcher
                .alloc_resident(bytes)
                .unwrap_or_else(|error| panic!("Fix: {what} resident allocation failed: {error:?}"))
        };
        Self {
            frontier: alloc(frontier_words * word, "frontier"),
            queue: alloc(queue_capacity as usize * word, "queue"),
            queue_len: alloc(word, "queue_len"),
            edge_offsets: alloc(graph.edge_offsets.len() * word, "edge_offsets"),
            edge_targets: alloc(graph.edge_targets.len() * word, "edge_targets"),
            edge_kind_mask: alloc(graph.edge_kind_mask.len() * word, "edge_kind_mask"),
            frontier_out: alloc(frontier_words * word, "frontier_out"),
            frontier_words,
            queue_capacity,
        }
    }

    /// Upload the CSR graph on its own, ahead of any query.
    ///
    /// A case that pins "the graph stays resident across queries" needs the
    /// graph bytes to land outside the fenced call it measures.
    pub(super) fn upload_graph(
        &self,
        dispatcher: &CudaProgramDispatcher<'_>,
        graph: &QueueGraph<'_>,
    ) {
        let edge_offsets = u32_bytes(graph.edge_offsets);
        let edge_targets = u32_bytes(graph.edge_targets);
        let edge_kind_mask = u32_bytes(graph.edge_kind_mask);
        dispatcher
            .upload_resident_many(&[
                (self.edge_offsets, edge_offsets.as_slice()),
                (self.edge_targets, edge_targets.as_slice()),
                (self.edge_kind_mask, edge_kind_mask.as_slice()),
            ])
            .expect("Fix: static CSR graph must upload once before repeated queue queries.");
    }

    /// Run queue-build then queue-consume in one fenced call and read back the
    /// requested ranges.
    ///
    /// Telemetry is reset immediately before the call, so the returned snapshot
    /// describes this sequence and nothing before it.
    pub(super) fn run(
        &self,
        backend: &CudaBackend,
        dispatcher: &CudaProgramDispatcher<'_>,
        graph: &QueueGraph<'_>,
        frontier: &[u32],
        build: &QueueBuild,
        upload: &GraphUpload,
        readback: &QueueReadback,
    ) -> QueueSequenceOutcome {
        let word = std::mem::size_of::<u32>();
        let queue_program = match build {
            QueueBuild::Serial => frontier_to_queue(
                "frontier",
                "active_queue",
                "queue_len",
                graph.node_count,
                self.queue_capacity,
            ),
            QueueBuild::Parallel => frontier_to_queue_parallel(
                "frontier",
                "active_queue",
                "queue_len",
                graph.node_count,
                self.queue_capacity,
            ),
        };
        let traverse_program = csr_queue_forward_traverse(
            "active_queue",
            "queue_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "frontier_out",
            graph.node_count,
            graph.edge_targets.len() as u32,
            self.queue_capacity,
            1,
        );
        let queue_handles = [self.frontier, self.queue, self.queue_len];
        let traverse_handles = [
            self.queue,
            self.queue_len,
            self.edge_offsets,
            self.edge_targets,
            self.edge_kind_mask,
            self.frontier_out,
        ];
        let steps = [
            ResidentDispatchStep {
                program: &queue_program,
                handle_ids: &queue_handles,
                grid_override: Some([graph.node_count.div_ceil(256).max(1), 1, 1]),
            },
            ResidentDispatchStep {
                program: &traverse_program,
                handle_ids: &traverse_handles,
                grid_override: Some([self.queue_capacity.div_ceil(256).max(1), 1, 1]),
            },
        ];

        let frontier_bytes = u32_bytes(frontier);
        let zero_queue = vec![0u8; self.queue_capacity as usize * word];
        let zero_count = vec![0u8; word];
        let zero_frontier_out = vec![0u8; self.frontier_words * word];
        let edge_offsets_bytes = u32_bytes(graph.edge_offsets);
        let edge_targets_bytes = u32_bytes(graph.edge_targets);
        let edge_kind_bytes = u32_bytes(graph.edge_kind_mask);
        let mut uploads: Vec<(u64, &[u8])> = vec![
            (self.frontier, frontier_bytes.as_slice()),
            (self.queue_len, zero_count.as_slice()),
            (self.frontier_out, zero_frontier_out.as_slice()),
        ];
        if matches!(upload, GraphUpload::Fused) {
            uploads.insert(1, (self.queue, zero_queue.as_slice()));
            uploads.push((self.edge_offsets, edge_offsets_bytes.as_slice()));
            uploads.push((self.edge_targets, edge_targets_bytes.as_slice()));
            uploads.push((self.edge_kind_mask, edge_kind_bytes.as_slice()));
        }
        let uploaded_bytes = uploads
            .iter()
            .map(|(_, bytes)| bytes.len() as u64)
            .sum::<u64>();

        let frontier_out_range = ResidentReadRange {
            handle_id: self.frontier_out,
            byte_offset: 0,
            byte_len: self.frontier_words * word,
        };
        let queue_len_range = ResidentReadRange {
            handle_id: self.queue_len,
            byte_offset: 0,
            byte_len: word,
        };
        let read_ranges: Vec<ResidentReadRange> = match readback {
            QueueReadback::FrontierOut => vec![frontier_out_range],
            QueueReadback::FrontierOutAndLen => vec![frontier_out_range, queue_len_range],
        };

        backend.reset_telemetry();
        let outputs = dispatcher
            .upload_resident_many_sequence_read_ranges(&uploads, &steps, &read_ranges)
            .expect("Fix: resident queue sparse traversal sequence failed.");
        let telemetry = backend.telemetry_snapshot();
        let queue_len = match readback {
            QueueReadback::FrontierOut => None,
            QueueReadback::FrontierOutAndLen => Some(bytes_u32(&outputs[1])[0]),
        };
        QueueSequenceOutcome {
            frontier_out: bytes_u32(&outputs[0]),
            queue_len,
            telemetry,
            uploaded_bytes,
        }
    }

    /// Free every handle in the set.
    pub(super) fn free(self, dispatcher: &CudaProgramDispatcher<'_>) {
        for handle in [
            self.frontier,
            self.queue,
            self.queue_len,
            self.edge_offsets,
            self.edge_targets,
            self.edge_kind_mask,
            self.frontier_out,
        ] {
            dispatcher
                .free_resident(handle)
                .expect("Fix: resident queue traversal cleanup failed.");
        }
    }
}

/// Run one whole manual queue-driven traversal: allocate, upload everything in
/// the fenced call, dispatch, read back `frontier_out` and `queue_len`, free.
///
/// This is the shape a one-shot case wants; a case that pins graph residency
/// across queries drives [`ManualQueueHandles`] directly instead.
pub(super) fn run_manual_queue_sequence(
    backend: &CudaBackend,
    dispatcher: &CudaProgramDispatcher<'_>,
    graph: &QueueGraph<'_>,
    frontier: &[u32],
    queue_capacity: u32,
    build: QueueBuild,
) -> QueueSequenceOutcome {
    let handles = ManualQueueHandles::alloc(dispatcher, graph, queue_capacity);
    let outcome = handles.run(
        backend,
        dispatcher,
        graph,
        frontier,
        &build,
        &GraphUpload::Fused,
        &QueueReadback::FrontierOutAndLen,
    );
    handles.free(dispatcher);
    outcome
}

/// A resident CSR queue graph with reusable scratch and a caller-owned output
/// buffer, driven through the public resident query API.
pub(super) struct ResidentQueueSession {
    graph: ResidentCsrQueueGraph,
    scratch: ResidentCsrQueueScratch,
    /// Allocated once at full capacity so a query that grew it would move the
    /// pointer, which is what [`Self::output_ptr`] lets a case detect.
    output: Vec<u8>,
    context: &'static str,
}

impl ResidentQueueSession {
    /// Upload `graph` and prepare scratch and output for repeated queries.
    ///
    /// `context` names the case in every failure message, since one API defect
    /// surfaces identically across graph shapes.
    pub(super) fn open(
        dispatcher: &CudaProgramDispatcher<'_>,
        graph: &QueueGraph<'_>,
        context: &'static str,
    ) -> Self {
        Self {
            graph: graph.upload_resident(dispatcher, context),
            scratch: ResidentCsrQueueScratch::default(),
            output: Vec::with_capacity(
                bitset_words(graph.node_count) as usize * std::mem::size_of::<u32>(),
            ),
            context,
        }
    }

    /// The output buffer's allocation, so a case can pin that a query reused
    /// caller-owned capacity instead of reallocating.
    pub(super) fn output_ptr(&self) -> *const u8 {
        self.output.as_ptr()
    }

    /// Bytes the last query wrote back.
    pub(super) fn output_len(&self) -> usize {
        self.output.len()
    }

    /// Run one query and assert the traversal matches the CPU reference.
    ///
    /// The returned telemetry covers this query alone.
    ///
    /// # Panics
    ///
    /// Panics when the query fails or the device traversal disagrees with the
    /// reference, which is the parity contract itself.
    pub(super) fn query(
        &mut self,
        backend: &CudaBackend,
        dispatcher: &CudaProgramDispatcher<'_>,
        graph: &QueueGraph<'_>,
        frontier: &[u32],
        queue_capacity: u32,
    ) -> CudaTelemetrySnapshot {
        let (expected_out, _) = graph.expected_traverse(frontier, queue_capacity);
        backend.reset_telemetry();
        run_resident_csr_queue_query_into(
            dispatcher,
            &self.graph,
            &mut self.scratch,
            frontier,
            queue_capacity,
            1,
            &mut self.output,
        )
        .unwrap_or_else(|error| {
            panic!(
                "Fix: {} resident CSR queue query failed on CUDA: {error:?}",
                self.context
            )
        });
        assert_eq!(
            bytes_u32(&self.output),
            expected_out,
            "Fix: {} resident CSR queue traversal must match the CPU reference.",
            self.context
        );
        backend.telemetry_snapshot()
    }

    /// Free the scratch and the graph.
    pub(super) fn close(mut self, dispatcher: &CudaProgramDispatcher<'_>) {
        self.scratch
            .free(dispatcher)
            .unwrap_or_else(|error| panic!("Fix: {} scratch cleanup failed: {error:?}", self.context));
        self.graph
            .free(dispatcher)
            .unwrap_or_else(|error| panic!("Fix: {} graph cleanup failed: {error:?}", self.context));
    }
}
