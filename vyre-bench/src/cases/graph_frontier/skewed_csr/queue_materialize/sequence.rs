//! The two split-queue sequence dispatchers for the skewed CSR frontier case.
//!
//! The binding layout and resource index arrays these expand against are owned
//! by [`crate::cases::queue_stage`].

use super::GraphCsrSkewedQueuePrepared;

crate::cases::queue_stage::define_resident_queue_sequence_dispatch!(
    pub(super) dispatch_resident_queue_sequence,
    GraphCsrSkewedQueuePrepared,
    "skewed CSR"
);

crate::cases::queue_stage::define_host_queue_sequence_dispatch!(
    pub(super) dispatch_host_queue_sequence,
    GraphCsrSkewedQueuePrepared,
    "skewed CSR"
);
