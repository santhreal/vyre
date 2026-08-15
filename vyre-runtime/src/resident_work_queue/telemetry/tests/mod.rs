// Tests for `telemetry.rs`. Split out per audit item #85 to keep the
// parent file focused on production code.

use super::*;
use crate::resident_work_queue::descriptor::WindowClass;
use crate::resident_work_queue::policy::{
    ResidentExecutionMode, ResidentLaunchRequest, ResidentQueueTopology,
};
use crate::resident_work_queue::protocol::{opcode, SLOT_WORDS};
use crate::resident_work_queue::ResidentWorkQueue;

mod decode_contracts;
mod recommendation_runtime_contracts;
mod sketch_watchdog_contracts;
mod window_contracts;
