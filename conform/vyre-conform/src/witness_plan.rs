//! Shared witness-input planner for release conformance paths.
//!
//! Fixtures are authored in logical input order, while artifact submission
//! expects one slice per canonical program input. The planner expands logical
//! witnesses into the stream consumed by both `vyre_reference::reference_eval`
//! and [`crate::production::ProductionSession`].

use vyre::ir::{BufferAccess, BufferDecl, MemoryKind, Program};

#[derive(Clone)]
enum WitnessInputSource {
    Fixture {
        fixture_index: usize,
        buffer_index: usize,
        byte_len: Option<usize>,
    },
    ReadWriteOrZero {
        fixture_index: usize,
        buffer_index: usize,
        zero_index: Option<usize>,
        byte_len: Option<usize>,
    },
}

/// Planned mapping from logical witness fixtures to backend/reference inputs.
#[derive(Clone)]
pub struct WitnessInputPlan {
    sources: Vec<WitnessInputSource>,
    zeroed_inputs: Vec<Vec<u8>>,
    buffer_len: usize,
}

impl WitnessInputPlan {
    /// Build the logical witness-input plan for a Program.
    ///
    /// Shared memory, declared output buffers, and pipeline live-out read-write
    /// buffers are not witness inputs. Static read-write buffers can be omitted
    /// from the fixture and are then zero-filled; runtime-sized read-write
    /// buffers require explicit fixture bytes.
    pub fn for_program(program: &Program) -> Result<Self, String> {
        let mut sources = Vec::with_capacity(program.buffers().len());
        let mut zeroed_inputs = Vec::with_capacity(program.buffers().len());
        let mut fixture_index = 0usize;
        for (buffer_index, buffer) in program.buffers().iter().enumerate() {
            if buffer.kind() == MemoryKind::Shared
                || buffer.access() == BufferAccess::Workgroup
                || buffer.is_backend_allocated_output()
            {
                continue;
            }
            if matches!(buffer.access(), BufferAccess::ReadWrite) {
                let byte_len = fixture_backed_byte_len(buffer, "read-write witness buffer")?;
                let zero_index = if let Some(byte_len) = byte_len {
                    let zero_index = zeroed_inputs.len();
                    zeroed_inputs.push(vec![0u8; byte_len]);
                    Some(zero_index)
                } else {
                    None
                };
                sources.push(WitnessInputSource::ReadWriteOrZero {
                    fixture_index,
                    buffer_index,
                    zero_index,
                    byte_len,
                });
                fixture_index += 1;
                continue;
            }
            let byte_len = fixture_backed_byte_len(buffer, "input witness buffer")?;
            sources.push(WitnessInputSource::Fixture {
                fixture_index,
                buffer_index,
                byte_len,
            });
            fixture_index += 1;
        }

        Ok(Self {
            sources,
            zeroed_inputs,
            buffer_len: program.buffers().len(),
        })
    }

    /// Number of executable input slices produced by this plan.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Number of static read-write inputs that this plan can synthesize.
    pub fn zeroed_input_count(&self) -> usize {
        self.zeroed_inputs.len()
    }

    /// Program buffer index behind each planned input slice, in stream order.
    ///
    /// The adversarial ULP companions rewrite one input at a time and need the
    /// buffer declaration behind each slice to know its element type. The plan
    /// skips outputs, shared memory and pipeline live-outs, so a stream
    /// position is not a `Program::buffers` position.
    pub fn buffer_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.sources.iter().map(|source| match source {
            WitnessInputSource::Fixture { buffer_index, .. }
            | WitnessInputSource::ReadWriteOrZero { buffer_index, .. } => *buffer_index,
        })
    }
}

fn fixture_backed_byte_len(buffer: &BufferDecl, role: &str) -> Result<Option<usize>, String> {
    buffer
        .static_byte_len()
        .map_err(|error| format!("{role} `{}`: {error}", buffer.name()))
}

/// Static byte length required when synthesising a complete fixture.
pub fn static_buffer_byte_len(buffer: &BufferDecl, role: &str) -> Result<usize, String> {
    buffer
        .static_byte_len()
        .map_err(|error| format!("{role} `{}`: {error}", buffer.name()))?
        .ok_or_else(|| {
            format!(
                "{role} `{}` is runtime-sized. Fix: provide explicit witness bytes for dynamically sized buffers.",
                buffer.name()
            )
        })
}

/// Expand logical fixture bytes into the planned dispatch input stream.
pub fn plan_witness_inputs_into<'a>(
    fixture_inputs: &'a [Vec<u8>],
    plan: &'a WitnessInputPlan,
    backend_inputs: &mut Vec<&'a [u8]>,
) -> Result<(), String> {
    if fixture_inputs.len() > plan.buffer_len {
        return Err(format!(
            "witness fixture provided {} buffer(s) but Program declares {}. Fix: fixture cases must not exceed Program::buffers order.",
            fixture_inputs.len(),
            plan.buffer_len
        ));
    }

    backend_inputs.clear();
    for source in &plan.sources {
        match source {
            WitnessInputSource::Fixture {
                fixture_index,
                buffer_index,
                byte_len,
            } => {
                if let Some(bytes) = matching_fixture_bytes(
                    fixture_inputs,
                    *buffer_index,
                    *fixture_index,
                    plan.sources.len(),
                ) {
                    if let Some(expected) = byte_len {
                        if bytes.len() != *expected {
                            return Err(format!(
                                "witness input buffer at fixture index `{fixture_index}` / program index `{buffer_index}` expected {expected} bytes from its static buffer declaration but received {} bytes. Fix: provide exact fixture byte lengths matching the Program declaration.",
                                bytes.len()
                            ));
                        }
                    }
                    backend_inputs.push(bytes);
                    continue;
                }
                return Err(format!(
                    "witness omitted required input buffer at fixture index `{fixture_index}` / program index `{buffer_index}`. Fix: every non-output read-only/uniform buffer must be present in the witness case."
                ));
            }
            WitnessInputSource::ReadWriteOrZero {
                fixture_index,
                buffer_index,
                zero_index,
                byte_len,
            } => {
                if let Some(bytes) = matching_fixture_bytes(
                    fixture_inputs,
                    *buffer_index,
                    *fixture_index,
                    plan.sources.len(),
                ) {
                    if let Some(expected) = byte_len {
                        if bytes.len() != *expected {
                            return Err(format!(
                                "witness input buffer at fixture index `{fixture_index}` / program index `{buffer_index}` expected {expected} bytes from its static buffer declaration but received {} bytes. Fix: provide exact fixture byte lengths matching the Program declaration.",
                                bytes.len()
                            ));
                        }
                    }
                    backend_inputs.push(bytes);
                    continue;
                }
                if let Some(zero_index) = zero_index {
                    if let Some(bytes) = plan.zeroed_inputs.get(*zero_index) {
                        backend_inputs.push(bytes.as_slice());
                        continue;
                    }
                    return Err(
                        "internal plan mismatch: zeroed input index is invalid.".to_string()
                    );
                }
                return Err(format!(
                    "witness omitted runtime-sized read-write buffer at fixture index `{fixture_index}` / program index `{buffer_index}`. Fix: provide concrete fixture bytes because dynamic read-write buffers cannot be zero-initialized without a byte length."
                ));
            }
        }
    }
    Ok(())
}

/// Expand logical fixture bytes into owned copies of the planned input stream.
///
/// A caller that mutates the stream cannot borrow it from the fixture. The ULP
/// adversarial companions overwrite every f32 input in place, so they need one
/// owned buffer per planned slice.
pub fn plan_witness_inputs_owned_into(
    fixture_inputs: &[Vec<u8>],
    plan: &WitnessInputPlan,
    owned_inputs: &mut Vec<Vec<u8>>,
) -> Result<(), String> {
    let mut borrowed = Vec::with_capacity(plan.sources.len());
    plan_witness_inputs_into(fixture_inputs, plan, &mut borrowed)?;
    owned_inputs.clear();
    owned_inputs.reserve(borrowed.len());
    owned_inputs.extend(borrowed.into_iter().map(<[u8]>::to_vec));
    Ok(())
}

fn matching_fixture_bytes(
    fixture_inputs: &[Vec<u8>],
    buffer_index: usize,
    fixture_index: usize,
    total_sources: usize,
) -> Option<&[u8]> {
    let index = if fixture_inputs.len() > total_sources {
        buffer_index
    } else {
        fixture_index
    };
    fixture_inputs.get(index).map(Vec::as_slice)
}
