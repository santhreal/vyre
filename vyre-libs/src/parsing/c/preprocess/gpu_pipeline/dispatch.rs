use vyre_foundation::ir::Program;

/// Raw-program execution oracle for C frontend parity and conformance.
///
/// Production execution must compile the frontend output through the canonical
/// artifact lifecycle. This seam exists only to compare frontend stage programs
/// against independent reference or target oracles.
pub trait ProgramOracle {
    /// Run `program` with owned inputs and return one buffer per output.
    fn dispatch(&self, program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String>;

    /// Run `program` with borrowed input buffers.
    ///
    /// The default stages the borrowed slices for oracles that implement only
    /// the owned path.
    fn dispatch_borrowed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, String> {
        let owned: Vec<Vec<u8>> = inputs.iter().map(|slice| slice.to_vec()).collect();
        ProgramOracle::dispatch(self, program, &owned)
    }

    /// Run `program` with borrowed inputs and write outputs into caller-owned slots.
    ///
    /// The default moves each returned buffer into `outputs`, preserving existing
    /// slot allocations where possible.
    fn dispatch_borrowed_into(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), String> {
        let result = self.dispatch_borrowed(program, inputs)?;
        replace_outputs_preserving_slots(outputs, result);
        Ok(())
    }

    /// Whether the oracle requires declared output buffers as input values.
    ///
    /// Target oracles allocate declared outputs themselves. The reference
    /// interpreter consumes one value per non-workgroup buffer and requires
    /// zero-initialized output buffers explicitly.
    fn requires_output_inputs(&self) -> bool {
        false
    }
}

fn replace_outputs_preserving_slots(outputs: &mut Vec<Vec<u8>>, result: Vec<Vec<u8>>) {
    let mut incoming = result.into_iter();
    let mut reused = 0usize;
    for (slot, mut next) in outputs.iter_mut().zip(incoming.by_ref()) {
        if next.len() <= slot.capacity() {
            slot.clear();
            slot.extend_from_slice(&next);
        } else {
            std::mem::swap(slot, &mut next);
        }
        reused += 1;
    }
    outputs.truncate(reused);
    outputs.extend(incoming);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_outputs_preserving_slots_reuses_retained_buffers() {
        let mut outputs = vec![Vec::with_capacity(16), Vec::with_capacity(8)];
        outputs[0].extend_from_slice(&[9, 9]);
        outputs[1].extend_from_slice(&[8]);
        let outer_ptr = outputs.as_ptr() as usize;
        let first_ptr = outputs[0].as_ptr() as usize;
        let second_ptr = outputs[1].as_ptr() as usize;

        replace_outputs_preserving_slots(&mut outputs, vec![vec![1, 2, 3], vec![4]]);

        assert_eq!(outputs, vec![vec![1, 2, 3], vec![4]]);
        assert_eq!(outputs.as_ptr() as usize, outer_ptr);
        assert_eq!(outputs[0].as_ptr() as usize, first_ptr);
        assert_eq!(outputs[1].as_ptr() as usize, second_ptr);
    }

    #[test]
    fn replace_outputs_preserving_slots_moves_oversized_buffers() {
        let mut outputs = vec![Vec::with_capacity(1)];
        outputs[0].push(9);
        let incoming = vec![vec![1, 2, 3, 4]];
        let incoming_ptr = incoming[0].as_ptr() as usize;

        replace_outputs_preserving_slots(&mut outputs, incoming);

        assert_eq!(outputs, vec![vec![1, 2, 3, 4]]);
        assert_eq!(
            outputs[0].as_ptr() as usize,
            incoming_ptr,
            "Fix: oversized C-preprocessor GPU outputs must be moved into retained slots instead of copied through too-small buffers."
        );
    }
}
