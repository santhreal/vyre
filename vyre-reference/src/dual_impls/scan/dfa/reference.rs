use crate::{dual_impls::evaluator, workgroup::Memory};
use vyre_primitives::PatternMatchDfa;

struct WireDfa {
    state_count: u32,
    transitions: Vec<u32>,
    accept: Vec<u32>,
}

impl WireDfa {
    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 {
            return Err("DFA wire blob too short for header".into());
        }
        if &bytes[0..4] != b"VDFA" {
            return Err("DFA wire blob bad magic".into());
        }
        let version = bytes
            .get(4..8)
            .and_then(|s| s.try_into().ok())
            .map(u32::from_le_bytes)
            .ok_or_else(|| "DFA wire blob too short for version".to_string())?;
        if version != 2 {
            return Err(format!("DFA wire version {version} != 2"));
        }
        let state_count = bytes
            .get(8..12)
            .and_then(|s| s.try_into().ok())
            .map(u32::from_le_bytes)
            .ok_or_else(|| "DFA wire blob too short for state count".to_string())?;
        let _max_pattern_len = bytes
            .get(12..16)
            .and_then(|s| s.try_into().ok())
            .map(u32::from_le_bytes)
            .ok_or_else(|| "DFA wire blob too short for max pattern length".to_string())?;
        let mut cursor = 16;
        let read_u32_slice = |cursor: &mut usize| -> Result<Vec<u32>, String> {
            if *cursor + 4 > bytes.len() {
                return Err("truncated section length".into());
            }
            let word_count = bytes
                .get(*cursor..*cursor + 4)
                .and_then(|s| s.try_into().ok())
                .map(u32::from_le_bytes)
                .ok_or_else(|| "truncated section length".to_string())?
                as usize;
            *cursor += 4;
            let byte_count = word_count * 4;
            if *cursor + byte_count > bytes.len() {
                return Err("truncated section payload".into());
            }
            let mut words = Vec::with_capacity(word_count);
            for chunk in bytes[*cursor..*cursor + byte_count].chunks_exact(4) {
                let word = chunk
                    .try_into()
                    .map(u32::from_le_bytes)
                    .map_err(|_| "invalid word chunk".to_string())?;
                words.push(word);
            }
            *cursor += byte_count;
            Ok(words)
        };

        let transitions = read_u32_slice(&mut cursor)?;
        let accept = read_u32_slice(&mut cursor)?;
        let _output_offsets = read_u32_slice(&mut cursor)?;
        let _output_records = read_u32_slice(&mut cursor)?;

        if transitions.len() != (state_count as usize) * 256 {
            return Err("transition table length does not match state_count * 256".into());
        }
        if accept.len() != state_count as usize {
            return Err("accept table length does not match state_count".into());
        }

        Ok(Self {
            state_count,
            transitions,
            accept,
        })
    }
}
impl evaluator::ReferenceEvaluator for PatternMatchDfa {
    fn evaluate(&self, inputs: &[Memory]) -> Result<Memory, evaluator::EvalError> {
        let haystack = evaluator::one_input(inputs, "scan_dfa")?;
        // Decode using the canonical V2 wire format produced by CompiledDfa::to_bytes.
        // The old hand-rolled V1 parser (magic + state_count + start + accept_count)
        // does not match the V2 envelope (magic + version + state_count + max_pattern_len
        // + length-prefixed sections). Using from_bytes here keeps the reference oracle
        // byte-identical with every other consumer of the DFA wire format.
        let compiled = WireDfa::from_bytes(&self.dfa).map_err(|e| {
            evaluator::EvalError::new(format!(
                "primitive `scan_dfa` could not decode DFA wire blob: {e}. \
                 Fix: populate PatternMatchDfa.dfa via CompiledDfa::to_bytes()."
            ))
        })?;

        // State 0 is always the root/start state in the Aho-Corasick DFA produced
        // by dfa_compile. There is no separate start field in the V2 format.
        let mut state = 0usize;
        let mut offsets = Vec::new();
        for (offset, byte) in haystack.iter().copied().enumerate() {
            let next_state_idx = state * 256 + usize::from(byte);
            let next = compiled.transitions[next_state_idx] as usize;
            if next >= compiled.state_count as usize {
                return Err(evaluator::EvalError::new(
                    "primitive `scan_dfa` transition targets an out-of-range state. \
                     Fix: validate every transition target in the DFA.",
                ));
            }
            state = next;
            // accept[state] is non-zero when the state matches at least one pattern.
            if compiled.accept[state] != 0 {
                offsets.push(u32::try_from(offset).map_err(|_| {
                    evaluator::EvalError::new(
                        "primitive `scan_dfa` offset exceeds u32. Fix: split haystacks before 4 GiB.",
                    )
                })?);
            }
        }
        Ok(evaluator::write_u32s(offsets))
    }
}

// Inline: covers items in the crate-private `dual_impls::evaluator` module, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual_impls::evaluator::ReferenceEvaluator;
    use crate::workgroup::Memory;
    #[test]
    fn test_dfa_reference_v2_roundtrip() {
        // Build a V2 wire DFA for pattern "abc" with 4 states:
        // state 0: 'a' -> 1, others -> 0
        // state 1: 'b' -> 2, 'a' -> 1, others -> 0
        // state 2: 'c' -> 3, 'a' -> 1, others -> 0
        // state 3: 'a' -> 1, others -> 0; accept = 1 (pid 0 + 1)
        let state_count = 4u32;
        let max_pattern_len = 3u32;
        let mut transitions = vec![0u32; (state_count as usize) * 256];
        transitions[0 * 256 + (b'a' as usize)] = 1;
        transitions[1 * 256 + (b'b' as usize)] = 2;
        transitions[1 * 256 + (b'a' as usize)] = 1;
        transitions[2 * 256 + (b'c' as usize)] = 3;
        transitions[2 * 256 + (b'a' as usize)] = 1;
        transitions[3 * 256 + (b'a' as usize)] = 1;

        let accept = vec![0u32, 0, 0, 1];
        let output_offsets = vec![0u32, 0, 0, 0, 1];
        let output_records = vec![0u32];

        let mut wire_bytes = Vec::new();
        wire_bytes.extend_from_slice(b"VDFA");
        wire_bytes.extend_from_slice(&2u32.to_le_bytes());
        wire_bytes.extend_from_slice(&state_count.to_le_bytes());
        wire_bytes.extend_from_slice(&max_pattern_len.to_le_bytes());

        let append_slice = |buf: &mut Vec<u8>, slice: &[u32]| {
            buf.extend_from_slice(&(slice.len() as u32).to_le_bytes());
            for word in slice {
                buf.extend_from_slice(&word.to_le_bytes());
            }
        };
        append_slice(&mut wire_bytes, &transitions);
        append_slice(&mut wire_bytes, &accept);
        append_slice(&mut wire_bytes, &output_offsets);
        append_slice(&mut wire_bytes, &output_records);

        let primitive = PatternMatchDfa { dfa: wire_bytes };
        // Haystack: "xxabcxx" (pattern starts at byte 2, ends (accepting) at byte 4).
        let haystack = Memory::from_bytes(b"xxabcxx".to_vec());
        let result = primitive
            .evaluate(&[haystack])
            .expect("Fix: V2 DFA roundtrip must succeed on valid haystack");

        // The evaluator records the offset of the accepting byte (offset 4, 0-indexed).
        let offsets: Vec<u32> = result
            .bytes()
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        // "abc" completes at index 4 (x=0,x=1,a=2,b=3,c=4).
        assert_eq!(
            offsets,
            vec![4u32],
            "Fix: V2 DFA reference evaluator must report offset 4 for 'abc' in 'xxabcxx'"
        );
    }
}
