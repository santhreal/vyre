//! Exact regex extents under finite accelerator replay budgets.

#![cfg(all(feature = "matching-regex", feature = "matching-dfa", feature = "rule"))]

use regex::bytes::Regex;
use vyre::ir::Program;
use vyre::scan::{
    build_regex_dfa_pipeline_with_policy, build_regex_dfa_pipeline_with_policy_ext,
    compile_regex_set, compile_regex_set_with_policy, pack_haystack_u32, pack_u32_slice,
    RegexCompileError, RegexReplayPolicy, RegionEvidencePipeline,
    DEFAULT_OPEN_ENDED_REPLAY_LIMIT_BYTES,
};
use vyre_foundation::match_result::ByteRange;
use vyre_reference::value::Value;

fn with_reference_dispatch_lanes(program: Program, lanes: u32) -> Program {
    let buffers = program
        .buffers()
        .iter()
        .cloned()
        .map(|buffer| {
            if buffer.name() == "match_count" {
                buffer.with_count(lanes.max(1)).with_output_byte_range(0..4)
            } else {
                buffer
            }
        })
        .collect();
    program.with_rewritten_buffers(buffers)
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn canonicalize(matches: &mut Vec<ByteRange>) {
    matches.sort_unstable_by_key(|m| (m.start, m.tag, m.end));
    let mut write = 0usize;
    for read in 0..matches.len() {
        let current = matches[read];
        if write > 0
            && matches[write - 1].start == current.start
            && matches[write - 1].tag == current.tag
        {
            matches[write - 1] = current;
        } else {
            matches[write] = current;
            write += 1;
        }
    }
    matches.truncate(write);
    matches.sort_unstable_by_key(|m| (m.start, m.end, m.tag));
}

fn independent_oracle(patterns: &[&str], replay_limits: &[u32], haystack: &[u8]) -> Vec<ByteRange> {
    let regexes: Vec<_> = patterns
        .iter()
        .map(|pattern| Regex::new(&format!("^(?:{pattern})")).unwrap())
        .collect();
    let mut matches = Vec::new();
    for origin in 0..haystack.len() {
        for (pattern_id, regex) in regexes.iter().enumerate() {
            let end = haystack
                .len()
                .min(origin.saturating_add(replay_limits[pattern_id] as usize));
            if let Some(found) = regex.find(&haystack[origin..end]) {
                matches.push(ByteRange::new(
                    pattern_id as u32,
                    origin as u32,
                    (origin + found.end()) as u32,
                ));
            }
        }
    }
    matches.sort_unstable_by_key(|m| (m.start, m.end, m.tag));
    matches
}

fn execute_pipeline(
    pipeline: &vyre::scan::RegexDfaPipeline,
    haystack: &[u8],
    max_matches: u32,
) -> Vec<ByteRange> {
    let lanes = haystack.len() as u32;
    let program = with_reference_dispatch_lanes(pipeline.program.clone(), lanes);
    let packed_haystack = pack_haystack_u32(haystack);
    let haystack_len = pack_u32_slice(&[lanes]);
    let match_count = pack_u32_slice(&vec![0; lanes.max(1) as usize]);
    let match_scratch = pack_u32_slice(&vec![0; max_matches as usize * 3]);
    let inputs = [
        Value::from(packed_haystack),
        Value::from(pack_u32_slice(&pipeline.dfa.transitions)),
        Value::from(pack_u32_slice(&pipeline.dfa.output_offsets)),
        Value::from(pack_u32_slice(&pipeline.dfa.output_records)),
        Value::from(pack_u32_slice(&pipeline.pattern_lengths)),
        Value::from(haystack_len),
        Value::from(match_count),
        Value::from(match_scratch),
    ];
    let outputs = vyre_reference::reference_eval(&program, &inputs)
        .expect("exact regex range program must execute in the reference backend");
    let count = words(&outputs[0].to_bytes())[0] as usize;
    let match_words = words(&outputs[1].to_bytes());
    let mut matches = match_words[..count * 3]
        .chunks_exact(3)
        .map(|triple| ByteRange::new(triple[0], triple[1], triple[2]))
        .collect();
    canonicalize(&mut matches);
    matches
}

/// Locks out the old `+`/`*` bug that advertised the minimum as the replay maximum.
#[test]
fn open_ended_patterns_publish_an_explicit_finite_replay_budget() {
    let compiled = compile_regex_set(&[r"key=[0-9]+"]).unwrap();
    assert_eq!(compiled.pattern_extents.len(), 1);
    assert_eq!(compiled.pattern_extents[0].min_bytes, 5);
    assert_eq!(compiled.pattern_extents[0].max_bytes, None);
    assert_eq!(
        compiled.pattern_extents[0].replay_limit_bytes,
        DEFAULT_OPEN_ENDED_REPLAY_LIMIT_BYTES
    );
    assert_eq!(
        compiled.plan.accept_states,
        vec![(0, DEFAULT_OPEN_ENDED_REPLAY_LIMIT_BYTES)]
    );
}

/// Proves callers can tighten the open-ended bound and get that exact accelerator window.
#[test]
fn explicit_open_ended_policy_controls_replay_without_changing_minimum() {
    let compiled = compile_regex_set_with_policy(
        &[r"a+"],
        RegexReplayPolicy {
            open_ended_limit_bytes: 17,
        },
    )
    .unwrap();
    assert_eq!(compiled.pattern_extents[0].min_bytes, 1);
    assert_eq!(compiled.pattern_extents[0].max_bytes, None);
    assert_eq!(compiled.pattern_extents[0].replay_limit_bytes, 17);
    assert_eq!(compiled.plan.accept_states, vec![(0, 17)]);
}

/// Prevents a configured replay cap from silently making a pattern's minimum unreachable.
#[test]
fn replay_policy_smaller_than_open_ended_minimum_fails_closed() {
    let error = compile_regex_set_with_policy(
        &[r"prefix[0-9]{8,}"],
        RegexReplayPolicy {
            open_ended_limit_bytes: 13,
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        RegexCompileError::OpenEndedReplayLimitTooSmall {
            pattern_index: 0,
            minimum: 14,
            limit: 13,
        }
    ));
}

/// Replays the production IR and compares every exact span with Rust regex as an independent oracle.
#[test]
fn whole_buffer_variable_matches_report_origin_derived_exact_extents() {
    let patterns = [r"a{2,4}", r"b+"];
    let policy = RegexReplayPolicy {
        open_ended_limit_bytes: 6,
    };
    let max_matches = 256;
    let pipeline =
        build_regex_dfa_pipeline_with_policy_ext(&patterns, max_matches, 16_384, policy, false)
            .unwrap();
    let haystack = b"xaaaa!ybbbbbbbb!aa";
    let actual = execute_pipeline(&pipeline, haystack, max_matches);
    let expected = independent_oracle(&patterns, &pipeline.pattern_lengths, haystack);
    assert_eq!(actual, expected);
    assert!(actual.contains(&ByteRange::new(0, 1, 5)));
    assert!(actual.contains(&ByteRange::new(1, 7, 13)));
}

/// Ensures evidence consumers expose one maximal token per origin instead of every accepting end.
#[test]
fn region_evidence_positions_use_the_same_leftmost_longest_contract() {
    let patterns = [r"a{2,4}"];
    let pipeline = build_regex_dfa_pipeline_with_policy(
        &patterns,
        64,
        16_384,
        RegexReplayPolicy {
            open_ended_limit_bytes: 8,
        },
    )
    .unwrap();
    let evidence = RegionEvidencePipeline::new(pipeline.dfa, 1, vec![1], vec![1])
        .unwrap()
        .reference_scan(b"aaaa", &[0], 0);
    assert_eq!(
        evidence.positions,
        vec![
            ByteRange::new(0, 0, 4),
            ByteRange::new(0, 1, 4),
            ByteRange::new(0, 2, 4),
        ]
    );
    assert_eq!(evidence.presence, vec![1]);
    assert_eq!(evidence.admission, vec![1]);
}
