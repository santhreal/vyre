//! One binary for every integration test in this crate.

#[path = "presence_oracle/mod.rs"]
pub mod presence_oracle;

#[path = "wire_words/mod.rs"]
pub mod wire_words;

#[path = "ac_walk_hostile_table_contents_oob.rs"]
pub mod ac_walk_hostile_table_contents_oob;

#[path = "adversarial_matching.rs"]
pub mod adversarial_matching;

#[path = "adversarial_nfa.rs"]
pub mod adversarial_nfa;

#[path = "regex_exact_coalesce_parity.rs"]
pub mod regex_exact_coalesce_parity;

#[allow(deprecated)]
#[path = "aho_corasick_kat.rs"]
pub mod aho_corasick_kat;

#[path = "bracket_match_proptest.rs"]
pub mod bracket_match_proptest;

#[path = "dfa_wire_contracts.rs"]
pub mod dfa_wire_contracts;

#[path = "literal_set_presence_and_positions_reference.rs"]
pub mod literal_set_presence_and_positions_reference;

#[path = "literal_set_presence_by_region_ground_truth.rs"]
pub mod literal_set_presence_by_region_ground_truth;

#[path = "literal_set_presence_reference.rs"]
pub mod literal_set_presence_reference;

#[allow(deprecated)]
#[path = "matching_nfa_scan_program_contracts.rs"]
pub mod matching_nfa_scan_program_contracts;

#[path = "matching_post_process_contracts.rs"]
pub mod matching_post_process_contracts;

#[path = "nfa_plan_contracts.rs"]
pub mod nfa_plan_contracts;

#[path = "regex_capture_mode_contracts.rs"]
pub mod regex_capture_mode_contracts;

#[path = "regex_compile_adversarial.rs"]
pub mod regex_compile_adversarial;

#[path = "regex_compile_ascii_class_contracts.rs"]
pub mod regex_compile_ascii_class_contracts;

#[path = "regex_compile_property.rs"]
pub mod regex_compile_property;

#[path = "regex_dfa_anchoring_differential.rs"]
pub mod regex_dfa_anchoring_differential;

#[path = "regex_dfa_char_class_exhaustive.rs"]
pub mod regex_dfa_char_class_exhaustive;

#[path = "regex_dfa_leftmost_longest_differential.rs"]
pub mod regex_dfa_leftmost_longest_differential;

#[path = "regex_unsupported_diagnostic_registry.rs"]
pub mod regex_unsupported_diagnostic_registry;

#[path = "region_adversarial.rs"]
pub mod region_adversarial;

#[path = "region_dedup_property.rs"]
pub mod region_dedup_property;

#[path = "region_gpu_flag_contracts.rs"]
pub mod region_gpu_flag_contracts;

#[path = "scan_ac_transition_walk_single_owner.rs"]
pub mod scan_ac_transition_walk_single_owner;

#[path = "scan_hit_buffer_layout_contracts.rs"]
pub mod scan_hit_buffer_layout_contracts;

#[path = "scan_prefilter_width_closure.rs"]
pub mod scan_prefilter_width_closure;

#[path = "subgroup_nfa_ir_parity_proptest.rs"]
pub mod subgroup_nfa_ir_parity_proptest;

#[path = "substring_search_boundaries.rs"]
pub mod substring_search_boundaries;

#[path = "wire_cross_crate_compat.rs"]
pub mod wire_cross_crate_compat;
