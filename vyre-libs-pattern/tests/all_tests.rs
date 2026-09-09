//! One binary for every integration test in this crate.

#[path = "literal_set_presence_and_positions_reference.rs"]
pub mod literal_set_presence_and_positions_reference;

#[path = "literal_set_presence_reference.rs"]
pub mod literal_set_presence_reference;

#[path = "matching_post_process_contracts.rs"]
pub mod matching_post_process_contracts;

#[path = "nfa_plan_contracts.rs"]
pub mod nfa_plan_contracts;

#[path = "presence_oracle/mod.rs"]
pub mod presence_oracle;

#[path = "region_adversarial.rs"]
pub mod region_adversarial;

#[path = "scan_ac_transition_walk_single_owner.rs"]
pub mod scan_ac_transition_walk_single_owner;

#[path = "scan_hit_buffer_layout_contracts.rs"]
pub mod scan_hit_buffer_layout_contracts;

#[path = "wire_words/mod.rs"]
pub mod wire_words;
