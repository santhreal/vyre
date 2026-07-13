use super::*;
use crate::backend::{BackendError, DispatchConfig, VyreBackend};
use std::sync::Arc;
use vyre_foundation::ir::Program;

mod cache_audit;
mod cache_identity;
mod on_disk;
mod passthrough;

#[test]
fn parse_positive_env_rejects_unset_zero_and_invalid() {
    let name = "VYRE_TEST_PARSE_POSITIVE_ENV_UNIQUE";
    std::env::remove_var(name);
    assert_eq!(super::parse_positive_env::<u32>(name, 7), 7);
    std::env::set_var(name, "0");
    assert_eq!(super::parse_positive_env::<u32>(name, 7), 7);
    std::env::set_var(name, "not-a-number");
    assert_eq!(super::parse_positive_env::<usize>(name, 9), 9);
    std::env::set_var(name, "42");
    assert_eq!(super::parse_positive_env::<u32>(name, 7), 42);
    std::env::remove_var(name);
}
