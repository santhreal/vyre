//! Target compilation: the boundary from one authenticated neutral artifact to
//! immutable target-native bytes.

#[path = "target_bundle.rs"]
mod bundle;
#[path = "target_compile.rs"]
mod compile;
#[path = "target_error.rs"]
mod error;

#[cfg(test)]
#[path = "target_tests.rs"]
mod tests;

pub use bundle::{
    ModuleNumericRecord, SelectedLowering, SelectedModule, TargetArmAssignment, TargetModuleBundle,
    TargetModuleImage, TARGET_MODULE_BUNDLE_SCHEMA_VERSION,
};
pub use compile::{attach_target, compile_selected_modules, EmittedTargetModule, TargetCompiler};
pub use error::TargetCompileError;
