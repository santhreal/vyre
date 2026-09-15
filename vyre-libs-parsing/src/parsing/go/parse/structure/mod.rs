//! Go declaration and span extraction, and the record widths a host reader
//! decodes them with.

mod decl_span;
mod declarations;
mod interface_decl;
mod packages;

pub use declarations::go_extract_declarations;
pub use packages::go_extract_packages_and_imports;

/// Words per emitted Go declaration record.
pub const GO_DECL_RECORD_WORDS: u32 = 5;
/// Words per emitted Go span record.
pub const GO_SPAN_RECORD_WORDS: u32 = 2;
/// Function declaration kind.
pub const GO_DECL_FUNC: u32 = 1;
/// Method declaration kind.
pub const GO_DECL_METHOD: u32 = 2;
/// Interface declaration kind.
pub const GO_DECL_INTERFACE: u32 = 3;
