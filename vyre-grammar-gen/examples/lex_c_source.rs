//! Preprocess a C translation unit on the host, then lex it into token kinds.
//!
//! This is the front half of the pipeline: `preprocess_c_host` removes line
//! splices, comments, `#if 0` blocks, and object-like macros, and
//! `lex_c11_max_munch_kinds` turns the result into one token-kind id per token
//! using longest-match (max-munch) semantics. The token kinds are exactly what
//! the GPU parser consumes, so this example is the smallest complete check that
//! a source file lexes at all.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example lex_c_source -p vyre-grammar-gen
//! ```

use vyre_grammar_gen::{kinds_blake3, lex_c11_max_munch_kinds, preprocess_c_host, C11_PATTERNS};

const SOURCE: &str = r#"
#define WIDTH 8

/* a block comment the preprocessor removes */
int add(int a, int b) {
    // a line comment, also removed
    return a + b * WIDTH;
}

#if 0
int never_compiled(void) { return 1; }
#endif
"#;

fn main() {
    println!(
        "the C11 lexer is built from {} patterns",
        C11_PATTERNS.len()
    );

    let preprocessed = preprocess_c_host(SOURCE);
    println!("--- preprocessed ---");
    print!("{preprocessed}");
    println!("--- end ---");

    let kinds = match lex_c11_max_munch_kinds(preprocessed.as_bytes()) {
        Ok(kinds) => kinds,
        Err(error) => {
            eprintln!("lexing failed: {error:?}");
            std::process::exit(1);
        }
    };

    println!("{} tokens", kinds.len());
    println!("first ten kind ids: {:?}", &kinds[..kinds.len().min(10)]);

    // The digest is stable for a given input, which is how the corpus goldens
    // pin lexer output without storing every token.
    println!("kinds digest: {}", kinds_blake3(&kinds).to_hex());
}
