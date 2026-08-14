//! Reference, property, and GPU parity tests for C11 scope semantics.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]

use proptest::prelude::*;
use std::sync::OnceLock;
use vyre::ir::{Expr, Program};
use vyre::validate;
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_emit_naga::program as naga_emit;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::sema::c_sema_scope;

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_frontend::rows::haystack_words;
use c_frontend::scope_fixture::{
    c_atoms, emit_u32_bytes, fixture, ident, scope_tree_words_for, tok, Atom, ScopeFixture,
};

const TEST_WORKGROUP_SIZE: [u32; 3] = [1, 1, 1];

fn reference_values(inputs: &[Vec<u8>]) -> Vec<vyre_reference::value::Value> {
    let owned_inputs;
    let inputs = if inputs.iter().any(Vec::is_empty) {
        owned_inputs = inputs
            .iter()
            .map(|input| {
                if input.is_empty() {
                    vec![0; 4]
                } else {
                    input.clone()
                }
            })
            .collect::<Vec<_>>();
        owned_inputs.as_slice()
    } else {
        inputs
    };
    let mut values = inputs
        .iter()
        .map(|input| input.as_slice().into())
        .collect::<Vec<_>>();
    if inputs.len() == 4 {
        let token_words = inputs[0].len() / 4;
        values.push(vec![0; token_words.saturating_mul(4).max(1) * 4].into());
    }
    values
}

fn program_for(num_tokens: u32, haystack_len: usize) -> Program {
    c_sema_scope(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "haystack",
        Expr::u32(haystack_len as u32),
        Expr::u32(num_tokens),
        "out_scope_tree",
    )
}

fn assert_exact_mapping(name: &str, expected: &[u32], actual: &[u8]) {
    let actual_words: Vec<u32> = actual
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 chunk")))
        .collect();
    assert_eq!(
        expected.len(),
        actual_words.len(),
        "{name}: scope-tree width mismatch, expected {} words, got {}",
        expected.len(),
        actual_words.len()
    );

    for (node_idx, chunk) in actual_words.chunks_exact(4).enumerate() {
        let expected_chunk = &expected[node_idx * 4..node_idx * 4 + 4];
        assert_eq!(
            chunk, expected_chunk,
            "{name}: exact mapping mismatch at node {node_idx}: expected {expected_chunk:?}, got {chunk:?}"
        );
    }
}

fn backend() -> &'static WgpuBackend {
    static BACKEND: OnceLock<WgpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| WgpuBackend::acquire().expect("Fix: GPU backend must be available"))
}

fn case_inputs(fix: &ScopeFixture) -> Vec<Vec<u8>> {
    vec![
        emit_u32_bytes(&fix.tok_types),
        emit_u32_bytes(&fix.tok_starts),
        emit_u32_bytes(&fix.tok_lens),
        haystack_words(&fix.haystack),
    ]
}

#[test]
fn c_sema_scope_program_emits_valid_wgsl() {
    let fixture = fixture("wgsl", &c_atoms("int main ( ) { return int x ; }"));
    let program = program_for(fixture.tok_types.len() as u32, fixture.haystack.len());
    let errors = validate(&program);
    assert!(errors.is_empty(), "c_sema_scope must validate: {errors:?}");
    let module = naga_emit::emit_module(&program, TEST_WORKGROUP_SIZE)
        .expect("Scope op must lower to a valid Naga module");
    let _info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("Naga must validate scope op module");
    assert!(
        module
            .entry_points
            .iter()
            .any(|entry| entry.stage == naga::ShaderStage::Compute),
        "Scope op Naga module should define a compute entry"
    );
}

#[test]
fn c_sema_scope_witness_matches_cpu_reference() {
    let fixture = fixture(
        "witness",
        &c_atoms("int main ( int x ; } label : goto label ; { }"),
    );
    let program = program_for(fixture.tok_types.len() as u32, fixture.haystack.len());
    let reference = scope_tree_words_for(&fixture);
    let reference_bytes = emit_u32_bytes(&reference);
    let inputs = case_inputs(&fixture);
    let result = vyre_reference::reference_eval(&program, &reference_values(&inputs))
        .expect("Reference evaluator must run");
    assert_eq!(result.len(), 1, "Expected one output buffer");
    let actual = result[0].to_bytes().to_vec();
    assert_eq!(actual, reference_bytes);
    assert_exact_mapping("witness", &reference, &actual);
}

fn named_case(name: &str, atoms: Vec<Atom>) -> (String, ScopeFixture) {
    (name.to_string(), fixture(name, &atoms))
}

/// Named adversarial cases. The name is the case label an assertion reports, so
/// it travels beside the fixture rather than inside it.
fn adversarial_fixtures() -> Vec<(String, ScopeFixture)> {
    let mut cases = Vec::new();
    for depth in 1..=12 {
        let mut atoms = Vec::new();
        for idx in 0..depth {
            atoms.push(tok(TOK_LBRACE));
            atoms.push(tok(TOK_INT));
            atoms.push(ident(&format!("outer_{depth}_{idx}")));
            atoms.push(tok(TOK_SEMICOLON));
        }
        for _ in 0..depth {
            atoms.push(tok(TOK_RBRACE));
        }
        cases.push(named_case(&format!("nested_blocks_depth_{depth}"), atoms));
    }

    for depth in 1..=10 {
        let mut atoms = vec![tok(TOK_LBRACE)];
        for idx in 0..depth {
            atoms.push(tok(TOK_INT));
            atoms.push(ident(&format!("x_{idx}")));
            atoms.push(tok(TOK_SEMICOLON));
            atoms.push(tok(TOK_LBRACE));
        }
        for _ in 0..=depth {
            atoms.push(tok(TOK_RBRACE));
        }
        cases.push(named_case(&format!("shadowing_levels_{depth}"), atoms));
    }

    for idx in 0..8 {
        let label = format!("lbl_{idx}");
        let atoms = vec![
            ident(&label),
            tok(TOK_COLON),
            tok(TOK_INT),
            ident("x"),
            tok(TOK_SEMICOLON),
            tok(TOK_GOTO),
            ident(&label),
            tok(TOK_SEMICOLON),
        ];
        cases.push(named_case(&format!("label_goto_{idx}"), atoms));
    }

    for idx in 0..8 {
        let fname = format!("kr_{idx}");
        let atoms = vec![
            tok(TOK_INT),
            ident(&fname),
            tok(TOK_LPAREN),
            ident("a"),
            tok(TOK_COMMA),
            ident("b"),
            tok(TOK_RPAREN),
            tok(TOK_INT),
            ident("a"),
            tok(TOK_SEMICOLON),
            tok(TOK_INT),
            ident("b"),
            tok(TOK_SEMICOLON),
            tok(TOK_LBRACE),
            tok(TOK_RETURN),
            ident("a"),
            tok(TOK_SEMICOLON),
            tok(TOK_RBRACE),
        ];
        cases.push(named_case(&format!("kr_style_{idx}"), atoms));
    }

    for idx in 0..8 {
        let atoms = vec![
            ident(&format!("__extension__{idx}")),
            tok(TOK_LPAREN),
            tok(TOK_LBRACE),
            tok(TOK_INT),
            ident(&format!("ext_{idx}")),
            tok(TOK_SEMICOLON),
            tok(TOK_RBRACE),
            tok(TOK_RPAREN),
        ];
        cases.push(named_case(&format!("gnu_extension_{idx}"), atoms));
    }

    for idx in 0..8 {
        let atoms = vec![
            ident("_Generic"),
            tok(TOK_LPAREN),
            ident("x"),
            tok(TOK_COMMA),
            tok(TOK_INT),
            ident(&format!("generic_{idx}")),
            tok(TOK_RPAREN),
            tok(TOK_SEMICOLON),
            ident(&format!("x{idx}")),
            tok(TOK_PLUS),
            ident(&format!("y{idx}")),
            tok(TOK_SEMICOLON),
        ];
        cases.push(named_case(&format!("generic_{idx}"), atoms));
    }

    for idx in 0..8 {
        let atoms = vec![
            tok(TOK_LPAREN),
            tok(TOK_LBRACE),
            tok(TOK_INT),
            ident(&format!("sx_{idx}")),
            tok(TOK_SEMICOLON),
            tok(TOK_RBRACE),
            tok(TOK_RPAREN),
        ];
        cases.push(named_case(&format!("statement_expr_{idx}"), atoms));
    }

    cases
}

#[test]
fn c_sema_scope_adversarial_fixtures_have_exact_node_scope_mapping() {
    let backend = backend();
    let lowered = |fix: &ScopeFixture| {
        let n = fix.tok_types.len() as u32;
        c_sema_scope(
            "tok_types",
            "tok_starts",
            "tok_lens",
            "haystack",
            Expr::u32(fix.haystack.len() as u32),
            Expr::u32(n),
            "out_scope_tree",
        )
    };

    for (name, case) in adversarial_fixtures() {
        let expected = scope_tree_words_for(&case);
        let expected_bytes = emit_u32_bytes(&expected);
        let program = lowered(&case);
        let inputs = case_inputs(&case);
        let cpu_result = vyre_reference::reference_eval(&program, &reference_values(&inputs))
            .expect("CPU reference must run");
        assert_eq!(
            cpu_result.len(),
            1,
            "CPU output should expose one RW buffer"
        );
        let cpu_output = cpu_result[0].to_bytes().to_vec();
        assert_exact_mapping(&name, &expected, &cpu_output);
        assert_eq!(
            cpu_output, expected_bytes,
            "{} CPU output differs from reference bytes",
            name
        );

        let optimized = vyre_foundation::optimizer::optimize(program.clone())
            .expect("registered optimizer must converge");
        let gpu_output = backend
            .dispatch(&optimized, &inputs, &DispatchConfig::default())
            .expect("GPU backend must dispatch");
        assert_eq!(gpu_output.len(), 1);
        assert_eq!(gpu_output[0].len(), expected_bytes.len());
        assert_exact_mapping(&name, &expected, &gpu_output[0]);
        assert_eq!(
            gpu_output[0], expected_bytes,
            "{} GPU output differs from reference bytes",
            name
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn c_sema_scope_random_fixture_parity(
        tokens in proptest::collection::vec(0u8..10, 1..64),
    ) {
        let mut atoms = Vec::new();
        let names = ["alpha", "beta", "gamma", "delta", "epsilon", "z"];
        for code in tokens {
            if code < 4 {
                atoms.push(ident(names[code as usize % names.len()]));
            } else if code == 4 {
                atoms.push(tok(TOK_LBRACE));
            } else if code == 5 {
                atoms.push(tok(TOK_RBRACE));
            } else if code == 6 {
                atoms.push(tok(TOK_LPAREN));
            } else if code == 7 {
                atoms.push(tok(TOK_RPAREN));
            } else if code == 8 {
                atoms.push(tok(TOK_INT));
            } else {
                atoms.push(tok(TOK_SEMICOLON));
            }
        }
        let fixture = fixture("random", &atoms);
        let expected = scope_tree_words_for(&fixture);
        let expected_bytes = emit_u32_bytes(&expected);
        let program = program_for(fixture.tok_types.len() as u32, fixture.haystack.len());
        let outputs = vyre_reference::reference_eval(
            &program,
            &reference_values(&case_inputs(&fixture)),
        ).expect("Reference evaluator must run for random fixture");
        assert_eq!(outputs.len(), 1, "Random fixture must expose one output buffer");
        let cpu_bytes = outputs[0].to_bytes().to_vec();
        assert_eq!(
            cpu_bytes,
            expected_bytes,
            "CPU reference must match deterministic CPU helper for random fixture"
        );
    }
}

#[test]
fn c_sema_scope_boundary_sizes_do_not_panic() {
    let fixture = fixture("boundary", &[tok(TOK_INT), ident("x"), tok(TOK_SEMICOLON)]);
    for n in [0u32, 1, 2, 8, 256, 257] {
        let mut short_tokens = fixture.tok_types.clone();
        let mut short_starts = fixture.tok_starts.clone();
        let mut short_lens = fixture.tok_lens.clone();
        let short_haystack = fixture.haystack.clone();
        if short_tokens.len() > n as usize {
            short_tokens.truncate(n as usize);
            short_starts.truncate(n as usize);
            short_lens.truncate(n as usize);
        } else {
            short_tokens.resize(n as usize, 0);
            short_starts.resize(n as usize, 0);
            short_lens.resize(n as usize, 0);
        }

        let test = ScopeFixture {
            tok_types: short_tokens,
            tok_starts: short_starts,
            tok_lens: short_lens,
            haystack: short_haystack.clone(),
        };

        let program = c_sema_scope(
            "tok_types",
            "tok_starts",
            "tok_lens",
            "haystack",
            Expr::u32(test.haystack.len() as u32),
            Expr::u32(n),
            "out_scope_tree",
        );
        assert!(
            validate(&program).is_empty(),
            "Boundary scope program should validate for n={n}"
        );
        let outputs =
            vyre_reference::reference_eval(&program, &reference_values(&case_inputs(&test)))
                .expect("Boundary fixture must execute in CPU reference");
        assert_eq!(outputs.len(), 1);
        let _ = outputs[0].to_bytes();
    }
}
