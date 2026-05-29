//! End-to-end execution oracle for the Rust nano-subset: lower the AST to a
//! Vyre `Program`, run it on the pure-Rust reference interpreter, and check the
//! result against two independent oracles:
//!   1. a direct AST tree-walk interpreter (independent of lowering), and
//!   2. real rustc: compile the module + a `main` that prints `entry(args)`,
//!      run it, and compare stdout.
//!
//! Programs are generated overflow-free and division-free so i32 semantics are
//! unambiguous across all three (no Rust debug-overflow panic, no div-by-zero).

#![forbid(unsafe_code)]

use std::collections::HashMap;

use vyre_libs::parsing::rust::lex::lexer::core::lex;
use vyre_libs::parsing::rust::lower::lower;
use vyre_libs::parsing::rust::parse::{parse, Expr, Module, Stmt};
use vyre_libs::parsing::rust::sema::{resolve, typeck, BindingId, Resolution};
use vyre_reference::value::Value;

use vyre_libs::parsing::rust::lex::tokens::{EQ, LT, MINUS, PLUS, STAR};

// ---------------------------------------------------------------------------
// Pipeline helpers
// ---------------------------------------------------------------------------

fn frontend(src: &str) -> (Module, Resolution) {
    let bytes = src.as_bytes();
    let tokens = lex(bytes).expect("lex");
    let module = parse(bytes, &tokens).expect("parse");
    let resolution = resolve(&module, bytes).expect("resolve");
    typeck(&module, bytes, &resolution).expect("typeck");
    (module, resolution)
}

fn value_to_i32(v: &Value) -> i32 {
    match v {
        Value::I32(x) => *x,
        Value::U32(x) => *x as i32,
        Value::Bool(b) => i32::from(*b),
        Value::Bytes(bytes) => i32::from_le_bytes(bytes[..4].try_into().expect("4 bytes")),
        other => panic!("unexpected output value {other:?}"),
    }
}

/// Lower `src`'s entry function and run it on the reference interpreter.
fn ir_exec(src: &str, inputs: &[i32]) -> i32 {
    let (module, resolution) = frontend(src);
    let program = lower(&module, &resolution).expect("lower");
    let values: Vec<Value> = inputs.iter().map(|&x| Value::I32(x)).collect();
    let out = vyre_reference::reference_eval(&program, &values).expect("reference_eval");
    assert_eq!(out.len(), 1, "entry must produce exactly one output");
    value_to_i32(&out[0])
}

// ---------------------------------------------------------------------------
// Independent AST tree-walk interpreter (oracle #1)
// ---------------------------------------------------------------------------

fn global_def_to_id(resolution: &Resolution) -> HashMap<u32, BindingId> {
    resolution
        .bindings
        .iter()
        .enumerate()
        .map(|(id, b)| (b.def_offset, id))
        .collect()
}

enum Flow {
    Return(i32),
    Fall,
}

struct Ev<'a> {
    module: &'a Module,
    resolution: &'a Resolution,
    def_to_id: &'a HashMap<u32, BindingId>,
}

impl Ev<'_> {
    fn run_fn(&self, idx: usize, args: &[i32]) -> i32 {
        let func = &self.module.functions[idx];
        let mut env: HashMap<BindingId, i32> = HashMap::new();
        for (i, (offset, _)) in func.params.iter().enumerate() {
            env.insert(self.def_to_id[offset], args[i]);
        }
        match self.exec(&func.body, &mut env) {
            Flow::Return(v) => v,
            Flow::Fall => 0,
        }
    }

    fn exec(&self, stmts: &[Stmt], env: &mut HashMap<BindingId, i32>) -> Flow {
        for stmt in stmts {
            match stmt {
                Stmt::Let { name, init, .. } => {
                    let v = self.eval_int(init, env);
                    env.insert(self.def_to_id[name], v);
                }
                Stmt::Return(Some(e)) => return Flow::Return(self.eval_int(e, env)),
                Stmt::Return(None) => return Flow::Return(0),
                Stmt::Expr(Expr::If { cond, then_block, else_block }) => {
                    let taken = if self.eval_bool(cond, env) {
                        Some(then_block.as_ref())
                    } else {
                        else_block.as_deref()
                    };
                    if let Some(Expr::Block(body)) = taken {
                        if let Flow::Return(v) = self.exec(body, env) {
                            return Flow::Return(v);
                        }
                    }
                }
                Stmt::Expr(_) => {}
            }
        }
        Flow::Fall
    }

    fn eval_int(&self, e: &Expr, env: &HashMap<BindingId, i32>) -> i32 {
        match e {
            Expr::LiteralInt(_, v) => *v as i32,
            Expr::Var(off) => env[&self.resolution.uses[off]],
            Expr::Binary { op, lhs, rhs } => {
                let (l, r) = (self.eval_int(lhs, env), self.eval_int(rhs, env));
                match *op {
                    PLUS => l.wrapping_add(r),
                    MINUS => l.wrapping_sub(r),
                    STAR => l.wrapping_mul(r),
                    other => panic!("non-arithmetic op {other} in integer position"),
                }
            }
            Expr::Call { name, args } => {
                let idx = self.resolution.calls[name];
                let a: Vec<i32> = args.iter().map(|x| self.eval_int(x, env)).collect();
                self.run_fn(idx, &a)
            }
            other => panic!("unexpected integer expr {other:?}"),
        }
    }

    fn eval_bool(&self, e: &Expr, env: &HashMap<BindingId, i32>) -> bool {
        match e {
            Expr::LiteralBool(_, b) => *b,
            Expr::Binary { op, lhs, rhs } => {
                let (l, r) = (self.eval_int(lhs, env), self.eval_int(rhs, env));
                match *op {
                    LT => l < r,
                    EQ => l == r,
                    other => panic!("non-comparison op {other} in bool position"),
                }
            }
            other => panic!("unexpected bool expr {other:?}"),
        }
    }
}

fn ast_interp(src: &str, inputs: &[i32]) -> i32 {
    let (module, resolution) = frontend(src);
    let def_to_id = global_def_to_id(&resolution);
    let ev = Ev { module: &module, resolution: &resolution, def_to_id: &def_to_id };
    ev.run_fn(module.functions.len() - 1, inputs)
}

// ---------------------------------------------------------------------------
// Program generator (overflow-free, division-free i32 nano programs)
// ---------------------------------------------------------------------------

struct Gen {
    state: u64,
}
impl Gen {
    fn new(seed: u64) -> Self {
        Self { state: seed ^ 0x517C_C1B7_2722_0A95 }
    }
    fn next(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }
    /// Small arithmetic expression over `nvars` in-scope `vN` variables. When
    /// `calls`, may emit a call `h(e, e)` to the 2-arg leaf helper.
    fn expr(&mut self, nvars: usize, depth: u32, calls: bool) -> String {
        if calls && depth > 0 && self.next() % 4 == 0 {
            return format!("h({}, {})", self.expr(nvars, depth - 1, calls), self.expr(nvars, depth - 1, calls));
        }
        if depth == 0 || self.next() % 3 == 0 {
            if self.next() % 2 == 0 {
                format!("v{}", (self.next() as usize) % nvars)
            } else {
                format!("{}", self.next() % 6) // literal 0..=5
            }
        } else {
            let op = ["+", "-", "*"][(self.next() % 3) as usize];
            format!("({} {} {})", self.expr(nvars, depth - 1, calls), op, self.expr(nvars, depth - 1, calls))
        }
    }
    fn cond(&mut self, nvars: usize) -> String {
        let op = if self.next() % 2 == 0 { "<" } else { "==" };
        format!("{} {} {}", self.expr(nvars, 1, false), op, self.expr(nvars, 1, false))
    }
}

/// Generate `(source, param_count)`. Half the programs include a leaf helper
/// `h(a, b)` whose calls the entry inlines. Params `v0..vP`, some `let`
/// bindings, then a straight-line or branching return. Magnitudes stay small so
/// no i32 overflow occurs and the verdict is unambiguous.
fn gen_program(seed: u64) -> (String, usize) {
    let mut g = Gen::new(seed);
    let with_helper = g.next() % 2 == 0;
    let mut module = String::new();
    if with_helper {
        module.push_str(&format!(
            "fn h(v0: i32, v1: i32) -> i32 {{ return {}; }}\n",
            g.expr(2, 2, false)
        ));
    }
    let nparams = 1 + (g.next() % 3) as usize; // 1..=3
    let mut nvars = nparams;
    let params: Vec<String> = (0..nparams).map(|i| format!("v{i}: i32")).collect();
    module.push_str(&format!("fn f({}) -> i32 {{", params.join(", ")));
    let nlets = (g.next() % 3) as usize;
    for _ in 0..nlets {
        module.push_str(&format!(" let v{}: i32 = {};", nvars, g.expr(nvars, 2, with_helper)));
        nvars += 1;
    }
    if g.next() % 2 == 0 {
        module.push_str(&format!(" return {}; }}", g.expr(nvars, 2, with_helper)));
    } else {
        module.push_str(&format!(
            " if {} {{ return {}; }} else {{ return {}; }} }}",
            g.cond(nvars),
            g.expr(nvars, 2, with_helper),
            g.expr(nvars, 2, with_helper)
        ));
    }
    (module, nparams)
}

fn gen_inputs(seed: u64, n: usize) -> Vec<i32> {
    let mut g = Gen::new(seed ^ 0xABCD_1234);
    (0..n).map(|_| (g.next() % 19) as i32 - 9).collect() // -9..=9
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn lowered_ir_matches_ast_interpreter() {
    for seed in 0..2000u64 {
        let (src, nparams) = gen_program(seed);
        for input_seed in 0..3u64 {
            let inputs = gen_inputs(seed.wrapping_mul(7).wrapping_add(input_seed), nparams);
            let expected = ast_interp(&src, &inputs);
            let got = ir_exec(&src, &inputs);
            assert_eq!(
                got, expected,
                "lowered IR diverged from AST interpreter at seed {seed} inputs {inputs:?}:\n  {src}"
            );
        }
    }
}

#[test]
fn curated_programs_execute_correctly() {
    let cases: &[(&str, &[i32], i32)] = &[
        ("fn f(a: i32, b: i32) -> i32 { return a + b; }", &[3, 4], 7),
        ("fn f(a: i32, b: i32) -> i32 { return a - b; }", &[10, 3], 7),
        ("fn f(a: i32) -> i32 { let b: i32 = a * 2; return b + 1; }", &[5], 11),
        ("fn f(a: i32, b: i32) -> i32 { if a < b { return b; } else { return a; } }", &[3, 9], 9),
        ("fn f(a: i32, b: i32) -> i32 { if a < b { return b; } else { return a; } }", &[9, 3], 9),
        ("fn f(a: i32) -> i32 { if a == 0 { return 100; } else { return a; } }", &[0], 100),
        ("fn g(a: i32, b: i32) -> i32 { return a + b; } fn f(a: i32) -> i32 { return g(a, 10); }", &[5], 15),
        ("fn g(a: i32) -> i32 { let d: i32 = a * a; return d - 1; } fn f(a: i32, b: i32) -> i32 { return g(a) + b; }", &[4, 2], 17),
    ];
    for (src, inputs, expected) in cases {
        assert_eq!(ir_exec(src, inputs), *expected, "{src} with {inputs:?}");
        assert_eq!(ast_interp(src, inputs), *expected, "AST interp: {src} with {inputs:?}");
    }
}

// ---------------------------------------------------------------------------
// rustc compile+run ground truth (oracle #2)
// ---------------------------------------------------------------------------

fn rustc_run(src: &str, inputs: &[i32]) -> Option<i32> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("vyre_lower_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let args = inputs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
    let main = format!("\nfn main() {{ println!(\"{{}}\", f({args})); }}\n");
    let rs = dir.join("m.rs");
    std::fs::write(&rs, format!("{src}{main}")).expect("write");
    let exe = dir.join("m");
    let build = std::process::Command::new("rustc")
        .args(["--edition", "2021", "-O", "--cap-lints", "allow", "-o"])
        .arg(&exe)
        .arg(&rs)
        .output()
        .expect("rustc on PATH");
    let result = if build.status.success() {
        std::process::Command::new(&exe)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<i32>().ok())
    } else {
        None
    };
    let _ = std::fs::remove_dir_all(&dir);
    result
}

#[test]
fn lowered_ir_matches_rustc_execution() {
    let mut checked = 0;
    for seed in 0..80u64 {
        let (src, nparams) = gen_program(seed);
        let inputs = gen_inputs(seed.wrapping_mul(13).wrapping_add(1), nparams);
        let Some(expected) = rustc_run(&src, &inputs) else { continue };
        let got = ir_exec(&src, &inputs);
        assert_eq!(
            got, expected,
            "lowered IR diverged from rustc-run at seed {seed} inputs {inputs:?}:\n  {src}"
        );
        checked += 1;
    }
    assert!(checked >= 40, "expected most generated programs to compile+run under rustc, got {checked}");
}
