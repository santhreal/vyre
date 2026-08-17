//! Sequential mathematical witnesses for logic compilation, categories, ZX rewrites, and causal adjustment sets.

/// One gate in a d-DNNF DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DnnfGate {
    /// Constant True.
    True,
    /// Constant False.
    False,
    /// Boolean literal with variable index and polarity (true = positive, false = negated).
    Literal(u32, bool),
    /// Decomposable AND gate over child gate indices.
    And(Vec<u32>),
    /// Deterministic OR gate over child gate indices.
    Or(Vec<u32>),
}

/// d-DNNF DAG: gate list (root is the last entry by convention) + variable count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DnnfDag {
    /// Gates indexed by id.
    pub gates: Vec<DnnfGate>,
    /// Number of variables the formula uses.
    pub num_vars: u32,
}

impl DnnfDag {
    /// Root gate id (the last entry in `gates` by convention).
    #[must_use]
    pub fn root(&self) -> u32 {
        self.gates.len().saturating_sub(1) as u32
    }
}

/// Compile a CNF formula into a d-DNNF DAG via Shannon decomposition.
#[must_use]
pub fn compile_dnnf_witness(
    clauses: &[Vec<(u32, bool)>],
    num_vars: u32,
    max_depth: u32,
) -> DnnfDag {
    let mut dag = DnnfDag {
        gates: Vec::new(),
        num_vars,
    };
    compile_recursive(&mut dag, clauses, num_vars, 0, max_depth);
    dag
}

fn smoothed_true(dag: &mut DnnfDag, num_vars: u32, var: u32) -> u32 {
    let mut conjuncts = Vec::new();
    for v in var..num_vars {
        let pos = dag.gates.len() as u32;
        dag.gates.push(DnnfGate::Literal(v, true));
        let neg = dag.gates.len() as u32;
        dag.gates.push(DnnfGate::Literal(v, false));
        let or_gate = dag.gates.len() as u32;
        dag.gates.push(DnnfGate::Or(vec![pos, neg]));
        conjuncts.push(or_gate);
    }
    if conjuncts.is_empty() {
        let id = dag.gates.len() as u32;
        dag.gates.push(DnnfGate::True);
        id
    } else if conjuncts.len() == 1 {
        conjuncts[0]
    } else {
        let id = dag.gates.len() as u32;
        dag.gates.push(DnnfGate::And(conjuncts));
        id
    }
}

fn compile_recursive(
    dag: &mut DnnfDag,
    clauses: &[Vec<(u32, bool)>],
    num_vars: u32,
    var: u32,
    remaining_depth: u32,
) -> u32 {
    if clauses.is_empty() {
        return smoothed_true(dag, num_vars, var);
    }
    if clauses.iter().any(|c| c.is_empty()) {
        let id = dag.gates.len() as u32;
        dag.gates.push(DnnfGate::False);
        return id;
    }
    if var >= num_vars || remaining_depth == 0 {
        return smoothed_true(dag, num_vars, var);
    }

    let pos_clauses = simplify_clauses(clauses, var, true);
    let neg_clauses = simplify_clauses(clauses, var, false);

    let pos_child = compile_recursive(dag, &pos_clauses, num_vars, var + 1, remaining_depth - 1);
    let neg_child = compile_recursive(dag, &neg_clauses, num_vars, var + 1, remaining_depth - 1);

    let lit_pos = dag.gates.len() as u32;
    dag.gates.push(DnnfGate::Literal(var, true));
    let and_pos = dag.gates.len() as u32;
    dag.gates.push(DnnfGate::And(vec![lit_pos, pos_child]));

    let lit_neg = dag.gates.len() as u32;
    dag.gates.push(DnnfGate::Literal(var, false));
    let and_neg = dag.gates.len() as u32;
    dag.gates.push(DnnfGate::And(vec![lit_neg, neg_child]));

    let or_root = dag.gates.len() as u32;
    dag.gates.push(DnnfGate::Or(vec![and_pos, and_neg]));
    or_root
}

fn simplify_clauses(
    clauses: &[Vec<(u32, bool)>],
    var: u32,
    assignment: bool,
) -> Vec<Vec<(u32, bool)>> {
    let mut out = Vec::new();
    for clause in clauses {
        let mut satisfied = false;
        let mut simplified = Vec::new();
        for &(v, polarity) in clause {
            if v == var {
                if polarity == assignment {
                    satisfied = true;
                    break;
                }
            } else {
                simplified.push((v, polarity));
            }
        }
        if !satisfied {
            out.push(simplified);
        }
    }
    out
}

/// Sequential bottom-up model count of a d-DNNF DAG.
#[must_use]
pub fn dnnf_model_count_witness(dag: &DnnfDag) -> u64 {
    let mut counts: Vec<u64> = Vec::with_capacity(dag.gates.len());
    for gate in &dag.gates {
        let count = match gate {
            DnnfGate::True => 1,
            DnnfGate::False => 0,
            DnnfGate::Literal(_, _) => 1,
            DnnfGate::And(children) => {
                let mut product = 1u64;
                for &child in children {
                    let child_count = counts.get(child as usize).copied().unwrap_or(0);
                    product = product.saturating_mul(child_count);
                }
                product
            }
            DnnfGate::Or(children) => {
                let mut sum = 0u64;
                for &child in children {
                    let child_count = counts.get(child as usize).copied().unwrap_or(0);
                    sum = sum.saturating_add(child_count);
                }
                sum
            }
        };
        counts.push(count);
    }
    counts.last().copied().unwrap_or(0)
}

/// Whether the d-DNNF formula has at least one satisfying assignment.
#[must_use]
pub fn dnnf_is_satisfiable_witness(dag: &DnnfDag) -> bool {
    dnnf_model_count_witness(dag) > 0
}

/// Whether every assignment over `num_vars` variables is a model.
#[must_use]
pub fn dnnf_is_tautology_witness(dag: &DnnfDag, num_vars: u32) -> bool {
    if num_vars >= 64 {
        return false;
    }
    dnnf_model_count_witness(dag) == (1u64 << num_vars)
}

/// Finite category: object count `n` and row-major `n × n` Hom-set cardinalities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteCategory {
    /// Number of objects.
    pub n: u32,
    /// Row-major Hom-set cardinalities.
    pub hom_size: Vec<u32>,
}

impl FiniteCategory {
    /// Discrete category on `n` objects: `|Hom(s, t)| = 1` if s == t, else 0.
    #[must_use]
    pub fn discrete(n: u32) -> Self {
        let n_us = n as usize;
        let mut hom_size = vec![0u32; n_us * n_us];
        for i in 0..n_us {
            hom_size[i * n_us + i] = 1;
        }
        Self { n, hom_size }
    }

    /// Cardinality of `Hom(source, target)`.
    #[must_use]
    pub fn hom(&self, source: u32, target: u32) -> u32 {
        if source >= self.n || target >= self.n {
            return 0;
        }
        self.hom_size
            .get((source * self.n + target) as usize)
            .copied()
            .unwrap_or(0)
    }
}

/// Functor between finite categories: object map only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteFunctor {
    /// `object_map[c] = F(c)` for c < domain.n.
    pub object_map: Vec<u32>,
}

impl FiniteFunctor {
    /// Identity functor on an `n`-object category.
    #[must_use]
    pub fn identity(n: u32) -> Self {
        Self {
            object_map: (0..n).collect(),
        }
    }

    /// Apply F to an object index.
    #[must_use]
    pub fn apply(&self, c: u32) -> u32 {
        self.object_map.get(c as usize).copied().unwrap_or(u32::MAX)
    }
}

/// Result of an adjoint-pair check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdjointPair {
    /// Whether the bijection holds at every `(c, d)`.
    pub is_adjoint: bool,
    /// First failing `(c, d)` pair if `is_adjoint` is false.
    pub witness: Option<(u32, u32)>,
}

/// Check `F ⊣ G` on finite categories `C, D`.
#[must_use]
pub fn adjoint_pair_witness(
    c_cat: &FiniteCategory,
    d_cat: &FiniteCategory,
    f: &FiniteFunctor,
    g: &FiniteFunctor,
) -> AdjointPair {
    if f.object_map.len() as u32 != c_cat.n || g.object_map.len() as u32 != d_cat.n {
        return AdjointPair {
            is_adjoint: false,
            witness: Some((0, 0)),
        };
    }

    for c in 0..c_cat.n {
        for d in 0..d_cat.n {
            let lhs = d_cat.hom(f.apply(c), d);
            let rhs = c_cat.hom(c, g.apply(d));
            if lhs != rhs {
                return AdjointPair {
                    is_adjoint: false,
                    witness: Some((c, d)),
                };
            }
        }
    }
    AdjointPair {
        is_adjoint: true,
        witness: None,
    }
}

/// Which universal construction a Kan extension takes over the preimage.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KanDirection {
    /// Colimit / sum (empty case = 0).
    Left,
    /// Limit / product (empty case = 1).
    Right,
}

impl KanDirection {
    const fn identity(self) -> u32 {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }

    const fn fold(self, accumulated: u32, value: u32) -> u32 {
        match self {
            Self::Left => accumulated.saturating_add(value),
            Self::Right => accumulated.saturating_mul(value),
        }
    }
}

/// Cardinality of the Kan extension at one object of the codomain.
#[must_use]
pub fn kan_extension_at_witness(
    direction: KanDirection,
    k: &FiniteFunctor,
    f_image: &[u32],
    c: u32,
) -> u32 {
    let mut accumulated = direction.identity();
    for (m, &image) in k.object_map.iter().enumerate() {
        if image == c {
            if let Some(&val) = f_image.get(m) {
                accumulated = direction.fold(accumulated, val);
            }
        }
    }
    accumulated
}

/// Cardinality of the Kan extension at every object of a codomain of size `c_n`.
#[must_use]
pub fn kan_extension_table_witness(
    direction: KanDirection,
    k: &FiniteFunctor,
    f_image: &[u32],
    c_n: u32,
) -> Vec<u32> {
    (0..c_n)
        .map(|c| kan_extension_at_witness(direction, k, f_image, c))
        .collect()
}

/// Yoneda embedding of object `x`: `[|Hom(c_0, x)|, |Hom(c_1, x)|, ...]`.
#[must_use]
pub fn yoneda_embedding_witness(category: &FiniteCategory, x: u32) -> Vec<u32> {
    (0..category.n).map(|c| category.hom(c, x)).collect()
}

/// Presheaf natural transformation count via Yoneda lemma `|Nat(Hom(-, x), F)| = |F(x)|`.
#[must_use]
pub fn natural_transformation_count_witness(
    _category: &FiniteCategory,
    _x: u32,
    f_at_x: u32,
) -> u32 {
    f_at_x
}

/// Spider color in a ZX diagram: Z (green) or X (red).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZxColor {
    /// Z spider.
    Z,
    /// X spider.
    X,
}

impl ZxColor {
    /// Opposite color.
    #[must_use]
    #[inline]
    pub const fn flip(self) -> Self {
        match self {
            Self::Z => Self::X,
            Self::X => Self::Z,
        }
    }
}

/// One ZX spider with color and phase numerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZxSpider {
    /// Z or X.
    pub color: ZxColor,
    /// Phase numerator modulo diagram phase denominator.
    pub phase_num: u32,
}

/// ZX diagram: spider list and edge multiset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZxDiagram {
    /// Phase denominator (> 0).
    pub phase_denom: u32,
    /// Spider list.
    pub spiders: Vec<ZxSpider>,
    /// Undirected edges.
    pub edges: Vec<(u32, u32)>,
}

/// Apply spider fusion (S1): merge adjacent same-color spiders until fixpoint.
#[must_use]
pub fn zx_spider_fusion_witness(mut diagram: ZxDiagram) -> ZxDiagram {
    if diagram.phase_denom == 0 {
        return diagram;
    }
    loop {
        let merge_pair = diagram.edges.iter().copied().find(|&(u, v)| {
            u != v
                && (u as usize) < diagram.spiders.len()
                && (v as usize) < diagram.spiders.len()
                && diagram.spiders[u as usize].color == diagram.spiders[v as usize].color
        });
        let Some((u, v)) = merge_pair else {
            break;
        };
        let combined = (diagram.spiders[u as usize].phase_num
            + diagram.spiders[v as usize].phase_num)
            % diagram.phase_denom;
        diagram.spiders[u as usize].phase_num = combined;

        let mut next_edges = Vec::with_capacity(diagram.edges.len().saturating_sub(1));
        for &(a, b) in &diagram.edges {
            if (a == u && b == v) || (a == v && b == u) {
                continue;
            }
            let new_a = if a == v { u } else { a };
            let new_b = if b == v { u } else { b };
            next_edges.push((new_a, new_b));
        }
        diagram.edges = next_edges;
        diagram.spiders.remove(v as usize);
        for e in &mut diagram.edges {
            if e.0 > v {
                e.0 -= 1;
            }
            if e.1 > v {
                e.1 -= 1;
            }
        }
    }
    diagram
}

/// Apply identity removal (S2): drop phase-0 spiders of degree 2 between same-color neighbors.
#[must_use]
pub fn zx_identity_removal_witness(mut diagram: ZxDiagram) -> ZxDiagram {
    loop {
        let mut removable: Option<u32> = None;
        for v in 0..diagram.spiders.len() {
            let s = diagram.spiders[v];
            if s.phase_num != 0 {
                continue;
            }
            let incident: Vec<(u32, u32)> = diagram
                .edges
                .iter()
                .copied()
                .filter(|&(a, b)| a == v as u32 || b == v as u32)
                .collect();
            if incident.len() != 2 {
                continue;
            }
            let has_self_loop = incident.iter().any(|&(a, b)| a == b);
            if has_self_loop {
                continue;
            }
            let other_endpoint =
                |edge: (u32, u32)| if edge.0 == v as u32 { edge.1 } else { edge.0 };
            let n1 = other_endpoint(incident[0]);
            let n2 = other_endpoint(incident[1]);
            if (n1 as usize) < diagram.spiders.len()
                && (n2 as usize) < diagram.spiders.len()
                && diagram.spiders[n1 as usize].color == s.color
                && diagram.spiders[n2 as usize].color == s.color
            {
                removable = Some(v as u32);
                break;
            }
        }
        let Some(v) = removable else {
            break;
        };
        let other_endpoint = |edge: (u32, u32)| if edge.0 == v { edge.1 } else { edge.0 };
        let incident: Vec<(u32, u32)> = diagram
            .edges
            .iter()
            .copied()
            .filter(|&(a, b)| a == v || b == v)
            .collect();
        let n1 = other_endpoint(incident[0]);
        let n2 = other_endpoint(incident[1]);

        let mut next_edges = Vec::with_capacity(diagram.edges.len().saturating_sub(1));
        for &(a, b) in &diagram.edges {
            if a == v || b == v {
                continue;
            }
            next_edges.push((a, b));
        }
        next_edges.push((n1, n2));
        diagram.edges = next_edges;
        diagram.spiders.remove(v as usize);
        for e in &mut diagram.edges {
            if e.0 > v {
                e.0 -= 1;
            }
            if e.1 > v {
                e.1 -= 1;
            }
        }
    }
    diagram
}

/// Apply color change (Hadamard conjugation) to spider `v`.
pub fn zx_color_change_witness(diagram: &mut ZxDiagram, v: u32) {
    if let Some(s) = diagram.spiders.get_mut(v as usize) {
        s.color = s.color.flip();
    }
}

/// Joint fixpoint of spider fusion and identity removal.
#[must_use]
pub fn zx_simplified_diagram_witness(diagram: ZxDiagram) -> ZxDiagram {
    let mut current = diagram;
    loop {
        let before = current.spiders.len();
        current = zx_identity_removal_witness(zx_spider_fusion_witness(current));
        if current.spiders.len() == before {
            return current;
        }
    }
}

/// Return whether ordering pass `t` before pass `o` is acyclic.
#[must_use]
pub fn adjustment_set_ordering_is_safe_witness(
    adj: &[u32],
    treatment: u32,
    outcome: u32,
    n: u32,
) -> bool {
    let Some(cells) = (n as usize).checked_mul(n as usize) else {
        return false;
    };
    if adj.len() != cells || treatment >= n || outcome >= n {
        return false;
    }
    if treatment == outcome {
        return true;
    }
    let closure = dense_transitive_closure(adj, n as usize);
    closure[(outcome * n + treatment) as usize] == 0
}

/// For each pass index `i`, return strict descendants reachable in the influence graph.
#[must_use]
pub fn adjustment_set_pass_descendants_witness(adj: &[u32], n: u32) -> Vec<Vec<u32>> {
    if n == 0 {
        return Vec::new();
    }
    let Some(cells) = (n as usize).checked_mul(n as usize) else {
        return Vec::new();
    };
    if adj.len() != cells {
        return Vec::new();
    }
    let closure = dense_transitive_closure(adj, n as usize);
    let mut out = vec![Vec::new(); n as usize];
    for i in 0..n {
        for j in 0..n {
            if i != j && closure[(i * n + j) as usize] != 0 {
                out[i as usize].push(j);
            }
        }
    }
    for row in &mut out {
        row.sort_unstable();
    }
    out
}

fn dense_transitive_closure(adj: &[u32], n: usize) -> Vec<u32> {
    let mut reach = adj.to_vec();
    for k in 0..n {
        for i in 0..n {
            if reach[i * n + k] != 0 {
                for j in 0..n {
                    reach[i * n + j] |= reach[k * n + j];
                }
            }
        }
    }
    reach
}

/// Sequential mathematical witness for categorical pass composition.
///
/// Collapses mapping `g` followed by mapping `f` into one column mapping
/// and applies it to `view_in`.
#[must_use]
pub fn compose_passes_witness(
    view_in: &[u32],
    mapping_g: &[u32],
    n_mid: u32,
    mapping_f: &[u32],
    n_out: u32,
) -> Vec<u32> {
    let mut out = Vec::new();
    let mut combined = Vec::new();
    compose_passes_witness_into(
        view_in,
        mapping_g,
        n_mid,
        mapping_f,
        n_out,
        &mut combined,
        &mut out,
    );
    out
}

/// Sequential mathematical witness for categorical pass composition writing into caller storage.
pub fn compose_passes_witness_into(
    view_in: &[u32],
    mapping_g: &[u32],
    n_mid: u32,
    mapping_f: &[u32],
    n_out: u32,
    combined: &mut Vec<u32>,
    out: &mut Vec<u32>,
) {
    assert_eq!(view_in.len(), mapping_g.len());
    assert_eq!(mapping_f.len(), n_mid as usize);
    combined.clear();
    combined.reserve(mapping_g.len());
    combined.extend(mapping_g.iter().map(|&mid_dst| mapping_f[mid_dst as usize]));
    super::graph::functor_apply_witness_into(view_in, combined, n_out, out);
}

/// Sequential mathematical witness for categorical identity functor.
#[must_use]
pub fn identity_functor_witness(n_cols: u32) -> Vec<u32> {
    let mut out = Vec::new();
    identity_functor_witness_into(n_cols, &mut out);
    out
}

/// Sequential mathematical witness for categorical identity functor writing into caller storage.
pub fn identity_functor_witness_into(n_cols: u32, out: &mut Vec<u32>) {
    out.clear();
    out.reserve(n_cols as usize);
    out.extend(0..n_cols);
}

/// Test whether two pass functors commute on a given input row.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn passes_commute_on_witness(
    view_in: &[u32],
    mapping_a: &[u32],
    n_mid_a: u32,
    mapping_b_after_a: &[u32],
    mapping_b: &[u32],
    n_mid_b: u32,
    mapping_a_after_b: &[u32],
    n_out: u32,
) -> bool {
    let ab = compose_passes_witness(view_in, mapping_a, n_mid_a, mapping_b_after_a, n_out);
    let ba = compose_passes_witness(view_in, mapping_b, n_mid_b, mapping_a_after_b, n_out);
    ab == ba
}

/// Evaluation context queried by the rule condition reference witness.
pub trait RuleEvaluationContextWitness {
    /// Number of times pattern `pattern_id` matched in the current record.
    fn pattern_count(&self, _pattern_id: u32) -> u32 {
        0
    }

    /// File size in bytes for the current record.
    fn file_size(&self) -> u64 {
        0
    }

    /// Resolve a named field value.
    fn field_value(&self, _name: &str) -> Option<&str> {
        None
    }
}

/// Canonical neutral rule condition representation for reference witnesses.
#[derive(Debug, Clone)]
pub enum RuleConditionWitness {
    /// True when the pattern has any match state.
    PatternExists {
        /// Pattern table index.
        pattern_id: u32,
    },
    /// True when pattern count is strictly greater than threshold.
    PatternCountGt {
        /// Pattern table index.
        pattern_id: u32,
        /// Exclusive lower bound.
        threshold: u32,
    },
    /// True when pattern count is greater than or equal to threshold.
    PatternCountGte {
        /// Pattern table index.
        pattern_id: u32,
        /// Inclusive lower bound.
        threshold: u32,
    },
    /// True when file size < threshold.
    FileSizeLt(u64),
    /// True when file size <= threshold.
    FileSizeLte(u64),
    /// True when file size > threshold.
    FileSizeGt(u64),
    /// True when file size >= threshold.
    FileSizeGte(u64),
    /// True when file size == threshold.
    FileSizeEq(u64),
    /// True when file size != threshold.
    FileSizeNe(u64),
    /// Constant true leaf.
    LiteralTrue,
    /// Constant false leaf.
    LiteralFalse,
    /// True when text matched by field satisfies regex pattern.
    RegexMatch {
        /// Source field name.
        field: std::sync::Arc<str>,
        /// Regular expression pattern.
        pattern: std::sync::Arc<str>,
    },
    /// True when haystack contains needle.
    SubstringMatch {
        /// Source text or field name.
        haystack: std::sync::Arc<str>,
        /// Required substring.
        needle: std::sync::Arc<str>,
    },
    /// True when value starts with prefix.
    PrefixMatch {
        /// Source text or field name.
        value: std::sync::Arc<str>,
        /// Required prefix.
        prefix: std::sync::Arc<str>,
    },
    /// True when value ends with suffix.
    SuffixMatch {
        /// Source text or field name.
        value: std::sync::Arc<str>,
        /// Required suffix.
        suffix: std::sync::Arc<str>,
    },
    /// True when value falls inside inclusive range.
    RangeMatch {
        /// Observed value.
        value: u64,
        /// Inclusive lower bound.
        min: u64,
        /// Inclusive upper bound.
        max: u64,
    },
    /// True when value is in set.
    SetMembership {
        /// Candidate value.
        value: std::sync::Arc<str>,
        /// Accepted set members.
        set: smallvec::SmallVec<[std::sync::Arc<str>; 4]>,
    },
    /// True when context field is in set.
    FieldInSet {
        /// Context field name to look up.
        field: std::sync::Arc<str>,
        /// Accepted set members.
        set: smallvec::SmallVec<[std::sync::Arc<str>; 4]>,
    },
    /// Extension-declared rule condition.
    Opaque(std::sync::Arc<dyn vyre_foundation::extension::RuleConditionExt>),
}

impl PartialEq for RuleConditionWitness {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::PatternExists { pattern_id: a }, Self::PatternExists { pattern_id: b }) => {
                a == b
            }
            (
                Self::PatternCountGt {
                    pattern_id: a,
                    threshold: ta,
                },
                Self::PatternCountGt {
                    pattern_id: b,
                    threshold: tb,
                },
            ) => a == b && ta == tb,
            (
                Self::PatternCountGte {
                    pattern_id: a,
                    threshold: ta,
                },
                Self::PatternCountGte {
                    pattern_id: b,
                    threshold: tb,
                },
            ) => a == b && ta == tb,
            (Self::FileSizeLt(a), Self::FileSizeLt(b)) => a == b,
            (Self::FileSizeLte(a), Self::FileSizeLte(b)) => a == b,
            (Self::FileSizeGt(a), Self::FileSizeGt(b)) => a == b,
            (Self::FileSizeGte(a), Self::FileSizeGte(b)) => a == b,
            (Self::FileSizeEq(a), Self::FileSizeEq(b)) => a == b,
            (Self::FileSizeNe(a), Self::FileSizeNe(b)) => a == b,
            (Self::LiteralTrue, Self::LiteralTrue) => true,
            (Self::LiteralFalse, Self::LiteralFalse) => true,
            (
                Self::RegexMatch {
                    field: af,
                    pattern: ap,
                },
                Self::RegexMatch {
                    field: bf,
                    pattern: bp,
                },
            ) => af == bf && ap == bp,
            (
                Self::SubstringMatch {
                    haystack: ah,
                    needle: an,
                },
                Self::SubstringMatch {
                    haystack: bh,
                    needle: bn,
                },
            ) => ah == bh && an == bn,
            (
                Self::PrefixMatch {
                    value: av,
                    prefix: ap,
                },
                Self::PrefixMatch {
                    value: bv,
                    prefix: bp,
                },
            ) => av == bv && ap == bp,
            (
                Self::SuffixMatch {
                    value: av,
                    suffix: as_,
                },
                Self::SuffixMatch {
                    value: bv,
                    suffix: bs,
                },
            ) => av == bv && as_ == bs,
            (
                Self::RangeMatch {
                    value: av,
                    min: amin,
                    max: amax,
                },
                Self::RangeMatch {
                    value: bv,
                    min: bmin,
                    max: bmax,
                },
            ) => av == bv && amin == bmin && amax == bmax,
            (
                Self::SetMembership {
                    value: av,
                    set: aset,
                },
                Self::SetMembership {
                    value: bv,
                    set: bset,
                },
            ) => av == bv && aset == bset,
            (
                Self::FieldInSet {
                    field: af,
                    set: aset,
                },
                Self::FieldInSet {
                    field: bf,
                    set: bset,
                },
            ) => af == bf && aset == bset,
            (Self::Opaque(a), Self::Opaque(b)) => a.extension_id() == b.extension_id(),
            _ => false,
        }
    }
}

impl Eq for RuleConditionWitness {}

/// Canonical neutral boolean formula tree for rule evaluation witnesses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleFormulaWitness {
    /// Leaf condition.
    Condition(RuleConditionWitness),
    /// Logical conjunction.
    And(Box<RuleFormulaWitness>, Box<RuleFormulaWitness>),
    /// Logical disjunction.
    Or(Box<RuleFormulaWitness>, Box<RuleFormulaWitness>),
    /// Logical negation.
    Not(Box<RuleFormulaWitness>),
}

/// Evaluate a single [`RuleConditionWitness`] against `ctx`.
#[must_use]
pub fn evaluate_condition_witness<C: RuleEvaluationContextWitness + ?Sized>(
    condition: &RuleConditionWitness,
    ctx: &C,
) -> bool {
    match condition {
        RuleConditionWitness::PatternExists { pattern_id } => ctx.pattern_count(*pattern_id) > 0,
        RuleConditionWitness::PatternCountGt {
            pattern_id,
            threshold,
        } => ctx.pattern_count(*pattern_id) > *threshold,
        RuleConditionWitness::PatternCountGte {
            pattern_id,
            threshold,
        } => ctx.pattern_count(*pattern_id) >= *threshold,
        RuleConditionWitness::FileSizeLt(t) => ctx.file_size() < *t,
        RuleConditionWitness::FileSizeLte(t) => ctx.file_size() <= *t,
        RuleConditionWitness::FileSizeGt(t) => ctx.file_size() > *t,
        RuleConditionWitness::FileSizeGte(t) => ctx.file_size() >= *t,
        RuleConditionWitness::FileSizeEq(t) => ctx.file_size() == *t,
        RuleConditionWitness::FileSizeNe(t) => ctx.file_size() != *t,
        RuleConditionWitness::LiteralTrue => true,
        RuleConditionWitness::LiteralFalse => false,
        RuleConditionWitness::RegexMatch { field, pattern } => {
            let Some(value) = ctx.field_value(field.as_ref()) else {
                return false;
            };
            use std::collections::HashMap;
            use std::sync::LazyLock;
            use std::sync::Mutex;
            static REGEX_CACHE: LazyLock<Mutex<HashMap<String, regex::Regex>>> =
                LazyLock::new(|| Mutex::new(HashMap::new()));
            let Ok(cache) = REGEX_CACHE.lock() else {
                return false;
            };
            let re = cache.get(pattern.as_ref()).cloned();
            drop(cache);
            match re {
                Some(re) => re.is_match(value),
                None => match regex::Regex::new(pattern.as_ref()) {
                    Ok(re) => {
                        let Ok(mut cache) = REGEX_CACHE.lock() else {
                            return false;
                        };
                        cache.insert(pattern.to_string(), re.clone());
                        re.is_match(value)
                    }
                    Err(_) => false,
                },
            }
        }
        RuleConditionWitness::SubstringMatch { haystack, needle } => ctx
            .field_value(haystack.as_ref())
            .map(|h| h.contains(needle.as_ref()))
            .unwrap_or(false),
        RuleConditionWitness::PrefixMatch { value, prefix } => ctx
            .field_value(value.as_ref())
            .map(|v| v.starts_with(prefix.as_ref()))
            .unwrap_or(false),
        RuleConditionWitness::SuffixMatch { value, suffix } => ctx
            .field_value(value.as_ref())
            .map(|v| v.ends_with(suffix.as_ref()))
            .unwrap_or(false),
        RuleConditionWitness::RangeMatch { value, min, max } => *value >= *min && *value <= *max,
        RuleConditionWitness::SetMembership { value, set } => set
            .iter()
            .map(std::sync::Arc::as_ref)
            .any(|m| m == value.as_ref()),
        RuleConditionWitness::FieldInSet { field, set } => {
            let Some(value) = ctx.field_value(field.as_ref()) else {
                return false;
            };
            set.iter().map(std::sync::Arc::as_ref).any(|m| m == value)
        }
        RuleConditionWitness::Opaque(ext) => ext.evaluate_opaque(&() as &dyn std::any::Any),
    }
}

/// Evaluate a [`RuleFormulaWitness`] tree against `ctx`.
#[must_use]
pub fn evaluate_formula_witness<C: RuleEvaluationContextWitness + ?Sized>(
    formula: &RuleFormulaWitness,
    ctx: &C,
) -> bool {
    match formula {
        RuleFormulaWitness::Condition(cond) => evaluate_condition_witness(cond, ctx),
        RuleFormulaWitness::And(left, right) => {
            evaluate_formula_witness(left, ctx) && evaluate_formula_witness(right, ctx)
        }
        RuleFormulaWitness::Or(left, right) => {
            evaluate_formula_witness(left, ctx) || evaluate_formula_witness(right, ctx)
        }
        RuleFormulaWitness::Not(inner) => !evaluate_formula_witness(inner, ctx),
    }
}
