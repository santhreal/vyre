//! Hoisting of `Let` bindings a split segment references but does not define,
//! so every segment is a self-contained program.

use vyre_foundation::ir::{Expr, Node};

fn collect_global_let_bindings(nodes: &[Node], map: &mut std::collections::HashMap<String, Node>) {
    for node in nodes {
        match node {
            Node::Let { name, .. } => {
                map.insert(name.as_str().to_string(), node.clone());
            }
            Node::If {
                then, otherwise, ..
            } => {
                collect_global_let_bindings(then, map);
                collect_global_let_bindings(otherwise, map);
            }
            Node::Loop { body, .. } | Node::Block(body) => {
                collect_global_let_bindings(body, map);
            }
            Node::Region { body, .. } => {
                collect_global_let_bindings(&body[..], map);
            }
            // Leaf case: the nesting variants above are exactly the ones `transform::visit::child_bodies` lists, so an unknown variant has no child statements to visit.
            _ => {}
        }
    }
}

fn collect_locally_defined_vars(nodes: &[Node], vars: &mut std::collections::HashSet<String>) {
    for node in nodes {
        match node {
            Node::Let { name, .. } => {
                vars.insert(name.as_str().to_string());
            }
            Node::Loop { var, body, .. } => {
                vars.insert(var.as_str().to_string());
                collect_locally_defined_vars(body, vars);
            }
            Node::If {
                then, otherwise, ..
            } => {
                collect_locally_defined_vars(then, vars);
                collect_locally_defined_vars(otherwise, vars);
            }
            Node::Block(body) => {
                collect_locally_defined_vars(body, vars);
            }
            Node::Region { body, .. } => {
                collect_locally_defined_vars(&body[..], vars);
            }
            // Leaf case: the nesting variants above are exactly the ones `transform::visit::child_bodies` lists, so an unknown variant has no child statements to visit.
            _ => {}
        }
    }
}

fn collect_referenced_vars(expr: &Expr, vars: &mut std::collections::HashSet<String>) {
    match expr {
        Expr::Var(name) => {
            vars.insert(name.as_str().to_string());
        }
        Expr::Load { index, .. } => {
            collect_referenced_vars(index, vars);
        }
        Expr::BinOp { left, right, .. } => {
            collect_referenced_vars(left, vars);
            collect_referenced_vars(right, vars);
        }
        Expr::UnOp { operand, .. } => {
            collect_referenced_vars(operand, vars);
        }
        Expr::Call { args, .. } => {
            for arg in args {
                collect_referenced_vars(arg, vars);
            }
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            collect_referenced_vars(cond, vars);
            collect_referenced_vars(true_val, vars);
            collect_referenced_vars(false_val, vars);
        }
        Expr::Cast { value, .. } => {
            collect_referenced_vars(value, vars);
        }
        Expr::Fma { a, b, c } => {
            collect_referenced_vars(a, vars);
            collect_referenced_vars(b, vars);
            collect_referenced_vars(c, vars);
        }
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => {
            collect_referenced_vars(index, vars);
            if let Some(expected) = expected {
                collect_referenced_vars(expected, vars);
            }
            collect_referenced_vars(value, vars);
        }
        Expr::SubgroupBallot { cond } => {
            collect_referenced_vars(cond, vars);
        }
        Expr::SubgroupShuffle { value, lane } => {
            collect_referenced_vars(value, vars);
            collect_referenced_vars(lane, vars);
        }
        Expr::SubgroupReduce { value, .. } => {
            collect_referenced_vars(value, vars);
        }
        _ => {}
    }
}

fn collect_node_referenced_vars(node: &Node, vars: &mut std::collections::HashSet<String>) {
    match node {
        Node::Let { value, .. } => {
            collect_referenced_vars(value, vars);
        }
        Node::Assign { value, .. } => {
            collect_referenced_vars(value, vars);
        }
        Node::Store { index, value, .. } => {
            collect_referenced_vars(index, vars);
            collect_referenced_vars(value, vars);
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            collect_referenced_vars(cond, vars);
            for n in then {
                collect_node_referenced_vars(n, vars);
            }
            for n in otherwise {
                collect_node_referenced_vars(n, vars);
            }
        }
        Node::Loop { from, to, body, .. } => {
            collect_referenced_vars(from, vars);
            collect_referenced_vars(to, vars);
            for n in body {
                collect_node_referenced_vars(n, vars);
            }
        }
        Node::Block(body) => {
            for n in body {
                collect_node_referenced_vars(n, vars);
            }
        }
        Node::Region { body, .. } => {
            for n in body.as_ref() {
                collect_node_referenced_vars(n, vars);
            }
        }
        Node::AsyncLoad { offset, size, .. } => {
            collect_referenced_vars(offset, vars);
            collect_referenced_vars(size, vars);
        }
        Node::AsyncStore { offset, size, .. } => {
            collect_referenced_vars(offset, vars);
            collect_referenced_vars(size, vars);
        }
        Node::Trap { address, .. } => {
            collect_referenced_vars(address, vars);
        }
        // Leaf case: the nesting variants above are exactly the ones `transform::visit::child_bodies` lists, so an unknown variant has no child statements to visit.
        _ => {}
    }
}

fn resolve_dependencies(
    name: &str,
    global_lets: &std::collections::HashMap<String, Node>,
    resolved_names: &mut std::collections::HashSet<String>,
    resolved_lets: &mut Vec<Node>,
) {
    if resolved_names.contains(name) {
        return;
    }
    if let Some(let_node) = global_lets.get(name) {
        resolved_names.insert(name.to_string());
        let mut deps = std::collections::HashSet::new();
        collect_node_referenced_vars(let_node, &mut deps);
        for dep in deps {
            resolve_dependencies(&dep, global_lets, resolved_names, resolved_lets);
        }
        resolved_lets.push(let_node.clone());
    }
}

pub(super) fn propagate_let_bindings(segments: &mut [Vec<Node>], hoisted_inner: &[Node]) {
    let mut global_lets = std::collections::HashMap::new();
    collect_global_let_bindings(hoisted_inner, &mut global_lets);

    for segment_nodes in segments {
        let mut locally_defined = std::collections::HashSet::new();
        collect_locally_defined_vars(segment_nodes, &mut locally_defined);

        let mut referenced = std::collections::HashSet::new();
        for node in segment_nodes.iter() {
            collect_node_referenced_vars(node, &mut referenced);
        }

        let mut free_vars = Vec::new();
        for name in referenced {
            if !locally_defined.contains(&name) {
                free_vars.push(name);
            }
        }

        let mut resolved_lets = Vec::new();
        let mut resolved_names = std::collections::HashSet::new();
        for name in free_vars {
            resolve_dependencies(&name, &global_lets, &mut resolved_names, &mut resolved_lets);
        }

        if !resolved_lets.is_empty() {
            resolved_lets.extend(std::mem::take(segment_nodes));
            *segment_nodes = resolved_lets;
        }
    }
}
