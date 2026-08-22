use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    Dict(Vec<String>),
    Tuple(usize),
    List(usize),
    Leaf,
}

#[derive(Debug, Clone)]
pub struct TreeDef {
    pub kind: NodeKind,
    pub children: Vec<Arc<TreeDef>>,
    pub num_leaves: usize,
}

impl TreeDef {
    pub fn leaf() -> Arc<Self> {
        Arc::new(TreeDef {
            kind: NodeKind::Leaf,
            children: Vec::new(),
            num_leaves: 1,
        })
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self.kind, NodeKind::Leaf)
    }

    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }
}
