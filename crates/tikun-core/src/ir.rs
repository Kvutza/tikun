use serde::{Deserialize, Serialize};

/// Hardware Workload Roles (CAKE Co-Design Architecture)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadRole {
    /// Load memory from VRAM/RAM into L1/L2 cache or registers
    Load,
    /// Vectorized SIMD / Tensor Core Compute FMA operations
    Compute,
    /// Store updated parameter buffers back to memory
    Store,
}

/// CAKE-inspired typed Schedule IR node for explicit hardware execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleIRNode {
    pub name: String,
    pub role: WorkloadRole,
    pub tile_size: usize,
    pub unroll_factor: usize,
    pub vector_width: usize,
    pub prefetch_distance: usize,
}

impl ScheduleIRNode {
    pub fn new(name: &str, role: WorkloadRole, vector_width: usize) -> Self {
        Self {
            name: name.to_string(),
            role,
            tile_size: 64, // Default 64-byte L1 cache line alignment
            unroll_factor: 4,
            vector_width,
            prefetch_distance: 2,
        }
    }
}

/// Two-tier Algebraic Representation: Tier 1 Global Monoidal Reductions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GlobalReduction {
    /// Global L2 Norm across all leaf tensors: (Σ ||g_i||²)^(1/2)
    GlobalL2Norm { max_norm: f32 },
}

/// Two-tier Algebraic Representation: Tier 2 Pointwise Kernel Specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PrimitiveOp {
    /// Temporal Memory Accumulator (EMA 1st/2nd moments, Nesterov)
    Accumulate { beta: f32, power: u32, nesterov: bool },
    /// Manifold/Matrix Metric Preconditioning (Polar Newton-Schulz, Diagonal RMS, Shampoo)
    Precondition { mode: PreconditionMode, steps: usize, eps: f32 },
    /// Coordinate-Wise Non-Linear Projection (Sign, Clamp, RMS scale)
    Project { mode: ProjectionMode, threshold: f32 },
    /// Manifold Retraction & Weight Update
    Retract { learn_rate: f32, weight_decay: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreconditionMode {
    CoordinateRsqrt,
    PolarNewtonSchulz,
    KroneckerShampoo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionMode {
    Identity,
    Sign,
    ClipByValue,
    RmsNorm,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PointwiseOp {
    AdamW {
        step_count: usize,
        learn_rate: f32,
        beta_one: f32,
        beta_two: f32,
        eps: f32,
        decay: f32,
    },
    Lion {
        learn_rate: f32,
        beta_one: f32,
        beta_two: f32,
        decay: f32,
    },
    SGD {
        learn_rate: f32,
        momentum: f32,
        decay: f32,
    },
    CustomDAG {
        primitives: Vec<PrimitiveOp>,
    },
}

/// A linear sequence of typed hardware schedule operations representing a pipeline
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulePipeline {
    pub nodes: Vec<ScheduleIRNode>,
    pub reduction: Option<GlobalReduction>,
}

impl SchedulePipeline {
    pub fn add_node(&mut self, node: ScheduleIRNode) {
        self.nodes.push(node);
    }
}
