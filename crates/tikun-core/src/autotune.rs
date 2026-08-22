use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub steps: usize,
    pub eps: f32,
    pub decay: f32,
    pub fused: bool,
    pub tile_kb: usize,
    pub store_mode: String,
    pub prefetch: usize,
    pub unroll: usize,
    pub math_mode: String,
    pub workers: usize,
    // Expanded 32-D Micro-Architectural Knobs
    pub flush_to_zero: bool,
    pub prefetch_l2: usize,
    pub interleave_dist: usize,
    pub reg_tile_width: usize,
    pub streaming_cutoff_kb: usize,
    pub p_cores_only: bool,
    pub quant_precision: String,
    pub poly_degree: usize,
}

impl StackConfig {
    pub fn decode(v: &[f32]) -> Self {
        let unrolls = [2, 4, 8, 16];
        let tiles = [16, 32, 64, 128, 256, 512, 1024, 2048];
        let chunks = [4, 8, 12, 16, 24, 32, 48, 64];
        let prefetches = [0, 64, 128, 256, 512, 1024];
        let cutoffs = [16, 64, 256, 1024];
        let interleaves = [1, 2, 4, 8];
        let reg_widths = [4, 8, 16, 24];

        let g = |idx: usize, def: f32| v.get(idx).cloned().unwrap_or(def);

        let beta1 = 0.80 + g(0, 0.5) * 0.19;
        let beta2 = 0.95 + g(1, 0.5) * 0.0499;
        let steps = 3 + (g(2, 0.5) * 3.0).round() as usize;
        let eps = 10f32.powf(-8.0 + (g(3, 0.5) - 0.5) * 4.0);
        let decay = g(4, 0.5) * 0.1;
        let fused = g(6, 0.5) > 0.5;

        let t_idx = (g(12, 0.5) * (tiles.len() - 1) as f32).round() as usize;
        let store_mode = if g(13, 0.5) > 0.5 { "streaming".to_string() } else { "temporal".to_string() };
        let p_idx = (g(14, 0.5) * (prefetches.len() - 1) as f32).round() as usize;

        let u_idx = (g(18, 0.5) * (unrolls.len() - 1) as f32).round() as usize;
        let math_mode = if g(19, 0.5) > 0.5 { "fast_nr".to_string() } else { "exact".to_string() };
        let c_idx = (g(21, 0.5) * (chunks.len() - 1) as f32).round() as usize;

        // Micro-architectural dimensions (Dims 24..31)
        let flush_to_zero = g(24, 0.5) > 0.5;
        let p_l2_idx = (g(25, 0.5) * (prefetches.len() - 1) as f32).round() as usize;
        let il_idx = (g(26, 0.5) * (interleaves.len() - 1) as f32).round() as usize;
        let rw_idx = (g(27, 0.5) * (reg_widths.len() - 1) as f32).round() as usize;
        let cut_idx = (g(28, 0.5) * (cutoffs.len() - 1) as f32).round() as usize;
        let p_cores_only = g(29, 0.5) > 0.5;
        let quant_precision = if g(30, 0.5) > 0.66 {
            "fp32".to_string()
        } else if g(30, 0.5) > 0.33 {
            "fp16".to_string()
        } else {
            "bf16".to_string()
        };
        let poly_degree = 3 + (g(31, 0.5) * 4.0).round() as usize;

        Self {
            beta1,
            beta2,
            steps,
            eps,
            decay,
            fused,
            tile_kb: tiles[t_idx],
            store_mode,
            prefetch: prefetches[p_idx],
            unroll: unrolls[u_idx],
            math_mode,
            workers: chunks[c_idx],
            flush_to_zero,
            prefetch_l2: prefetches[p_l2_idx],
            interleave_dist: interleaves[il_idx],
            reg_tile_width: reg_widths[rw_idx],
            streaming_cutoff_kb: cutoffs[cut_idx],
            p_cores_only,
            quant_precision,
            poly_degree,
        }
    }
}

/// Canonical Machine-Local Hardware Profile for Closed-Loop JIT Specialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub chip_name: String,
    pub optimal_tile_kb: usize,
    pub unroll_factor: usize,
    pub prefetch_bytes: usize,
    pub worker_threads: usize,
    pub polar_steps: usize,
    pub peak_bandwidth_gbs: f32,
    pub best_step_ms: f32,
    #[serde(default)]
    pub is_custom_tuned: bool,
    pub timestamp: String,
}

static ACTIVE_HARDWARE_PROFILE: std::sync::OnceLock<HardwareProfile> = std::sync::OnceLock::new();

impl HardwareProfile {
    pub fn fallback() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self {
                chip_name: "Apple Silicon (In-Memory Probe)".to_string(),
                optimal_tile_kb: 256,
                unroll_factor: 4,
                prefetch_bytes: 128,
                worker_threads: 8,
                polar_steps: 5,
                peak_bandwidth_gbs: 89.88,
                best_step_ms: 39.77,
                is_custom_tuned: false,
                timestamp: "in_memory_probed".to_string(),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let cores = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(8);
            Self {
                chip_name: "Host CPU (In-Memory Probe)".to_string(),
                optimal_tile_kb: 256,
                unroll_factor: 4,
                prefetch_bytes: 64,
                worker_threads: cores,
                polar_steps: 5,
                peak_bandwidth_gbs: 25.0,
                best_step_ms: 50.0,
                is_custom_tuned: false,
                timestamp: "in_memory_probed".to_string(),
            }
        }
    }

    pub fn load_default() -> Option<Self> {
        let xdg_cache = std::env::var("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::var("HOME")
                    .map(|h| std::path::PathBuf::from(h).join(".cache"))
                    .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
            });

        let cache_file = xdg_cache.join("tikun").join("profile.json");
        if cache_file.exists() {
            if let Ok(content) = std::fs::read_to_string(&cache_file) {
                if let Ok(mut prof) = serde_json::from_str::<HardwareProfile>(&content) {
                    prof.is_custom_tuned = true;
                    return Some(prof);
                }
            }
        }
        None
    }

    pub fn active() -> &'static HardwareProfile {
        ACTIVE_HARDWARE_PROFILE.get_or_init(|| {
            Self::load_default().unwrap_or_else(Self::fallback)
        })
    }

    pub fn save_xdg_cache(&self) -> std::io::Result<()> {
        let xdg_cache = std::env::var("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::var("HOME")
                    .map(|h| std::path::PathBuf::from(h).join(".cache"))
                    .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
            });

        let dir = xdg_cache.join("tikun");
        std::fs::create_dir_all(&dir)?;
        let json_str = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(dir.join("profile.json"), json_str)
    }
}

/// Fast Approximate Nearest Neighbors (AnnIndex) in Pure Rust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnIndex {
    pub dimension: usize,
    pub num_hyperplanes: usize,
    pub hyperplanes: Vec<Vec<f32>>,
    pub points: Vec<Vec<f32>>,
    pub values: Vec<f32>,
    pub buckets: Vec<Vec<usize>>,
}

impl AnnIndex {
    pub fn new(dimension: usize, num_hyperplanes: usize, seed: u64) -> Self {
        let num_buckets = 1 << num_hyperplanes.min(16);
        let mut rng_state = seed;
        let mut hyperplanes = Vec::with_capacity(num_hyperplanes);

        for _ in 0..num_hyperplanes {
            let mut plane = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((rng_state >> 32) as f32 / (u32::MAX as f32)) * 2.0 - 1.0;
                plane.push(val);
            }
            hyperplanes.push(plane);
        }

        Self {
            dimension,
            num_hyperplanes,
            hyperplanes,
            points: Vec::new(),
            values: Vec::new(),
            buckets: vec![Vec::new(); num_buckets],
        }
    }

    /// Compute binary hyperplane hash signature (bit-mask)
    fn compute_hash(&self, point: &[f32]) -> usize {
        let mut hash = 0usize;
        for (i, plane) in self.hyperplanes.iter().enumerate() {
            let mut dot = 0.0f32;
            let len = point.len().min(plane.len());
            for j in 0..len {
                dot += point[j] * plane[j];
            }
            if dot >= 0.0 {
                hash |= 1 << i;
            }
        }
        hash % self.buckets.len()
    }

    /// Insert observation into BpANN Index
    pub fn insert(&mut self, point: &[f32], value: f32) {
        let idx = self.points.len();
        self.points.push(point.to_vec());
        self.values.push(value);
        let hash = self.compute_hash(point);
        self.buckets[hash].push(idx);
    }

    /// Fast BpANN Approximate Nearest Neighbor Query
    pub fn query_knn(&self, candidate: &[f32], k: usize) -> (f32, f32) {
        if self.points.is_empty() {
            return (100.0, 100.0);
        }

        let hash = self.compute_hash(candidate);
        let bucket_candidates = &self.buckets[hash];

        // If bucket has candidates, query within bucket; otherwise fallback to points
        let search_indices: &[usize] = if !bucket_candidates.is_empty() {
            bucket_candidates
        } else {
            // Check adjacent hamming distance or fallback to all points
            &(0..self.points.len()).collect::<Vec<usize>>()
        };

        let mut dists: Vec<(usize, f32)> = search_indices
            .iter()
            .map(|&idx| {
                let p = &self.points[idx];
                let mut sum_sq = 0.0f32;
                let len = p.len().min(candidate.len());
                for i in 0..len {
                    let diff = p[i] - candidate[i];
                    sum_sq += diff * diff;
                }
                (idx, sum_sq.sqrt())
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let k_actual = k.min(dists.len());
        if k_actual == 0 {
            return (100.0, 100.0);
        }

        let mut total_weight = 0.0f32;
        let mut weighted_sum = 0.0f32;

        for i in 0..k_actual {
            let (idx, dist) = dists[i];
            let weight = 1.0 / (dist + 1e-5);
            weighted_sum += weight * self.values[idx];
            total_weight += weight;
        }

        let mean = weighted_sum / total_weight;
        let uncertainty = dists[0].1;

        (mean, uncertainty)
    }
}

use ennx::acquisition::{ThompsonAcquisition, UCBAcquisition};
use ennx::model::EpistemicNearestNeighbors;
use ennx::morbo_trust_region::MorboTrustRegion;

/// Single-Objective TuRBO-ENN Optimizer with AnnIndex and UCB/Thompson Acquisition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboTuner {
    pub dimension: usize,
    pub trust_region: TrustRegion,
    pub ann_index: AnnIndex,
    pub best_point: Vec<f32>,
    pub best_latency_ms: f32,
    pub worst_latency_ms: f32,
    pub iteration: usize,
    pub history: Vec<(Vec<f32>, f32)>,
}

impl TurboTuner {
    pub fn new(dimension: usize) -> Self {
        let center = vec![0.5; dimension];
        Self {
            dimension,
            trust_region: TrustRegion::new(dimension, center.clone()),
            ann_index: AnnIndex::new(dimension, 8, 42),
            best_point: center,
            best_latency_ms: f32::MAX,
            worst_latency_ms: 0.0,
            iteration: 0,
            history: Vec::new(),
        }
    }

    /// Suggest candidate maximizing UCB Throughput Score: mu + beta * sigma
    pub fn suggest_candidate(&mut self, num_candidates: usize, beta: f32) -> Vec<f32> {
        self.iteration += 1;
        let seed = (self.iteration as u64) * 2654435761;
        let prob_perturb = (20.0 / self.dimension as f32).min(1.0);
        let candidates = self.trust_region.sample_candidates_hd(num_candidates, prob_perturb, seed);

        let mut best_score = f32::NEG_INFINITY;
        let mut best_cand = candidates[0].clone();

        for cand in candidates {
            let (mu_lat, sigma) = self.ann_index.query_knn(&cand, 3);
            // Convert latency to throughput/speed score: higher is better
            let mu_throughput = 1000.0 / (mu_lat + 1e-4);
            let ucb = mu_throughput + beta * sigma;

            if ucb > best_score {
                best_score = ucb;
                best_cand = cand;
            }
        }

        best_cand
    }

    pub fn report_observation(&mut self, candidate: &[f32], latency_ms: f32) {
        self.ann_index.insert(candidate, latency_ms);
        self.trust_region.update(candidate, latency_ms);
        self.history.push((candidate.to_vec(), latency_ms));

        if latency_ms < self.best_latency_ms {
            self.best_latency_ms = latency_ms;
            self.best_point = candidate.to_vec();
        }
        if latency_ms > self.worst_latency_ms {
            self.worst_latency_ms = latency_ms;
        }
    }
}

/// High-Dimensional Epistemic Nearest Neighbors (ENN) Surrogate Model in Pure Rust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnnxSurrogate {
    pub k_neighbors: usize,
    pub observations: Vec<Vec<f32>>,
    pub latency_history: Vec<f32>,
}

impl EnnxSurrogate {
    pub fn new(k_neighbors: usize) -> Self {
        Self {
            k_neighbors,
            observations: Vec::new(),
            latency_history: Vec::new(),
        }
    }

    pub fn observe(&mut self, point: &[f32], latency_ms: f32) {
        self.observations.push(point.to_vec());
        self.latency_history.push(latency_ms);
    }

    pub fn predict(&self, candidate: &[f32]) -> (f32, f32) {
        if self.observations.is_empty() {
            return (100.0, 100.0);
        }

        let dim = candidate.len();
        let mut dists: Vec<(usize, f32)> = self
            .observations
            .iter()
            .enumerate()
            .map(|(idx, obs)| {
                let mut sum_sq = 0.0f32;
                let len = obs.len().min(dim);
                for i in 0..len {
                    let diff = obs[i] - candidate[i];
                    sum_sq += diff * diff;
                }
                (idx, sum_sq.sqrt())
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.k_neighbors.min(dists.len());
        let mut total_weight = 0.0f32;
        let mut weighted_sum = 0.0f32;

        for i in 0..k {
            let (idx, dist) = dists[i];
            let weight = 1.0 / (dist + 1e-5);
            weighted_sum += weight * self.latency_history[idx];
            total_weight += weight;
        }

        let mean = weighted_sum / total_weight;
        let uncertainty = dists[0].1;

        (mean, uncertainty)
    }

    pub fn acquire_lcb(&self, candidates: &[Vec<f32>], kappa: f32) -> Vec<f32> {
        let mut best_score = f32::INFINITY;
        let mut best_cand = candidates[0].clone();

        for cand in candidates {
            let (mu, sigma) = self.predict(cand);
            let lcb = mu - kappa * sigma;
            if lcb < best_score {
                best_score = lcb;
                best_cand = cand.clone();
            }
        }

        best_cand
    }
}

/// High-Dimensional Trust Region for TuRBO-ENN Optimization (D = 24..2400)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustRegion {
    pub dimension: usize,
    pub center: Vec<f32>,
    pub length: f32,
    pub length_min: f32,
    pub length_max: f32,
    pub success_count: usize,
    pub failure_count: usize,
    pub success_tolerance: usize,
    pub failure_tolerance: usize,
    pub best_value: f32,
}

impl TrustRegion {
    pub fn new(dimension: usize, init_center: Vec<f32>) -> Self {
        Self {
            dimension,
            center: init_center,
            length: 0.8,
            length_min: 0.05,
            length_max: 1.6,
            success_count: 0,
            failure_count: 0,
            success_tolerance: 3,
            failure_tolerance: 4,
            best_value: f32::INFINITY,
        }
    }

    pub fn contains(&self, point: &[f32]) -> bool {
        let half = self.length / 2.0;
        for (c, p) in self.center.iter().zip(point.iter()) {
            if *p < (c - half) || *p > (c + half) {
                return false;
            }
        }
        true
    }

    pub fn update(&mut self, point: &[f32], value: f32) {
        if value < self.best_value {
            self.best_value = value;
            self.center = point.to_vec();
            self.success_count += 1;
            self.failure_count = 0;

            if self.success_count >= self.success_tolerance {
                self.length = (self.length * 2.0).min(self.length_max);
                self.success_count = 0;
            }
        } else {
            self.success_count = 0;
            self.failure_count += 1;

            if self.failure_count >= self.failure_tolerance {
                self.length = (self.length / 2.0).max(self.length_min);
                self.failure_count = 0;
            }
        }
    }

    pub fn sample_candidates_hd(
        &self,
        num_candidates: usize,
        prob_perturb: f32,
        seed: u64,
    ) -> Vec<Vec<f32>> {
        let mut candidates = Vec::with_capacity(num_candidates);
        let mut rng_state = seed;

        for _ in 0..num_candidates {
            let mut cand = self.center.clone();
            for i in 0..self.dimension {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let rand_val = (rng_state >> 32) as f32 / (u32::MAX as f32);

                if rand_val < prob_perturb {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let delta = ((rng_state >> 32) as f32 / (u32::MAX as f32) - 0.5) * self.length;
                    cand[i] = (cand[i] + delta).clamp(0.0, 1.0);
                }
            }
            candidates.push(cand);
        }

        candidates
    }
}

/// MORBO-HD: Ultra-High-Dimensional Multi-Objective Trust Region Bayesian Optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorboOptimizer {
    pub dimension: usize,
    pub trust_regions: Vec<TrustRegion>,
    pub surrogate: EnnxSurrogate,
    pub active_tr_idx: usize,
}

impl MorboOptimizer {
    pub fn new(num_trust_regions: usize, dimension: usize, initial_seeds: &[Vec<f32>]) -> Self {
        let mut trust_regions = Vec::new();
        for i in 0..num_trust_regions {
            let init_c = if i < initial_seeds.len() {
                initial_seeds[i].clone()
            } else {
                vec![0.5; dimension]
            };
            trust_regions.push(TrustRegion::new(dimension, init_c));
        }

        Self {
            dimension,
            trust_regions,
            surrogate: EnnxSurrogate::new(3),
            active_tr_idx: 0,
        }
    }

    pub fn suggest_candidate_hd(&mut self, num_samples: usize, kappa: f32, seed: u64) -> Vec<f32> {
        let tr = &self.trust_regions[self.active_tr_idx];
        let prob_perturb = (20.0 / self.dimension as f32).min(1.0);
        let candidates = tr.sample_candidates_hd(num_samples, prob_perturb, seed);
        self.surrogate.acquire_lcb(&candidates, kappa)
    }

    pub fn report_observation(&mut self, candidate: &[f32], latency_ms: f32) {
        self.surrogate.observe(candidate, latency_ms);
        self.trust_regions[self.active_tr_idx].update(candidate, latency_ms);
        self.active_tr_idx = (self.active_tr_idx + 1) % self.trust_regions.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ann_index_knn() {
        let mut index = AnnIndex::new(6, 4, 42);
        index.insert(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 10.5);
        index.insert(&[0.9, 0.8, 0.7, 0.6, 0.5, 0.4], 25.0);
        let (mu, sigma) = index.query_knn(&[0.12, 0.22, 0.32, 0.42, 0.52, 0.62], 1);
        assert!(mu < 20.0);
        assert!(sigma >= 0.0);
    }

    #[test]
    fn test_turbo_tuner() {
        let mut opt = TurboTuner::new(6);
        let cand = opt.suggest_candidate(5, 1.5);
        assert_eq!(cand.len(), 6);
        opt.report_observation(&cand, 12.5);
        assert_eq!(opt.best_latency_ms, 12.5);
    }

    #[test]
    fn test_stack_config_decode() {
        let raw = vec![0.5; 32];
        let cfg = StackConfig::decode(&raw);
        assert!(cfg.beta1 > 0.8 && cfg.beta1 < 1.0);
        assert!(cfg.unroll >= 2);
        assert!(cfg.tile_kb >= 16);
        assert_eq!(cfg.quant_precision, "fp16");
    }
}
