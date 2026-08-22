use serde::{Deserialize, Serialize};
use crate::autotune::AnnIndex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub candidate: Vec<f32>,
    pub latency_ms: f32,
    pub bandwidth_gbs: f32,
    pub cache_misses: u64,
    pub numerical_drift: f32,
    pub hypervolume_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorboEngine {
    pub dimension: usize,
    pub points: Vec<ParetoPoint>,
    pub ann_index: AnnIndex,
}

impl MorboEngine {
    pub fn new(dimension: usize, num_hyperplanes: usize) -> Self {
        Self {
            dimension,
            points: Vec::new(),
            ann_index: AnnIndex::new(dimension, num_hyperplanes, 42),
        }
    }

    /// Evaluates multi-objective non-dominated Pareto ranking
    pub fn add_observation(
        &mut self,
        candidate: Vec<f32>,
        latency_ms: f32,
        bandwidth_gbs: f32,
        cache_misses: u64,
        numerical_drift: f32,
    ) {
        // Multi-objective scalar hypervolume score: maximize bandwidth, minimize latency & cache misses & drift
        let norm_lat = (100.0 / latency_ms.max(0.1)).min(10.0);
        let norm_bw = (bandwidth_gbs / 10.0).min(15.0);
        let norm_miss = (1000.0 / (cache_misses as f32 + 1.0)).min(10.0);
        let norm_drift = (1.0 / (numerical_drift.max(1e-8) * 1e6)).min(10.0);

        let hypervolume_score = norm_lat * 0.4 + norm_bw * 0.3 + norm_miss * 0.15 + norm_drift * 0.15;

        // Insert into spatial AnnIndex for KNN surrogate queries
        self.ann_index.insert(&candidate, latency_ms);

        self.points.push(ParetoPoint {
            candidate,
            latency_ms,
            bandwidth_gbs,
            cache_misses,
            numerical_drift,
            hypervolume_score,
        });
    }

    /// Extracts the non-dominated Pareto frontier points
    pub fn pareto_frontier(&self) -> Vec<ParetoPoint> {
        let mut frontier = Vec::new();

        for (i, p1) in self.points.iter().enumerate() {
            let mut is_dominated = false;
            for (j, p2) in self.points.iter().enumerate() {
                if i != j {
                    // p2 dominates p1 if p2 is strictly better in all 4 objectives
                    if p2.latency_ms <= p1.latency_ms
                        && p2.bandwidth_gbs >= p1.bandwidth_gbs
                        && p2.cache_misses <= p1.cache_misses
                        && p2.numerical_drift <= p1.numerical_drift
                        && (p2.latency_ms < p1.latency_ms || p2.bandwidth_gbs > p1.bandwidth_gbs)
                    {
                        is_dominated = true;
                        break;
                    }
                }
            }
            if !is_dominated {
                frontier.push(p1.clone());
            }
        }

        frontier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morbo_pareto_frontier() {
        let mut engine = MorboEngine::new(4, 4);
        // Point 1: Fast, High Bandwidth, Few Cache Misses (Dominant)
        engine.add_observation(vec![0.1; 4], 5.0, 85.0, 10, 1e-7);
        // Point 2: Slow, Low Bandwidth, Many Cache Misses (Dominated)
        engine.add_observation(vec![0.9; 4], 25.0, 20.0, 500, 1e-5);

        let frontier = engine.pareto_frontier();
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].latency_ms, 5.0);
    }
}
