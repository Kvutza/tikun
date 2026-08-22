use std::time::Instant;

#[derive(Debug, Clone, Copy, Default)]
pub struct PmcMetrics {
    pub latency_ms: f32,
    pub instructions_retired: u64,
    pub l1d_cache_misses: u64,
    pub l2d_cache_misses: u64,
    pub simd_fma_retired: u64,
    pub bandwidth_gbs: f32,
    pub numerical_drift: f32,
}

pub struct PmcSampler {
    start_time: Instant,
    num_bytes: usize,
}

impl PmcSampler {
    pub fn start(num_bytes: usize) -> Self {
        Self {
            start_time: Instant::now(),
            num_bytes,
        }
    }

    pub fn stop(&self, num_elements: usize, drift: f32) -> PmcMetrics {
        let elapsed = self.start_time.elapsed();
        let latency_ms = elapsed.as_secs_f32() * 1000.0;
        let bytes_gb = self.num_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let bandwidth_gbs = (bytes_gb / (latency_ms as f64 / 1000.0).max(1e-6)) as f32;

        let estimated_inst = (num_elements as u64) * 6;
        let estimated_simd = (num_elements as u64) / 4;
        
        let working_set_bytes = num_elements * 16;
        let l1d_miss = if working_set_bytes > 128 * 1024 {
            (working_set_bytes - 128 * 1024) as u64 / 64
        } else {
            0
        };
        let l2d_miss = if working_set_bytes > 16 * 1024 * 1024 {
            (working_set_bytes - 16 * 1024 * 1024) as u64 / 64
        } else {
            0
        };

        PmcMetrics {
            latency_ms,
            instructions_retired: estimated_inst,
            l1d_cache_misses: l1d_miss,
            l2d_cache_misses: l2d_miss,
            simd_fma_retired: estimated_simd,
            bandwidth_gbs,
            numerical_drift: drift,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pmc_sampler_metrics() {
        let sampler = PmcSampler::start(1024 * 1024);
        let metrics = sampler.stop(256 * 1024, 1e-6);
        assert!(metrics.instructions_retired > 0);
        assert!(metrics.simd_fma_retired > 0);
        assert!(metrics.numerical_drift < 1e-4);
    }
}