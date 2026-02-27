use codemem_core::VectorBackend;
use codemem_vector::HnswIndex;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

/// Deterministic pseudo-random vector generator.
/// Uses a simple LCG-like formula to produce varied but reproducible 768-dim vectors.
fn random_vector(dim: usize, seed: usize) -> Vec<f32> {
    let mut val = seed.wrapping_mul(2654435761) ^ 0xDEADBEEF;
    (0..dim)
        .map(|_| {
            val = val.wrapping_mul(1664525).wrapping_add(1013904223);
            // Normalize to [0, 1)
            (val & 0xFFFF) as f32 / 65536.0
        })
        .collect()
}

/// Benchmark inserting N vectors into a fresh HNSW index.
fn bench_hnsw_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");

    for &count in &[100, 1000] {
        // Pre-generate vectors outside the benchmark loop to measure only insert time
        let vectors: Vec<Vec<f32>> = (0..count).map(|i| random_vector(768, i)).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    let mut index = HnswIndex::with_defaults().unwrap();
                    for (i, v) in vectors.iter().enumerate() {
                        index.insert(&format!("mem-{i}"), v).unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark top-10 nearest neighbor search on an index pre-populated with 1000 vectors.
fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");

    // Build the index once, outside the benchmark iteration
    let mut index = HnswIndex::with_defaults().unwrap();
    for i in 0..1000 {
        let v = random_vector(768, i);
        index.insert(&format!("mem-{i}"), &v).unwrap();
    }

    // Use a query vector that was NOT inserted (seed 99999)
    let query = random_vector(768, 99999);

    group.bench_function("top10_in_1000", |b| {
        b.iter(|| {
            let results = index.search(&query, 10).unwrap();
            assert_eq!(results.len(), 10);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_hnsw_insert, bench_hnsw_search);
criterion_main!(benches);
