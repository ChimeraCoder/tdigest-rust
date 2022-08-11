use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tdigest;
use rand::distributions::{Distribution, Uniform};

fn criterion_benchmark(c: &mut Criterion){
    const COMPRESSION: f64 = 100.0;

    let mut td = tdigest::new_merging(COMPRESSION, false);
    let mut rng = rand::thread_rng();
    let between = Uniform::from(0.0..1.0);

    c.bench_function("tdigest add", |b| b.iter(|| {
        td.add(between.sample(&mut rng), 1.0);
    }));
}


criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
