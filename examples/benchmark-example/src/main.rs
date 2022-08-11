use tdigest;

use rand::distributions::{Distribution, Uniform};

fn generate_rand() -> f64 {
    let mut rng = rand::thread_rng();
    let between = Uniform::from(0.0..100.0);
    between.sample(&mut rng)
}

fn main(){
    const COMPRESSION: f64 = 1000.0;
    let mut td = tdigest::new_merging(COMPRESSION);

    let mut v = Vec::with_capacity(10000);
    for _ in 0..v.capacity() -1 {
        v.push(generate_rand());
    }

    for v in v.into_iter() {
        td.add(v, 1.0);
    }

    println!("median is {}", td.quantile(0.5));
    println!("99th percentile is {}", td.quantile(0.99));
    println!("min is {}", td.min());
    println!("reciprocal sum is {}", td.reciprocal_sum());
}
