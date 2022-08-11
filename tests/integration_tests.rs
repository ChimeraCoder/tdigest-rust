use tdigest;
use rand::{thread_rng, Rng};

#[test]
fn test_histogram_generation() {
    const COMPRESSION: f64 = 10.0;
    let td = tdigest::new_merging(COMPRESSION, false);
    add_samples(td)

}


fn add_samples(mut td: tdigest::MergingDigest){
    let mut rng = thread_rng();
    let m: f64 = rng.gen_range(0.0, 1.0);
    td.add(m, 1.0);
}
