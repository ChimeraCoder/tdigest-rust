use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fmt;

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    #[test]
    fn merging_digest() {
        const COMPRESSION: f64 = 100.0;
        const SAMPLE_SIZE: i64 = 1000;

        let mut td = super::new_merging(COMPRESSION, false);

        td.debug = true;

        // TODO increase sample size
        for _i in 0..SAMPLE_SIZE {
            let mut rng = thread_rng();
            let m: f64 = rng.gen_range(0.0, 1.0);
            td.add(m, 1.0);
        }

        validate_merging_digest(&mut td);

        const EPSILON: f64 = 0.05;

        let median = td.quantile(0.5);
        println!("median is {}", median);
        assert!(
            (median - 0.5).abs() < EPSILON,
            "median was {:.5} (expected {:.5})",
            median,
            0.5
        );

        assert!(
            td.min >= 0.0,
            "minimum was {:.5} (expected non-negative)",
            td.min
        );
        assert!(td.max < 1.0, "maximum was {:.5} (expected below 1)", td.max);
        //assert!(td.sum() > 0.0, "sum was {:.5} (expected greater than 0)", td.sum());
        assert!(
            td.reciprocal_sum > 0.0,
            "reciprocal sum was {:.5} (expected greater than 0)",
            td.reciprocal_sum
        );
    }

    fn validate_merging_digest(td: &mut super::MergingDigest) {
        td.merge_all_temps();

        let mut index = 0.0;
        let mut quantile = 0.0;
        let end = td.main_centroids.len() - 1;

        for (i, centroid) in td.main_centroids.iter().enumerate() {
            let next_index = td.index_estimate(quantile + centroid.weight / td.main_weight);
            // Skip the first and the last centroid
            if i > 0 && i < end {
                assert!(next_index - index <= 1.0 || centroid.weight == 1.0, "\n\n\ncentroid {} of {} is oversized: {}\n\nnext_index: {}, index:{}, centroid_weight: {}\n\n\n", i, td.main_centroids.len(), centroid, next_index, index, centroid.weight);
            }

            quantile += centroid.weight / td.main_weight;
            index = next_index;
        }
    }
}

#[derive(Clone)]
pub struct Centroid {
    mean: f64,
    weight: f64,
    samples: Vec<f64>,
}

impl fmt::Display for Centroid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Centroid(mean: {:.5}, weight: {:.5}, samples: <{} samples> {:?})",
            self.mean,
            self.weight,
            self.samples.len(),
            self.samples
        )
    }
}

pub struct MergingDigest {
    compression: f64,

    main_centroids: Vec<Centroid>,
    main_weight: f64,

    temp_centroids: Vec<Centroid>,
    temp_weight: f64,

    min: f64,
    max: f64,
    reciprocal_sum: f64,

    debug: bool,
}

impl MergingDigest {
    pub fn add(&mut self, value: f64, weight: f64) {
        if value.is_nan() || value.is_infinite() || weight <= 0.0 {
            panic!("invalid value added")
        }

        if self.temp_centroids.len() == self.temp_centroids.capacity() {
            self.merge_all_temps();
        }

        self.min = min(self.min, value);
        self.max = max(self.max, value);
        self.reciprocal_sum += (1.0 / value) * weight;

        let mut samples = vec![];
        if self.debug {
            samples = vec![value];
        }

        let next = Centroid {
            mean: value,
            weight,
            samples,
        };

        self.temp_centroids.push(next);
        self.temp_weight += weight;
    }

    // in-place merge into main_centroids
    fn merge_all_temps(&mut self) {
        if self.temp_centroids.is_empty() {
            return;
        }

        self.temp_centroids.sort_by(|a, b| {
            a.mean
                .partial_cmp(&b.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut tmp_index = 0;

        // This will be the total weight of the t-digest after everything i merged
        let total_weight = self.main_weight + self.temp_weight;

        // running tally of how much weight has been merged so far
        let mut merged_weight = 0.0;

        // the index of the most recent quantile to be merged into the previous centroid
        // this value is updated every time we split a new centroid out instead of merging
        // into the current centroid
        let mut last_merged_index = 0.0;

        // since we are merging in-place into the main centroids, we need to keep track of
        // the indices of the remaining elements
        let mut actual_main_centroids = self.main_centroids.clone();
        self.main_centroids = Vec::with_capacity(actual_main_centroids.capacity());

        let mut swapped_centroids: Vec<Centroid> = Vec::new();

        while actual_main_centroids.len() + swapped_centroids.len() != 0
            || tmp_index < self.temp_centroids.len()
        {
            let mut next_temp = Centroid {
                mean: std::f64::INFINITY,
                weight: 0.0,
                samples: vec![],
            };

            if tmp_index < self.temp_centroids.len() {
                next_temp = self.temp_centroids[tmp_index].clone();
            }

            let mut next_main = Centroid {
                mean: std::f64::INFINITY,
                weight: 0.0,
                samples: vec![],
            };

            if !swapped_centroids.is_empty() {
                next_main = swapped_centroids[0].clone()
            } else if !actual_main_centroids.is_empty() {
                next_main = actual_main_centroids[0].clone()
            }

            if next_main.mean < next_temp.mean {
                if !actual_main_centroids.is_empty() {
                    if !swapped_centroids.is_empty() {
                        // if this came from swap, before merging, we have to save
                        // the next main centroid at the end

                        // shift, then push
                        swapped_centroids = swapped_centroids[1..].to_vec();
                        swapped_centroids.push(actual_main_centroids[0].clone());
                    }
                    actual_main_centroids = actual_main_centroids[1..].to_vec();
                } else {
                    swapped_centroids = swapped_centroids[1..].to_vec();
                }


                last_merged_index =
                    self.merge_one(merged_weight, total_weight, last_merged_index, &next_main);
                merged_weight += next_main.weight;
            } else {
                if !actual_main_centroids.is_empty() {
                    swapped_centroids.push(actual_main_centroids[0].clone());
                    actual_main_centroids = actual_main_centroids[1..].to_vec();
                }
                tmp_index += 1;

                last_merged_index =
                    self.merge_one(merged_weight, total_weight, last_merged_index, &next_temp);
                merged_weight += next_temp.weight;
            }
        }

        self.temp_centroids = Vec::with_capacity(self.temp_centroids.capacity());
        self.temp_weight = 0.0;
        self.main_weight = total_weight;
    }

    pub fn merge_one(
        &mut self,
        before_weight: f64,
        total_weight: f64,
        before_index: f64,
        next: &Centroid,
    ) -> f64 {
        let next_index = self.index_estimate((before_weight + next.weight) / total_weight);

        if (next_index - before_index) > 1.0 || self.main_centroids.is_empty() {
            // the new index is far away from the last index of the current centroid
            // thereofre we cannot merge into the current centroid
            // or it would become to wide
            // so we will append a new centroid
            self.main_centroids.push(next.clone());

            // return the last index that was merged into the previous centroid
            self.index_estimate(before_weight / total_weight)
        } else {
            let main_centroids_len = self.main_centroids.len();

            // the new index fits into the range of the current centroid
            // so we combine it into the current centroid's values
            // this computation is known as Welford's method, and the order matters
            // weight must be updated before mean
            self.main_centroids[main_centroids_len - 1].weight += next.weight;
            self.main_centroids[main_centroids_len - 1].mean +=
                (next.mean - self.main_centroids[main_centroids_len - 1].mean) * next.weight
                    / self.main_centroids[main_centroids_len - 1].weight;
            if self.debug {
                if let Some(centroid) = self.main_centroids.last_mut() {
                    centroid.samples.extend_from_slice(&next.samples);
                }

            }

            // we did not create a new centroid, so the trailing index of the previous centroid
            // remains

            before_index
        }
    }

    // Approximate the index of the centroid that would contain a
    // particular value, at the configured compression level
    pub fn index_estimate(&self, quantile: f64) -> f64 {
        let asin = ((2.0 * quantile - 1.0) as f64).asin();
        let scalar = (asin / std::f64::consts::PI) + 0.5;
        self.compression * scalar
    }

    // Return the approximate percentage of values in the digest that are below
    // the specified value (ie, the approximate cumulative distribution function).
    // Return NaN if the digest is empty.
    pub fn cdf(&mut self, value: f64) -> f64 {
        self.merge_all_temps();

        if self.main_centroids.is_empty() {
            return std::f64::NAN;
        }

        if value <= self.min {
            return 0.0;
        }

        if value >= self.max {
            return 1.0;
        }

        let mut weight_so_far = 0.0;
        let mut lower_bound = self.min;

        for (i, c) in self.main_centroids.clone().into_iter().enumerate() {
            let upper_bound = self.centroid_upper_bound(i);
            if value < upper_bound {
                // the value falls inside the bounds of this centroid
                // based on the assumed uniform distribution, we calculate how much
                // of this centroid's weight is below the value

                weight_so_far += c.weight * (value - lower_bound) / (upper_bound - lower_bound);
                return weight_so_far / (self.main_weight as f64);
            }

            // the value is above this centroid, so add the weight and move on
            weight_so_far += c.weight;
            lower_bound = upper_bound;
        }

        // unreachable, since the final loop compares value < self.max
        //std::f64::NAN
        unreachable!("final loop compares value < self.max"); // does it?
    }

    fn centroid_upper_bound(&self, i: usize) -> f64 {
        if i != self.main_centroids.len() - 1 {
            (self.main_centroids[i + 1].mean + self.main_centroids[i].mean) / 2.0
        } else {
            self.max
        }
    }

    pub fn quantile(&mut self, quantile: f64) -> f64 {
        assert!((0.0..=1.0).contains(&quantile));

        self.merge_all_temps();

        let q = quantile * self.main_weight;
        let mut weight_so_far = 0.0;
        let mut lower_bound = self.min;

        for i in 0..self.main_centroids.len() {
            let c = &self.main_centroids[i];
            let upper_bound = self.centroid_upper_bound(i);
            if q < weight_so_far + c.weight {
                // the target quantile is inside this centroid
                // compute how much of this centroid's weight falls into the quantile
                let proportion = (q - weight_so_far) / c.weight;

                // and then interpolate what value that corresponds to inside a uniform
                // distribution

                return lower_bound + (proportion * (upper_bound - lower_bound));
            }

            // the quantile is above this centroid, so sum the weight and carry on
            weight_so_far += c.weight;
            lower_bound = upper_bound;
        }

        // should never be reached unless empty, since the final comparison is
        // q <= td.main_weight
        std::f64::NAN
    }

    // Merge another digest into this one
    pub fn merge(&mut self, other: MergingDigest) {
        let old_reciprocal_sum = self.reciprocal_sum;

        // centroids are pre-sorted, which is bad for merging
        // solution: shuffle them and then merge in a random order
        // see also: quicksort

        let mut shuffled_indices: Vec<usize> = (0..other.main_centroids.len()).collect();
        let mut rng = thread_rng();
        shuffled_indices.shuffle(&mut rng);

        for &index in shuffled_indices.iter() {
            // TODO why is this necessary?
            // why does the collection create a pointer that needs to be explicitly
            // cast/dereferenced?
            let i = index as usize;

            self.add(other.main_centroids[i].mean, other.main_centroids[i].weight);
        }

        // the temp centroids are unsorted so they don't need to be shuffled

        for i in 0..other.temp_centroids.len() - 1 {
            self.add(other.temp_centroids[i].mean, other.temp_centroids[i].weight);
        }

        self.reciprocal_sum = old_reciprocal_sum + other.reciprocal_sum;
    }
}

pub fn new_merging(compression: f64, debug: bool) -> MergingDigest {
    // upper bound on the size of the centroid limit
    // TODO this can actually be reduced even further
    let _size_bound = ((std::f64::consts::PI * compression / 2.0) + 0.5) as i64;

    let tmp_buffer = estimate_temp_buffer(compression) as usize;

    MergingDigest {
        compression,
        main_centroids: vec![],
        main_weight: 0.0,
        temp_centroids: Vec::with_capacity(tmp_buffer),
        temp_weight: 0.0,
        min: std::f64::INFINITY,
        max: std::f64::NEG_INFINITY,
        reciprocal_sum: 0.0,
        debug,
    }
}

pub fn estimate_temp_buffer(compression: f64) -> usize {
    let temp_compression = std::cmp::min(925, std::cmp::max(20, compression as i64)) as f64;

    (7.5 + 0.37 * temp_compression - 2e-4 * temp_compression * temp_compression) as usize
}

fn min(a: f64, b: f64) -> f64 {
    if a < b {
        return a;
    }
    b
}

fn max(a: f64, b: f64) -> f64 {
    if a > b {
        return a;
    }
    b
}
