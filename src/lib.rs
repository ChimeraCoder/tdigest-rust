extern crate rand;

use std::fmt;

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    #[test]
    fn merging_digest() {
        let mut td = super::new_merging(1000.0, false);

        td.debug = true;

        // TODO increase sample size
        for i in 0..10000 {
            let mut rng = thread_rng();
            let mut m: f64 = rng.gen_range(0.0, 1.0);
            td.add(m, 1.0);
        }

        validate_merging_digest(&mut td);
    }

    fn validate_merging_digest(td: &mut super::MergingDigest) {
        td.merge_all_temps();

        let mut index = 0.0;
        let mut quantile = 0.0;
        let mut running_weight = 0.0;

        for i in 0..td.main_centroids.len() - 1 {
            // I would do
            // let c = td.main_centroids[i]
            // but the borrow checker doesn't like that
            // TODO find a way to satisfy borrow checker

            let centroid_weight = td.main_centroids[i].weight;
            let next_index = td.index_estimate(quantile + centroid_weight / td.main_weight);

            // Skip the first and the last centroid
            if i != 0 && i != td.main_centroids.len() - 1 {
                assert!(next_index - index <= 1.0 || centroid_weight == 1.0, "\n\n\ncentroid {} of {} is oversized: {}\n\nnext_index: {}, index:{}, centroid_weight: {}\n\n\n", i, td.main_centroids.len(), td.main_centroids[i], next_index, index, centroid_weight);
            }

            quantile += td.main_centroids[i].weight / td.main_weight;
            index = next_index;
            running_weight += td.main_centroids[i].weight;
        }
    }
}

#[derive(Clone)]
struct Centroid {
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

struct MergingDigest {
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
    fn add(&mut self, value: f64, weight: f64) {
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
            weight: weight,
            samples: samples,
        };

        self.temp_centroids.push(next);
        self.temp_weight += weight;
    }

    // in-place merge into main_centroids
    fn merge_all_temps(&mut self) {
        if self.temp_centroids.len() == 0 {
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

            if swapped_centroids.len() != 0 {
                next_main = swapped_centroids[0].clone()
            } else if actual_main_centroids.len() != 0 {
                next_main = actual_main_centroids[0].clone()
            }

            if next_main.mean < next_temp.mean {
                if actual_main_centroids.len() != 0 {
                    if swapped_centroids.len() != 0 {
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

                // TODO another way to satisfy the borrow checker?
                let next_main_weight = next_main.weight;

                last_merged_index =
                    self.merge_one(merged_weight, total_weight, last_merged_index, next_main);
                merged_weight += next_main_weight;
            } else {
                if actual_main_centroids.len() != 0 {
                    swapped_centroids.push(actual_main_centroids[0].clone());
                    actual_main_centroids = actual_main_centroids[1..].to_vec();
                }
                tmp_index += 1;

                // TODO another way to satisfy the borrow checker?
                let next_temp_weight = next_temp.weight;

                last_merged_index =
                    self.merge_one(merged_weight, total_weight, last_merged_index, next_temp);
                merged_weight += next_temp_weight;
            }
        }

        self.temp_centroids = Vec::with_capacity(self.temp_centroids.capacity());
        self.temp_weight = 0.0;
        self.main_weight = total_weight;
    }

    fn merge_one(
        &mut self,
        before_weight: f64,
        total_weight: f64,
        before_index: f64,
        next: Centroid,
    ) -> f64 {
        let next_index = self.index_estimate((before_weight + next.weight) / total_weight);

        if (next_index - before_index) > 1.0 || (self.main_centroids.len() == 0) {
            // the new index is far away from the last index of the current centroid
            // thereofre we cannot merge into the current centroid
            // or it would become to wide
            // so we will append a new centroid
            self.main_centroids.push(next);

            // return the last index that was merged into the previous centroid
            return self.index_estimate(before_weight / total_weight);
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
                // TODO how to append succinctly?

                for sample in next.samples.into_iter() {
                    self.main_centroids[main_centroids_len - 1]
                        .samples
                        .push(sample);
                }
            }

            // we did not create a new centroid, so the trailing index of the previous centroid
            // remains

            return before_index;
        }
    }

    // Approximate the index of the centroid that would contain a
    // particular value, at the configured compression level
    fn index_estimate(&mut self, quantile: f64) -> f64 {
        let asin = ((2.0 * quantile - 1.0) as f64).asin();
        let scalar = (asin / std::f64::consts::PI) + 0.5;
        return self.compression * scalar;
    }

    // Return the approximate percentage of values in the digest that are below
    // the specified value (ie, the approximate cumulative distribution function).
    // Return NaN if the digest is empty.
    fn cdf(&mut self, value: f64) -> f64 {
        self.merge_all_temps();

        if self.main_centroids.len() == 0 {
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
        return std::f64::NAN;
    }

    fn centroid_upper_bound(&self, i: usize) -> f64 {
        if i != self.main_centroids.len() - 1 {
            return (self.main_centroids[i + 1].mean + self.main_centroids[i].mean) / 2.0;
        } else {
            return self.max;
        }
    }
}

fn new_merging(compression: f64, debug: bool) -> MergingDigest {
    // upper bound on the size of the centroid limit
    // TODO this can actually be reduced even further
    let _size_bound = ((std::f64::consts::PI * compression / 2.0) + 0.5) as i64;

    let tmp_buffer = estimate_temp_buffer(compression) as usize;

    MergingDigest {
        compression: compression,
        main_centroids: vec![],
        main_weight: 0.0,
        temp_centroids: Vec::with_capacity(tmp_buffer),
        temp_weight: 0.0,
        min: std::f64::INFINITY,
        max: std::f64::NEG_INFINITY,
        reciprocal_sum: 0.0,
        debug: debug,
    }
}

fn estimate_temp_buffer(compression: f64) -> usize {
    let temp_compression = std::cmp::min(925, std::cmp::max(20, compression as i64)) as f64;

    (7.5 + 0.37 * temp_compression - 2e-4 * temp_compression * temp_compression) as usize
}

fn min(a: f64, b: f64) -> f64 {
    if a < b {
        return a;
    }
    return b;
}

fn max(a: f64, b: f64) -> f64 {
    if a > b {
        return a;
    }
    return b;
}
