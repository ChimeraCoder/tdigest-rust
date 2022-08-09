#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

#[derive(Clone)]
struct Centroid {
    mean: f64,
    weight: f64,
    samples: Vec<f64>,
}

struct MergingDigest {
    compression: f64,
    
    main_centroids:Vec<Centroid>,
    main_weight:f64,

    tempCentroids:Vec<Centroid>,
    temp_weight:f64,

    min:f64,
    max:f64,
    reciprocalSum:f64,

    debug:bool,
}

impl MergingDigest {
    fn Add(&mut self, value: f64, weight: f64){
        //TODO check for invalid weight

        if self.tempCentroids.len() == self.tempCentroids.capacity() {
            self.merge_all_temps();
        }
    }

    // in-place merge into main_centroids
    fn merge_all_temps(&mut self){
        if self.tempCentroids.len() == 0 {
            return
        }

        self.tempCentroids.sort_by(|a,b| a.mean.partial_cmp(&b.mean).unwrap_or(std::cmp::Ordering::Equal));

        let mut tmp_index = 0;

        let total_weight = self.main_weight + self.temp_weight;

        let mut merged_weight = 0.0;

        let mut last_merged_index = 0.0;

        let mut actual_main_centroids = self.main_centroids.clone();
        self.main_centroids = Vec::with_capacity(actual_main_centroids.capacity());

        let mut swapped_centroids: Vec<Centroid> = Vec::new();

        while actual_main_centroids.len() + swapped_centroids.len() != 0 || tmp_index < self.tempCentroids.len(){
            let mut next_temp = Centroid{
                mean: std::f64::INFINITY,
                weight: 0.0,
                samples: vec![],
            };

            if tmp_index < self.tempCentroids.len() {
                next_temp = self.tempCentroids[tmp_index].clone();
            }

            let mut next_main = Centroid{
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

                last_merged_index = self.mergeOne(merged_weight, total_weight, last_merged_index, next_main);
                merged_weight += next_main_weight;

            } else {
                if actual_main_centroids.len() != 0{
                    swapped_centroids.push(actual_main_centroids[0].clone());
                    actual_main_centroids = actual_main_centroids[1..].to_vec();
                }
                tmp_index+=1;

                // TODO another way to satisfy the borrow checker?
                let next_temp_weight = next_temp.weight;

                last_merged_index = self.mergeOne(merged_weight, total_weight, last_merged_index, next_temp);
                merged_weight += next_temp_weight;
            }
        }

        self.tempCentroids = Vec::with_capacity(self.tempCentroids.capacity());
        self.temp_weight = 0.0;
        self.main_weight = total_weight;
    }

    fn mergeOne(&mut self, beforeWeight: f64, totalWeight: f64, beforeIndex: f64, next: Centroid) ->f64{

        let next_index = self.indexEstimate((beforeWeight + next.weight) / totalWeight);

        if (next_index - beforeIndex) > 1.0 || (self.main_centroids.len() == 0) {
            // the new index is far away from the last index of the current centroid
            // thereofre we cannot merge into the current centroid
            // or it would become to wide
            // so we will append a new centroid
            self.main_centroids.push(next);

            // return the last index that was merged into the previous centroid
            return self.indexEstimate(beforeWeight/totalWeight);
        } else {

            let main_centroids_len = self.main_centroids.len();


            // the new index fits into the range of the current centroid
            // so we combine it into the current centroid's values
            // this computation is known as Welford's method, and the order matters
            // weight must be updated before mean
            self.main_centroids[main_centroids_len-1].weight += next.weight;
            self.main_centroids[main_centroids_len-1].mean += (next.mean - self.main_centroids[main_centroids_len - 1].mean) * next.weight / self.main_centroids[main_centroids_len - 1].weight;
            if self.debug {
                // TODO how to append succinctly?

                for sample in next.samples.into_iter() {
                    self.main_centroids[main_centroids_len -1 ].samples.push(sample);
                }

            }

            // we did not create a new centroid, so the trailing index of the previous centroid
            // remains

            return beforeIndex;
        }

    }

    fn indexEstimate(&mut self, quantile: f64) -> f64 {
        let asin = ((2.0 * quantile - 1.0) as f64).asin() / std::f64::consts::PI;
        return self.compression * (asin + 0.5);

    }
}

fn new_merging(compression: f64, debug: bool) -> MergingDigest {

    let size_bound = ((std::f64::consts::PI * compression/2.0) + 0.5) as i64;

    MergingDigest{
        compression: compression,
        main_centroids: vec![],
        main_weight: 0.0,
        tempCentroids: Vec::with_capacity(estimate_temp_buffer(compression) as usize),
        temp_weight: 0.0,
        min: std::f64::INFINITY,
        max: std::f64::NEG_INFINITY,
        reciprocalSum: 0.0,
        debug: debug,
    }
}

fn estimate_temp_buffer(compression: f64) -> i64 {
    let temp_compression = std::cmp::min(925, std::cmp::max(20, compression as i64)) as f64;

    (7.5 + 0.37 * temp_compression - 2e4 * temp_compression * temp_compression) as i64
}
