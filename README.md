t-digest
============

A Rust implementation of the [t-digest algorithm](https://arxiv.org/abs/1902.04023) for fast computation of histograms and rank-based statistics (such as median, percentiles, trimmed means, etc.).


The traditional method for computing histograms and ranked-based statistics requires storing the entire dataset, which is infeasible for stream operations and large-scale data.

Like the [reference implementation in Java](https://github.com/tdunning/t-digest), this implementation of the t-digest is fast, uses bounded memory, and is suitable for distributed computation (such as map-reduce).


Usage & Example
--------

```rust
    // The compression factor allows you to trade off precision
    // for memory usage.
    const COMPRESSION: f64 = 20.0;
    let mut td = tdigest::new_merging(COMPRESSION, false);
    for _ in 0..1000 {
        td.add(generate_rand(), 1.0);
    }

    println!("median is {}", td.quantile(0.5));
    println!("99th percentile is {}", td.quantile(0.99));
    println!("min is {}", td.min());
    println!("reciprocal sum is {}", td.reciprocal_sum());
```

Contributing
------------

Contributions and pull requests are welcome.


License
--------

This project is provided under the MIT license. It draws inspiration from the [Java reference implementation](https://github.com/tdunning/t-digest) and the [Go implementation](https://github.com/stripe/veneur#approximate-histograms) of the algorithm, which are provided under the [Apache](https://github.com/tdunning/t-digest/blob/main/LICENSE) and [MIT](https://github.com/stripe/veneur/blob/master/LICENSE) licenses, respectively.
