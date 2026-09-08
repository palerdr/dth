use crate::profiles::{Profile, potential};

pub type Buckets = [Vec<usize>; 601];

pub fn build_buckets(profiles: &[Profile]) -> Buckets {
    let mut buckets: Buckets = std::array::from_fn(|_| Vec::new());
    for (i, profile) in profiles.iter().enumerate() {
        buckets[potential(profile)].push(i);
    }
    buckets
}

pub fn layer_sizes(buckets: &Buckets) -> [usize; 1201] {
    let mut layer_sz = [0; 1201];
    for p in 0_usize..1201 {
        let mut acc: usize = 0;
        for a in p.saturating_sub(600)..=p.min(600) {
            acc += buckets[a].len() * buckets[p - a].len();
        }
        layer_sz[p] = acc;
    }
    layer_sz
}

pub fn layer_pairs(buckets: &Buckets, p: usize) -> Vec<(usize, usize)> {
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    if p <= 1200 {
        for a in p.saturating_sub(600)..=p.min(600) {
            for pd in buckets[p - a].iter() {
                for pc in buckets[a].iter() {
                    pairs.push((*pc, *pd))
                }
            }
        }
    }
    pairs
}
