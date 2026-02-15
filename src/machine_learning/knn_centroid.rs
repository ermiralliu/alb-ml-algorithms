use ordered_float::{Float, OrderedFloat};
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap};

use rustc_hash::FxBuildHasher;

pub trait EuclideanDistance<Other> {
  fn euclidean_distance(&self, other: &Other) -> f64;
}

pub type SparseVector = HashMap<u16, u16, FxBuildHasher>;

// Implement for any HashMap with any hasher S
impl EuclideanDistance<SparseVector> for SparseVector {
  fn euclidean_distance(&self, other: &SparseVector) -> f64 {
    let mut sum_sq = 0.0;

    // Iterate over 'self' and find matches in 'other'
    for (word_id, count_a) in self {
      let count_b = other.get(word_id).unwrap_or(&0);
      let diff = (*count_a as f64) - (*count_b as f64);
      sum_sq += diff * diff;
    }

    // Add contributions from words only present in 'other'
    for (word_id, count_b) in other {
      if !self.contains_key(word_id) {
        let diff = 0.0 - (*count_b as f64);
        sum_sq += diff * diff;
      }
    }

    sum_sq.sqrt()
  }
}
// In your predict loop, if vectors are pre-normalized:
fn dot_product(b: &SparseVector, a: &SparseVector, a_magnitude: f32) -> f64 {
  let mut dot = 0;
  let b_magnitude: f64 = b.values().map(|val| val.pow(2) as f64).sum();
  if a_magnitude == 0.0 || b_magnitude == 0.0 {
    return 0.0;
  }
  let b_magnitude = b_magnitude.sqrt();
  for (word_id, val_a) in a {
    if let Some(val_b) = b.get(word_id) {
      let a = *val_a as u64;
      let b = *val_b as u64;
      dot += a * b; // This is the entire calculation!
    }
  }
  (dot as f64) / (b_magnitude * a_magnitude as f64)
}

pub fn cosine_similarity_full(
    a: &SparseVector, 
    b: &SparseVector, 
) -> f64 {
    let mag_a: u64 = a.values().map(|x| (*x as u64).pow(2)).sum();
    let mag_a = (mag_a as f64).sqrt();
    let mag_b: u64 = b.values().map(|x| (*x as u64).pow(2)).sum();
    let mag_b = (mag_b as f64).sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    // Optimization: Iterate over the vector with fewer non-zero elements
    let (small, large) = if a.len() < b.len() { (a, b) } else { (b, a) };

    let dot_product: f64 = small
        .iter()
        .filter_map(|(key, &val_a)| {
            large.get(key).map(|&val_b| (val_a as f64) * (val_b as f64))
        })
        .sum();

    // The Cosine Similarity formula: (A · B) / (||A|| * ||B||)
    dot_product / (mag_a * mag_b)
}

pub struct KNearestNeighbors {
  k: usize,
  // I is now fixed as FxHashMap<u16, u16>
  training_data: Vec<SparseVector>,
  training_labels: Vec<Vec<u16>>,
  normalized_magnitudes: Vec<f32>,
  centroids: Vec<SparseVector>,
  buckets: Vec<Vec<usize>>,
}

impl KNearestNeighbors {
  pub fn new_with_k(k: usize) -> Self {
    Self {
      k,
      training_data: Vec::new(),
      training_labels: Vec::new(),
      normalized_magnitudes: Vec::new(),
      centroids: Vec::new(),
      buckets: Vec::new(),
    }
  }

  pub fn train(&mut self, data: &[SparseVector], labels: &[Vec<u16>]) -> Result<(), String> {
    if data.len() != labels.len() {
      return Err("Data and labels must have the same length.".to_string());
    }
    // self.training_data = data.to_vec();
    self.training_data = data.to_vec();
    self.normalized_magnitudes = data
      .iter()
      .map(|x| {
        let sum: u32 = x.values().map(|val| (*val as u32).pow(2)).sum();
        (sum as f32).sqrt()
      })
      .collect();
    self.training_labels = labels.to_vec();
    // Number of clusters (sqrt of N is a good rule of thumb, so ~775 for 600k)
    let k = (data.len() as f64).sqrt() as usize; 
    
    // A. Pick initial centroids (just take the first K articles for simplicity)
    let mut centroids: Vec<SparseVector> = data.iter().take(k).cloned().collect();
    
    // B. Create "Buckets" to hold indices of articles
    // buckets[centroid_index] = Vec<article_index>
    let mut buckets: Vec<Vec<usize>> = vec![vec![]; k];

    // C. Assign every article to the nearest centroid (Single Pass)
    for (article_idx, article) in data.iter().enumerate() {
        let mut best_centroid = 0;
        let mut max_sim = -1.0;

        for (c_idx, centroid) in centroids.iter().enumerate() {
            let sim = cosine_similarity_full(article, centroid);
            if sim > max_sim {
                max_sim = sim;
                best_centroid = c_idx;
            }
        }
        buckets[best_centroid].push(article_idx);
    }

    // D. Recalculate Centroids as the Mean of their buckets
    for (c_idx, article_indices) in buckets.iter().enumerate() {
        if article_indices.is_empty() { continue; }
        
        let mut mean_map: HashMap<u16, f32> = HashMap::new();
        let count = article_indices.len() as f32;

        for &idx in article_indices {
            for (&dim, &val) in &data[idx] {
                *mean_map.entry(dim).or_insert(0.0) += val as f32 / count;
            }
        }
        
        // Update centroid with the new mean
        centroids[c_idx] = mean_map; 
    }

    // Store these in your struct for use during 'predict'
    self.centroids = centroids;
    self.buckets = buckets;
    Ok(())
  }

  pub fn predict(&self, data_point: &SparseVector) -> Vec<u16> {
    let mut distances: Vec<(f64, usize)> = self
      .training_data
      .iter()
      .enumerate()
      .map(|(i, d)| (d.euclidean_distance(data_point), i))
      // .map(|(i, d)| (dot_product(data_point, d, self.normalized_magnitudes[i]), i))
      .collect();

    // Sort by distance (lowest first)
    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut counts = HashMap::new();
    for i in 0..self.k.min(distances.len()) {
      let label_index = distances[i].1;
      // Assuming training_labels is Vec<Vec<O>> as per your code
      for label in &self.training_labels[label_index] {
        *counts.entry(label.clone()).or_insert(0) += 1;
      }
    }

    // Return the unique labels found in the top K
    counts.into_keys().collect()
  }
  pub fn predict_new(&self, data_point: &SparseVector) -> Vec<u16> {
    // 1. Initialize a Max-Heap with capacity K
    let mut heap: BinaryHeap<(OrderedFloat<f64>, usize)> = BinaryHeap::with_capacity(self.k);

    // 2. Scan the training data
    self
      .training_data
      .par_iter()
      .enumerate()
      .fold(
        || BinaryHeap::with_capacity(self.k), // Create a heap for EACH thread
        |mut heap, (i, train_vec)| {
          // let dist = calculate_dist(train_vec, data_point);
          let dist = OrderedFloat(dot_product(
            data_point,
            train_vec,
            self.normalized_magnitudes[i],
          ));
          if heap.len() < self.k {
            // Still filling up the initial K slots
            heap.push((dist, i));
          } else {
            // Heap is full. Check if current distance is better than the "worst of the best"
            if let Some(mut top) = heap.peek_mut() {
              if dist < top.0 {
                // This new point is closer! Replace the furthest neighbor in our heap.
                top.0 = dist;
                top.1 = i;
              }
            }
          }
          heap
        },
      )
      .reduce(
        || BinaryHeap::with_capacity(self.k),
        |mut heap1, mut heap2| {
          // Merge the heaps from different threads
          heap1.append(&mut heap2);
          heap1
        },
      );

    // 3. Voting Logic (Heap now contains exactly the K closest neighbors)
    let mut counts = HashMap::new();
    while let Some(neighbor) = heap.pop() {
      for label in &self.training_labels[neighbor.1] {
        *counts.entry(label.clone()).or_insert(0) += 1;
      }
    }

    counts.into_keys().collect()
  }
}
