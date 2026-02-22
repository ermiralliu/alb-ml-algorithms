use std::collections::{BinaryHeap, HashMap};

// use crate::TfidfBag;

use ordered_float::OrderedFloat;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;

// fn dot_product(b: &TfidfBag, a: &TfidfBag, a_magnitude: f32) -> f64 {
//   let mut dot = 0;
//   let b_magnitude: f64 = b.values().map(|val| (*val as f64) * (*val as f64)).sum();
//   let b_magnitude = b_magnitude.sqrt();
//   for (word_id, val_a) in a {
//     if let Some(val_b) = b.get(word_id) {
//       let a = *val_a as u64;
//       let b = *val_b as u64;
//       dot += a * b; // This is the entire calculation!
//     }
//   }
//   (dot as f64) / (b_magnitude * a_magnitude as f64)
// }

// pub struct KNearestNeighbors {
//   k: usize,
//   // I is now fixed as FxHashMap<u16, u16>
//   training_data: Vec<TfidfBag>,
//   training_labels: Vec<Vec<u16>>,
//   normalized_magnitudes: Vec<f32>,
//   classes_weights: Vec<HashMap<u16, f64, FxBuildHasher>>,
// }
// impl KNearestNeighbors {
//   pub fn new_with_k(k: usize) -> Self {
//     Self {
//       k,
//       training_data: Vec::new(),
//       training_labels: Vec::new(),
//       normalized_magnitudes: Vec::new(),
//       classes_weights: Vec::new(),
//     }
//   }

//   pub fn train(&mut self, data: &[TfidfBag], labels: &[Vec<u16>]) -> Result<(), String> {
//     if data.len() != labels.len() {
//       return Err("Data and labels must have the same length.".to_string());
//     }
//     // self.training_data = data.to_vec();
//     self.training_data = data.to_vec();
//     self.normalized_magnitudes = data
//       .iter()
//       .map(|x| {
//         let sum: u32 = x.values().map(|val| (*val as u32).pow(2)).sum();
//         (sum as f32).sqrt()
//       })
//       .collect();
//     self.training_labels = labels.to_vec();
//     let mut class_counts: HashMap<u16, usize> = HashMap::default();
//     for label_vec in labels {
//       for &label in label_vec {
//         *class_counts.entry(label).or_insert(0) += 1;
//       }
//     }

//     let total_samples = labels.len() as f64;
//     let class_weights: HashMap<u16, f64, FxBuildHasher> = class_counts
//       .into_iter()
//       .map(|(label, count)| (label, total_samples / count as f64))
//       .collect();
//     Ok(())
//   }

//   pub fn predict_new(&self, data_point: &TfidfBag) -> Vec<u16> {
//     use std::cmp::Reverse;

//     // Use a Min-Heap (via Reverse) to keep the K largest similarities
//     // Or keep track of K smallest distances if using euclidean_distance
//     let heap = self
//       .training_data
//       .par_iter()
//       .enumerate()
//       .fold(
//         || BinaryHeap::with_capacity(self.k),
//         |mut heap: BinaryHeap<(Reverse<OrderedFloat<f64>>, usize)>, (i, train_vec)| {
//           // For cosine similarity (larger is better), we want to keep the K largest
//           let similarity = dot_product(data_point, train_vec, self.normalized_magnitudes[i]);
//           let score = Reverse(OrderedFloat(similarity));

//           // Or for euclidean distance (smaller is better):
//           // let distance = train_vec.euclidean_distance(data_point);
//           // let score = OrderedFloat(distance);  // No Reverse needed

//           if heap.len() < self.k {
//             heap.push((score, i));
//           } else if let Some(worst) = heap.peek() {
//             // For similarity: if new similarity is LARGER than worst (smallest) in heap
//             if score > worst.0 {
//               heap.pop();
//               heap.push((score, i));
//             }
//           }
//           heap
//         },
//       )
//       .reduce(
//         || BinaryHeap::with_capacity(self.k),
//         |mut heap1, heap2| {
//           // Merge heap2 into heap1, keeping only K best
//           for item in heap2 {
//             if heap1.len() < self.k {
//               heap1.push(item);
//             } else if let Some(worst) = heap1.peek() {
//               if item.0 > worst.0 {
//                 heap1.pop();
//                 heap1.push(item);
//               }
//             }
//           }
//           heap1
//         },
//       );

//     //   // Voting logic
//     //   let threshold = 0.4;
//     //   let final_neighbors_count = heap.len();
//     //   let mut counts: HashMap<u16, u16, FxBuildHasher> = HashMap::default();
//     //   for (_, idx) in heap {
//     //     for label in &self.training_labels[idx] {
//     //       *counts.entry(*label).or_insert(0) += 1;
//     //     }
//     //   }
//     //
//     //   counts
//     //     .into_iter()
//     //     .filter_map(|(label, count)| {
//     //       let confidence = count as f64 / final_neighbors_count as f64;
//     //       if confidence >= threshold {
//     //         Some(label)
//     //       } else {
//     //         None
//     //       }
//     //     })
//     //     .collect()
//     //   // counts.into_keys().collect()
//     // }
//     let final_neighbors_count = heap.len();
//     if final_neighbors_count == 0 {
//       return vec![];
//     }

//     // Use f64 for weights to maintain precision
//     let mut weighted_counts: HashMap<u16, f64, FxBuildHasher> = HashMap::default();
//     let mut total_weight = 0.0;
//     let epsilon = 1e-9;

//     for (Reverse(score), idx) in heap {
//       let similarity = score.into_inner() + epsilon;

//       for label in &self.training_labels[idx] {
//         // Look up the pre-calculated weight for this specific class
//         let class_penalty_weight = self.classes_weights.get(label).unwrap_or(&1.0);

//         // Combine Similarity Weight AND Class Weight
//         let final_weight = similarity * class_penalty_weight;

//         *weighted_counts.entry(*label).or_insert(0.0) += final_weight;
//         total_weight += final_weight;
//       }
//     }
//     let threshold = 0.4;

//     weighted_counts
//       .into_iter()
//       .filter_map(|(label, sum_weight)| {
//         // Confidence is now: (Sum of weights for this label) / (Total weight of all K neighbors)
//         let confidence = sum_weight / total_weight;

//         if confidence >= threshold {
//           Some(label)
//         } else {
//           None
//         }
//       })
//       .collect()
//   }
// }

// 1. Fixed Dot Product: No more integer truncation!
fn calculate_cosine_similarity(
  query: &HashMap<u16, f32, FxBuildHasher>,
  train_vec: &HashMap<u16, f32, FxBuildHasher>,
  query_mag: f64,
  train_mag: f64,
) -> f64 {
  if query_mag == 0.0 || train_mag == 0.0 {
    return 0.0;
  }

  let mut dot = 0.0;
  // Iterate over the smaller map for speed
  if query.len() < train_vec.len() {
    for (id, q_val) in query {
      if let Some(t_val) = train_vec.get(id) {
        dot += (*q_val as f64) * (*t_val as f64);
      }
    }
  } else {
    for (id, t_val) in train_vec {
      if let Some(q_val) = query.get(id) {
        dot += (*q_val as f64) * (*t_val as f64);
      }
    }
  }

  let res = dot / (query_mag * train_mag);
  // dbg!(res);
  res
}

pub struct KNearestNeighbors {
  k: usize,
  training_data: Vec<HashMap<u16, f32, FxBuildHasher>>,
  training_labels: Vec<Vec<u8>>,
  normalized_magnitudes: Vec<f64>,
  classes_weights: HashMap<u8, f64, FxBuildHasher>,
}

impl KNearestNeighbors {
  pub fn new_with_k(k: usize) -> Self {
    Self {
      k,
      training_data: Vec::new(),
      training_labels: Vec::new(),
      normalized_magnitudes: Vec::new(),
      classes_weights: HashMap::default(),
    }
  }
  pub fn train(
    &mut self,
    data: &[HashMap<u16, f32, FxBuildHasher>],
    labels: &[Vec<u8>],
  ) -> Result<(), String> {
    if data.len() != labels.len() {
      return Err("Data and labels must have the same length.".to_string());
    }

    self.training_data = data.to_vec();
    self.training_labels = labels.to_vec();

    // Fix 2: Proper float magnitude calculation
    self.normalized_magnitudes = data
      .iter()
      .map(|vec| {
        let sum_sq: f64 = vec.values().map(|&val| (val as f64).powi(2)).sum();
        sum_sq.sqrt()
      })
      .collect();

    let mut class_counts: HashMap<u8, usize, FxBuildHasher> = HashMap::default();
    for label_vec in labels {
      for &label in label_vec {
        *class_counts.entry(label).or_insert(0) += 1;
      }
    }

    let total_samples = labels.len() as f64;
    // Fix 3: Actually assign the weights to the struct field
    // self.classes_weights = class_counts
    //   .into_iter()
    //   .map(|(label, count)| (label, total_samples / count as f64))
    //   .collect();
    self.classes_weights = class_counts
      .into_iter()
      .map(|(label, count)| {
        let weight = (total_samples / count as f64).ln() + 1.0;
        (label, weight)
      })
      .collect();

    Ok(())
  }

  pub fn predict_new(&self, data_point: &HashMap<u16, f32, FxBuildHasher>) -> Vec<u8> {
    use std::cmp::Reverse;

    // Fix 4: Pre-calculate query magnitude once
    let query_mag: f64 = data_point
      .values()
      .map(|&v| (v as f64).powi(2))
      .sum::<f64>()
      .sqrt();
    if query_mag == 0.0 {
      return vec![];
    }

    let heap = self
      .training_data
      .par_iter()
      .enumerate()
      .fold(
        || BinaryHeap::with_capacity(self.k),
        |mut heap, (i, train_vec)| {
          let sim = calculate_cosine_similarity(
            data_point,
            train_vec,
            query_mag,
            self.normalized_magnitudes[i],
          );
          let score = Reverse(OrderedFloat(sim));

          if heap.len() < self.k {
            heap.push((score, i));
          } else if let Some(worst) = heap.peek() {
            if score < worst.0 {
              let _popped = heap.pop();
              // dbg!(popped);
              // dbg!(score);
              heap.push((score, i));
            }
            // dbg!(&heap);
            // println!("{:?}", heap);
          }
          heap
        },
      )
      .reduce(
        || BinaryHeap::with_capacity(self.k),
        |mut h1, h2| {
          // println!("{:?}", h1);
          for item in h2 {
            if h1.len() < self.k {
              h1.push(item);
            } else if item.0 < h1.peek().unwrap().0 {
              h1.pop();
              h1.push(item);
            }
          }
          h1
        },
      );
    println!("Final heap: {:?}", heap);

    // Voting logic
    let mut weighted_counts: HashMap<u8, f64, FxBuildHasher> = HashMap::default();
    // let mut total_weight = 0.0;

    // for (Reverse(score), idx) in heap {
    //   let similarity = score.into_inner();
    //   if similarity <= 0.0 {
    //     continue;
    //   }

    //   for label in &self.training_labels[idx] {
    //     let weight = self.classes_weights.get(label).unwrap_or(&1.0);
    //     let final_weight = similarity * weight;
    //     *weighted_counts.entry(*label).or_insert(0.0) += final_weight;
    //     total_weight += final_weight;
    //   }
    // }
    // for (Reverse(score), idx) in heap {
    //   let similarity = score.into_inner();
    //
    //   // Exponentially reward high similarity to prevent "noisy" neighbors
    //   // from out-voting a perfect match.
    //   // let boosted_similarity = (similarity + 0.05).powi(4);
    //   // let boosted_similarity = (boosted_similarity + 0.2).powi(2);
    //   let boosted_similarity = (similarity * 100.0).powi(2);
    //
    //   for label in &self.training_labels[idx] {
    //     // let class_weight = self.classes_weights.get(label).unwrap_or(&1.0);
    //     let class_weight = 1.0;
    //     let final_weight = boosted_similarity * class_weight;
    //
    //     *weighted_counts.entry(*label).or_insert(0.0) += final_weight;
    //     total_weight += final_weight;
    //   }
    // }
    //
    let mut dot_total_size = 0.0;
    for (Reverse(score), idx) in heap {
      dot_total_size += score.into_inner();
      for label in &self.training_labels[idx] {
        // let class_weight = self.classes_weights.get(label).unwrap_or(&1.0);
        *weighted_counts.entry(*label).or_insert(0.0) += score.into_inner();
      }
    }

    if dot_total_size == 0.0 {
      return vec![];
    }


    // let max_confidence = weighted_counts
    //   .values()
    //   .map(|&v| v / total_weight)
    //   .fold(0.0, f64::max);
    // let threshold = 8000.0 / (self.k as f64).log(2.0); // Lowered threshold for better recall
    let threshold = 0.9 / self.k as f64; // this needs better logic

    weighted_counts
      .into_iter()
      .filter_map(|(label, sum_weight)| {
        // let conf = sum_weight / total_weight;
        // let conf = sum_weight / (self.k as f64);
        let conf = sum_weight/dot_total_size;
        dbg!(&label, conf);
        if conf >= threshold
            // || (conf == max_confidence && max_confidence > 0.0) 
        {
          Some(label)
        } else {
          None
        }
      })
      .collect()
  }
}
