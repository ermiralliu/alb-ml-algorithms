pub mod machine_learning;
pub mod map_category;
use std::collections::HashMap;

use machine_learning::{bnb, mnnb};
use map_category::category_name_for_group_id;
use rkyv::{Archive, Deserialize, Serialize};
use rustc_hash::FxBuildHasher;

use crate::machine_learning::{
  knn_hashmap::{KNearestNeighbors, SparseVector},
  knn_tfidf,
};

// const FST_CATEGORY_DATA: &[u8] = include_bytes!("../data/categories.bin");

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[rkyv(
    // This will generate a PartialEq impl between archived and normal types
    compare(PartialEq),
    // bytecheck can be used to validate your data if you want
    derive(Debug),
)]

struct DataU16 {
  matrix: Vec<Vec<u16>>,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[rkyv(
    // This will generate a PartialEq impl between archived and normal types
    compare(PartialEq),
    // bytecheck can be used to validate your data if you want
    derive(Debug),
)]

struct DataU8 {
  matrix: Vec<Vec<u8>>,
}

// fn remove_first_occurrence(vec: &mut Vec<u16>, target: u16) {
//   if let Some(pos) = vec.iter().position(|&x| x == target) {
//     vec.remove(pos);
//   }
// }

fn main() -> Result<(), Box<dyn std::error::Error>> {
  println!("Hello, world!");
  // let fst_categories = Fst::new(FST_CATEGORY_DATA).unwrap();
  let file_bytes_categories = std::fs::read("out/albanian_test_categories_1.rkyv")?;
  let categories_archived =
    rkyv::access::<ArchivedDataU8, rkyv::rancor::Error>(&file_bytes_categories)?;
  let categories_matrix: DataU8 =
    rkyv::deserialize::<DataU8, rkyv::rancor::Error>(categories_archived)?;
  let (map_of_word_usages, mut categories_matrix): (Vec<SparseVector>, Vec<Vec<u8>>) = {
    let file_bytes_articles = std::fs::read("out/albanian_test_1.rkyv")?;
    let articles_archived =
      rkyv::access::<ArchivedDataU16, rkyv::rancor::Error>(&file_bytes_articles)?;
    let article_matrix = rkyv::deserialize::<DataU16, rkyv::rancor::Error>(articles_archived)?;
    let (article_matrix, categories_matrix): (Vec<Vec<u16>>, Vec<Vec<u8>>) = article_matrix
      .matrix
      .into_iter()
      .zip(categories_matrix.matrix.into_iter()) // Zip the outer Vecs
      .map(|(article, category)| {
        // 2. Map the individual category IDs within this row
        let mapped_category: Vec<u8> = category;
          // category.into_iter().map(|id| map_category_id(id)).unique().collect(); // You MUST collect here so it's a Vec<u16>, not an iterator

        (article, mapped_category)
      })
      .unzip();
    // .matrix
    // .into_iter()
    //
    // .zip(categories_matrix.matrix.into_iter().map(|x|  x.iter().map(|y| map_category_id(*y)).into_iter()).into_iter())
    // .map(|(article, mut category)| {
    // remove_first_occurrence(&mut category, 260);
    //   remove_first_occurrence(&mut category, 381);
    //   remove_first_occurrence(&mut category, 327);
    //   (article, category)
    // })
    // .filter(|(_article, category)| !(category.len() == 0) && rand::random_bool(0.1))
    // .skip(60000)
    // .take(60000)
    // .unzip();
    let article_matrix = article_matrix.iter().map(|x| vectorize_sparse(x)).collect();
    (article_matrix, categories_matrix)
  };
  println!("Data size: {}", map_of_word_usages.len());
  // KNN TEST BELOW
  // let mut knn_test = KNearestNeighbors::new_with_k(3);

  let mut tfidf_transformer = TfidfTransformer::new();
  tfidf_transformer.fit(&map_of_word_usages);
  let bag: Vec<TfidfBag> = map_of_word_usages
    .iter()
    .map(|article| tfidf_transformer.transform(article))
    .collect();

  let mut knn_test = knn_tfidf::KNearestNeighbors::new_with_k(2);
  knn_test.train(&bag, &categories_matrix)?;
  //
  // knn_test.train(&map_of_word_usages, &categories_matrix.matrix)?;
  // let data_point = &article_matrix.matrix[0];
  // knn_test.predict(data_point);
  // Multinomial naive bayes below:
  // let mut mnnb_v = bnb::MultiLabelNB::new(1.0);
  // mnnb_v.train(&map_of_word_usages, &categories_matrix);

  let test_data = &map_of_word_usages;
  let test_labels = &mut categories_matrix;

  let mut correct_predictions = 0.0;
  let total_points = test_data.len();

  for i in 0..50 {
    let mut actual_label = &mut test_labels[i];
    // knn
    let tfidf_transformed_test_data = tfidf_transformer.transform(&test_data[i]);

    // println!("\n=== Testing point {} ===", i);
    // println!("Test vector: {:?}", tfidf_transformed_test_data);
    // println!("Train vector {}: {:?}", i, bag[i]);
    // println!("Are they equal? {}", (tfidf_transformed_test_data == bag[i]));
    // KNN
    let mut prediction = knn_test.predict_new(&tfidf_transformed_test_data);
    // MNNB
    // let mut prediction = mnnb_v.predict(&map_of_word_usages[i]);
    correct_predictions += jaccard_similarity(&mut prediction, &mut actual_label);
    {
      // let actual_label_arr: Vec<Vec<u8>> = actual_label
      //   .iter()
      //   .map(|x| fst_categories.get_key(*x as u64))
      //   .flatten()
      //   .collect();
      // let actual_label_str: Vec<&str> = actual_label_arr
      //   .iter()
      //   .map(|x| unsafe { std::str::from_utf8_unchecked(&x) })
      //   .collect();
      let actual_label_str: Vec<&str> = actual_label
        .iter()
        .map(|x| category_name_for_group_id(*x))
        .collect();
      // let prediction_label_arr: Vec<Vec<u8>> = prediction
      //   .iter()
      //   .map(|x| fst_categories.get_key(*x as u64))
      //   .flatten()
      //   .collect();
      // let prediction_label_str: Vec<&str> = prediction_label_arr
      //   .iter()
      //   .map(|x| unsafe { std::str::from_utf8_unchecked(x) })
      //   .collect();
      let prediction_label_str: Vec<&str> = prediction
        .iter()
        .map(|x| category_name_for_group_id(*x))
        .collect();
      println!(
        "Prediction: {:?},\nActual label: {:?}",
        prediction_label_str, actual_label_str
      );
    }
    // if prediction == *actual_label {
    //   correct_predictions += 1;
    // }
    let accuracy = (correct_predictions as f64 / (i + 1) as f64) * 100.0;
    println!("Prediction {i} finished, accuracy: {accuracy}");

    // Since predict returns a Vec<O>, check if it matches the actual label
    // If your labels are single-element vectors, compare directly:
  }

  let accuracy = (correct_predictions as f64 / total_points as f64) * 100.0;
  println!(
    "Model Accuracy: {}% ({} out of {})",
    accuracy, correct_predictions, total_points
  );
  Ok(())
}

/* fn vectorize(sequence: &[u16], vocab_size: usize) -> Vec<u16> {
  let mut vector = vec![0u16; vocab_size];
  for &word_id in sequence {
    if (word_id as usize) < vocab_size {
      // Increment the count at the index of this word
      vector[word_id as usize] += 1;
    }
  }
  vector
} */

fn vectorize_sparse(sequence: &[u16]) -> SparseVector {
  // Initialize with the fast hasher
  let mut sparse_vec = SparseVector::with_hasher(FxBuildHasher::default());

  for &word_id in sequence {
    // .entry() handles checking if the key exists and updating it in one go
    *sparse_vec.entry(word_id).or_insert(0) += 1;
  }

  sparse_vec
}
pub fn jaccard_similarity(a: &mut Vec<u8>, b: &mut Vec<u8>) -> f64 {
  if a.is_empty() && b.is_empty() {
    return 1.0;
  }

  // Ensure they are sorted for the intersection check
  a.sort_unstable();
  b.sort_unstable();

  let mut intersection = 0;
  let mut i = 0;
  let mut j = 0;

  while i < a.len() && j < b.len() {
    if a[i] == b[j] {
      intersection += 1;
      i += 1;
      j += 1;
    } else if a[i] < b[j] {
      i += 1;
    } else {
      j += 1;
    }
  }

  let union = a.len() + b.len() - intersection;
  intersection as f64 / union as f64
}

type TfidfBag = HashMap<u16, f32, FxBuildHasher>;

pub struct TfidfTransformer {
  // Maps word_id -> Inverse Document Frequency
  idf: TfidfBag,
  total_docs: usize,
}

impl TfidfTransformer {
  pub fn new() -> Self {
    TfidfTransformer {
      idf: HashMap::default(),
      total_docs: 0,
    }
  }
  /// Step 1: Learn the IDF weights from the training corpus
  pub fn fit(&mut self, training_data: &[SparseVector]) {
    self.total_docs = training_data.len();
    let mut doc_counts: HashMap<u16, usize, FxBuildHasher> = HashMap::default();

    for vec in training_data {
      for &word_id in vec.keys() {
        *doc_counts.entry(word_id).or_insert(0) += 1;
      }
    }

    for (word_id, count) in doc_counts {
      // Formula: idf = log(Total Docs / Docs containing word)
      let idf_val = (self.total_docs as f32 / count as f32).ln();
      self.idf.insert(word_id, idf_val);
    }
  }
  // pub fn fit(&mut self, training_data: &[SparseVector]) {
  //   self.total_docs = training_data.len();
  //   let mut doc_counts: HashMap<u16, usize, FxBuildHasher> = HashMap::default();

  //   for vec in training_data {
  //     for &word_id in vec.keys() {
  //       *doc_counts.entry(word_id).or_insert(0) += 1;
  //     }
  //   }

  //   // Harsh bounds:
  //   // - Ignore words appearing in > 40% of docs (too common to be useful)
  //   // - Ignore words appearing in < 3 docs (likely noise/typos)
  //   let max_df = (self.total_docs as f32 * 0.9) as usize;
  //   let min_df = 0;

  //   self.idf = doc_counts
  //     .into_iter()
  //     .filter(|(_, count)| *count <= max_df && *count >= min_df)
  //     .map(|(word_id, count)| {
  //       // Standard smooth IDF formula
  //       let idf_val = ((self.total_docs as f32) / (count as f32)).ln();
  //       (word_id, idf_val)
  //     })
  //     .collect();
  // }
  /// Step 2: Convert a frequency vector into a weighted TF-IDF vector
  pub fn transform(&self, input: &SparseVector) -> TfidfBag {
    let mut tfidf_vec = HashMap::default();
    for (word_id, count) in input {
      if let Some(idf_val) = self.idf.get(word_id) {
        // TF-IDF = (count) * (idf)
        tfidf_vec.insert(*word_id, (*count as f32) * idf_val);
      }
    }
    tfidf_vec
  }
  //   pub fn transform(&self, input: &SparseVector) -> TfidfBag {
  //     let mut tfidf_vec = HashMap::default();
  //     let mut norm_sq = 0.0;

  //     for (word_id, count) in input {
  //         // If word was too common/rare, it's not in self.idf, so we skip it!
  //         if let Some(&idf_val) = self.idf.get(word_id) {
  //             let tf = 1.0 + (*count as f32).ln(); // Sublinear scaling
  //             let val = tf * idf_val;
  //             tfidf_vec.insert(*word_id, val);
  //             norm_sq += val * val;
  //         }
  //     }

  //     // Normalize to unit length: This makes the dot product
  //     // equal to the Cosine Similarity automatically.
  //     // let inv_norm = 1.0 / norm_sq.sqrt();
  //     // for val in tfidf_vec.values_mut() {
  //     //     *val *= inv_norm;
  //     // }
  //     if norm_sq > 0.0 {
  //         let inv_norm = 1.0 / norm_sq.sqrt();
  //         for val in tfidf_vec.values_mut() {
  //             *val *= inv_norm;
  //         }
  //     } else {
  //         // Log this! It means the input document has NO words
  //         // that exist in your filtered IDF training set.
  //         eprintln!("Warning: Document transformed to empty vector.");
  //     }

  //     tfidf_vec
  // }
  //   pub fn fit(&mut self, training_data: &[SparseVector]) {
  //     self.total_docs = training_data.len();
  //     if self.total_docs == 0 { return; }

  //     let mut doc_counts: HashMap<u16, usize, FxBuildHasher> = HashMap::default();

  //     // 1. Count document frequency (DF) for each word
  //     for vec in training_data {
  //         for &word_id in vec.keys() {
  //             *doc_counts.entry(word_id).or_insert(0) += 1;
  //         }
  //     }

  //     // 2. Set sensible bounds
  //     // If you have 100 docs, max_df is 90. If you have 10 docs, max_df is 9.
  //     let max_df = (self.total_docs as f32 * 0.9) as usize;
  //     // Ensure min_df is at least 1, but usually 2+ to filter unique typos/noise
  //     let min_df = 1;

  //     // 3. Calculate IDF with Smoothing
  //     self.idf = doc_counts
  //         .into_iter()
  //         .filter(|(_, count)| {
  //             // Only filter if we have enough data to make filtering meaningful
  //             if self.total_docs > 10 {
  //                 *count <= max_df && *count >= min_df
  //             } else {
  //                 true // Keep everything for tiny datasets
  //             }
  //         })
  //         .map(|(word_id, count)| {
  //             // Smoothed IDF: ln((N + 1) / (df + 1)) + 1
  //             // The "+ 1" outside the log ensures words with df = N still have a weight > 0
  //             let idf_val = ((self.total_docs as f32 + 1.0) / (count as f32 + 1.0)).ln() + 1.0;
  //             (word_id, idf_val)
  //         })
  //         .collect();

  //     // Debugging: check how many words survived the "purge"
  //     println!("Vocabulary size after fit: {}", self.idf.len());
  // }
  //   pub fn transform(&self, input: &SparseVector) -> TfidfBag {
  //     let mut tfidf_vec = HashMap::default();
  //     let mut norm_sq = 0.0;

  //     for (word_id, count) in input {
  //       if let Some(&idf_val) = self.idf.get(word_id) {
  //         // Ensure count > 0 to avoid ln(0)
  //         let tf = if *count > 0 {
  //           1.0 + (*count as f32).ln()
  //         } else {
  //           0.0
  //         };
  //         let val = tf * idf_val;

  //         if val > 0.0 {
  //           tfidf_vec.insert(*word_id, val);
  //           norm_sq += val * val;
  //         }
  //       }
  //     }

  //     if norm_sq > 0.0 {
  //       let inv_norm = 1.0 / norm_sq.sqrt();
  //       for val in tfidf_vec.values_mut() {
  //         *val *= inv_norm;
  //       }
  //     } else {
  //       // If this hits, your KNN will definitely fail.
  //       eprintln!("Warning: Query has no words matching the trained vocabulary.");
  //     }

  //     tfidf_vec
  //   }
}
