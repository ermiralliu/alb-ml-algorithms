pub mod machine_learning;
pub mod map_category;
pub mod metrics;

use std::collections::{HashMap, HashSet};
// Your defined type
pub type SparseVector = HashMap<u16, u16, FxBuildHasher>;

use map_category::category_name_for_group_id;
use metrics::MetricsAccumulator;
use rkyv::{Archive, Deserialize, Serialize};
use rustc_hash::FxBuildHasher;

use crate::machine_learning::{
  knn_tfidf,
  mnnb::MultinomialNB,
  support_vector_machine::{SparseVec as SvmVec, SupportVectorMachine},
};

// ─── rkyv data types (unchanged) ────────────────────────────────────────────

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct DataU16 {
  matrix: Vec<Vec<u16>>,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct DataU8 {
  matrix: Vec<Vec<u8>>,
}

// ─── TF-IDF transformer (unchanged from original) ───────────────────────────

type TfidfBag = HashMap<u16, f32, FxBuildHasher>;

pub struct TfidfTransformer {
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

  pub fn fit(&mut self, training_data: &[SparseVector]) {
    self.total_docs = training_data.len();
    let mut doc_counts: HashMap<u16, usize, FxBuildHasher> = HashMap::default();
    for vec in training_data {
      for &word_id in vec.keys() {
        *doc_counts.entry(word_id).or_insert(0) += 1;
      }
    }
    for (word_id, count) in doc_counts {
      let idf_val = (self.total_docs as f32 / count as f32).ln();
      self.idf.insert(word_id, idf_val);
    }
  }

  pub fn transform(&self, input: &SparseVector) -> TfidfBag {
    let mut out = HashMap::default();
    for (word_id, count) in input {
      if let Some(idf_val) = self.idf.get(word_id) {
        out.insert(*word_id, (*count as f32) * idf_val);
      }
    }
    out
  }
}

// ─── data loading ───────────────────────────────────────────────────────────

/// Load and pair articles + categories from two rkyv files.
/// Returns `(sparse_bow_vectors, multi_labels)`.
pub fn load_dataset(
  articles_path: &str,
  categories_path: &str,
) -> Result<(Vec<SparseVector>, Vec<Vec<u8>>), Box<dyn std::error::Error>> {
  // Categories
  let cat_bytes = std::fs::read(categories_path)?;
  let cat_archived = rkyv::access::<ArchivedDataU8, rkyv::rancor::Error>(&cat_bytes)?;
  let cat_data: DataU8 = rkyv::deserialize::<DataU8, rkyv::rancor::Error>(cat_archived)?;

  // Articles
  let art_bytes = std::fs::read(articles_path)?;
  let art_archived = rkyv::access::<ArchivedDataU16, rkyv::rancor::Error>(&art_bytes)?;
  let art_data: DataU16 = rkyv::deserialize::<DataU16, rkyv::rancor::Error>(art_archived)?;

  let (bow_vecs, labels): (Vec<SparseVector>, Vec<Vec<u8>>) = art_data
    .matrix
    .into_iter()
    .zip(cat_data.matrix.into_iter())
    .map(|(article, label)| (vectorize_sparse(&article), label))
    .unzip();

  Ok((bow_vecs, labels))
}

// ─── sparse BoW ─────────────────────────────────────────────────────────────

pub fn vectorize_sparse(sequence: &[u16]) -> SparseVector {
  let mut v = SparseVector::with_hasher(FxBuildHasher::default());
  for &word_id in sequence {
    *v.entry(word_id).or_insert(0) += 1;
  }
  v
}

// ─── helpers ────────────────────────────────────────────────────────────────

/// Convert a SparseVector (HashMap<u16,u16>) to the SVM's SparseVec (HashMap<u16,f32>)
pub fn to_svm_vec(bow: &SparseVector) -> SvmVec {
  let mut m = HashMap::with_hasher(FxBuildHasher::default());
  for (&k, &v) in bow {
    m.insert(k, v as f32);
  }
  m
}

pub fn jaccard_similarity(a: &mut Vec<u8>, b: &mut Vec<u8>) -> f64 {
  if a.is_empty() && b.is_empty() {
    return 1.0;
  }
  a.sort_unstable();
  b.sort_unstable();
  let mut intersection = 0;
  let (mut i, mut j) = (0, 0);
  while i < a.len() && j < b.len() {
    match a[i].cmp(&b[j]) {
      std::cmp::Ordering::Equal => {
        intersection += 1;
        i += 1;
        j += 1;
      }
      std::cmp::Ordering::Less => i += 1,
      std::cmp::Ordering::Greater => j += 1,
    }
  }
  let union = a.len() + b.len() - intersection;
  intersection as f64 / union as f64
}
/// Prints the distribution of classes (0-32)
fn print_class_distribution(labels: &[Vec<u8>], label_name: &str) {
  let mut counts = [0usize; 33];
  for row in labels {
    for &class in row {
      if (class as usize) < counts.len() {
        counts[class as usize] += 1;
      }
    }
  }

  println!("\n--- Class Distribution: {} ---", label_name);
  for (i, count) in counts.iter().enumerate() {
    if *count > 0 {
      println!("Class {:2}: {} occurrences", i, count);
    }
  }
}

/// Prints how many unique words (u16 keys) appear as data scales
fn print_vocabulary_growth(data: &[SparseVector], name: &str, checkpoints: &[usize]) {
  let mut unique_words = HashSet::new();
  let mut checkpoint_idx = 0;

  println!("\n--- Vocabulary Growth: {} ---", name);
  for (i, row) in data.iter().enumerate() {
    let current_count = i + 1;

    // Add all unique word IDs from the current sparse vector
    for &word_id in row.keys() {
      unique_words.insert(word_id);
    }

    if checkpoint_idx < checkpoints.len() && current_count == checkpoints[checkpoint_idx] {
      println!(
        "At {:>8} rows: {:>6} unique words",
        current_count,
        unique_words.len()
      );
      checkpoint_idx += 1;
    }
  }
}


fn print_dataset_evolution(data: &[SparseVector], labels: &[Vec<u8>], name: &str, checkpoints: &[usize]) {
    let mut unique_words = HashSet::new();
    let mut class_counts = [0usize; 33];
    let mut checkpoint_idx = 0;

    println!("\n=== Evolution Analysis: {} ===", name);
    println!("{:<10} | {:<12} | {:<20}", "Rows", "Unique Words", "Top Classes (Class:Count)");
    println!("{:-<60}", "");

    for (i, (row, label_vec)) in data.iter().zip(labels.iter()).enumerate() {
        let current_count = i + 1;

        // Update Vocab
        for &word_id in row.keys() {
            unique_words.insert(word_id);
        }

        // Update Class Counts
        for &class in label_vec {
            if (class as usize) < class_counts.len() {
                class_counts[class as usize] += 1;
            }
        }

        // Checkpoint Trigger
        if checkpoint_idx < checkpoints.len() && current_count == checkpoints[checkpoint_idx] {
            // Get the 3 most frequent classes for a concise view
            let mut sorted_classes: Vec<(usize, usize)> = class_counts
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 0)
                .map(|(id, &count)| (id, count))
                .collect();
            
            // Sort by count descending
            sorted_classes.sort_by(|a, b| b.1.cmp(&a.1));
            
            let top_classes = sorted_classes.iter()
                // .take(3)
                .map(|(id, count)| format!("{}:{}", id, count))
                .collect::<Vec<_>>()
                .join(", ");

            println!(
                "{:<10} | {:<12} | {}",
                current_count,
                unique_words.len(),
                top_classes
            );
            checkpoint_idx += 1;
        }
    }
}

// ─── main ───────────────────────────────────────────────────────────────────

pub fn use_this_library_in_main() -> Result<(), Box<dyn std::error::Error>> {
  // ── Configuration ───────────────────────────────────────────────────────
  //
  // Separate paths for training data and testing data.
  // Both are pairs of (articles .rkyv, categories .rkyv).
  //
  let train_articles_path = "out/albanian_train_1.rkyv";
  let train_categories_path = "out/albanian_train_categories_1.rkyv";
  let test_articles_path = "out/albanian_test_1.rkyv";
  let test_categories_path = "out/albanian_test_categories_1.rkyv";

  // How many rows to use (None = all)
  let train_limit: Option<usize> = None;
  let test_limit: Option<usize> = None;

  // ── Load data ───────────────────────────────────────────────────────────
  println!("Loading training data...");
  let (mut train_bow, mut train_labels) = load_dataset(train_articles_path, train_categories_path)?;

  if let Some(limit) = train_limit {
    train_bow.truncate(limit);
    train_labels.truncate(limit);
  }
  println!("Training samples: {}", train_bow.len());

  println!("Loading test data...");
  let (mut test_bow, test_labels) = load_dataset(test_articles_path, test_categories_path)?;

  if let Some(limit) = test_limit {
    test_bow.truncate(limit);
    // test_labels is immutable after this; truncate a clone
  }
  let train_checkpoints = &[
    2000, 10000, 20000, 50000, 100000, 200000, 600000, 1000000, 2600000,
  ];
  let test_checkpoints = &[200, 2000, 10000, 20000, 50000, 100000, 200000, 590000];
  println!("{}", train_bow.len());
  println!("{}", test_bow.len());

//   // 1. Show Vocabulary Growth
//   print_vocabulary_growth(&train_bow, "Training Set", train_checkpoints);
//   print_vocabulary_growth(&test_bow, "Test Set", test_checkpoints);

//   // 2. Show Class Distribution
//   print_class_distribution(&train_labels, "Training Set");
//   print_class_distribution(&test_labels, "Test Set");
  print_dataset_evolution(&train_bow, &train_labels, "Training Data", train_checkpoints);
  print_dataset_evolution(&test_bow, &test_labels, "Test Data", test_checkpoints);
  return Ok(());
  let test_labels: Vec<Vec<u8>> = test_labels.into_iter().take(test_bow.len()).collect();
  println!("Test samples: {}", test_bow.len());

  // ── TF-IDF (fit on training data only) ──────────────────────────────────
  println!("Fitting TF-IDF...");
  let mut tfidf = TfidfTransformer::new();
  tfidf.fit(&train_bow);

  let train_tfidf: Vec<TfidfBag> = train_bow.iter().map(|b| tfidf.transform(b)).collect();
  let test_tfidf: Vec<TfidfBag> = test_bow.iter().map(|b| tfidf.transform(b)).collect();

  // SVM needs f32 sparse vecs (same keys, cast values)
  let train_svm: Vec<SvmVec> = train_tfidf
    .iter()
    .map(|t| {
      let mut m = HashMap::with_hasher(FxBuildHasher::default());
      for (&k, &v) in t {
        m.insert(k, v);
      }
      m
    })
    .collect();
  let test_svm: Vec<SvmVec> = test_tfidf
    .iter()
    .map(|t| {
      let mut m = HashMap::with_hasher(FxBuildHasher::default());
      for (&k, &v) in t {
        m.insert(k, v);
      }
      m
    })
    .collect();

  // ── Train KNN ───────────────────────────────────────────────────────────
  println!("\n=== Training KNN (TF-IDF, k=5) ===");
  let mut knn = knn_tfidf::KNearestNeighbors::new_with_k(5);
  knn.train(&train_tfidf, &train_labels)?;

  // ── Train Naive Bayes ───────────────────────────────────────────────────
  println!("=== Training Multinomial Naive Bayes ===");
  let mut nb = MultinomialNB::new(1.0);
  nb.threshold = 2.0; // within 2 nats of best → multi-label
  nb.train(&train_bow, &train_labels);

  // ── Train SVM ───────────────────────────────────────────────────────────
  println!("=== Training SVM (Pegasos, OvR) ===");
  let mut svm = SupportVectorMachine::new(
    1e-4, // lambda (regularisation)
    5,    // epochs
    0.0,  // score_threshold (0 = best class always included)
    4,    // max_labels
  );
  svm.train(&train_svm, &train_labels)?;

  // ── Optional: save models ───────────────────────────────────────────────
  // machine_learning::model_io::save_model(&nb,  "models/nb.rkyv")?;
  // machine_learning::model_io::save_model(&svm, "models/svm.rkyv")?;

  // ── Evaluate all three ──────────────────────────────────────────────────
  println!("\n=== Evaluating on {} test samples ===", test_bow.len());

  let mut knn_acc = MetricsAccumulator::new();
  let mut nb_acc = MetricsAccumulator::new();
  let mut svm_acc = MetricsAccumulator::new();

  for i in 0..test_bow.len() {
    let actual = &test_labels[i];

    // KNN
    let knn_pred = knn.predict_new(&test_tfidf[i]);
    knn_acc.update(&knn_pred, actual);

    // Naive Bayes
    let nb_pred = nb.predict(&test_bow[i]);
    nb_acc.update(&nb_pred, actual);

    // SVM
    let svm_pred = svm.predict(&test_svm[i]);
    svm_acc.update(&svm_pred, actual);
  }

  // ── Print reports ────────────────────────────────────────────────────────
  let label_fn = |id: u8| category_name_for_group_id(id);

  println!("\n╔══════════════════════════════════╗");
  println!("║   KNN (TF-IDF cosine, k=5)       ║");
  println!("╚══════════════════════════════════╝");
  let knn_report = knn_acc.report();
  knn_report.print_confusion_matrix(&label_fn);
  knn_report.print(&label_fn);

  println!("\n╔══════════════════════════════════╗");
  println!("║   Multinomial Naive Bayes        ║");
  println!("╚══════════════════════════════════╝");
  let nb_report = nb_acc.report();
  nb_report.print_confusion_matrix(&label_fn);
  nb_report.print(&label_fn);

  println!("\n╔══════════════════════════════════╗");
  println!("║   SVM (Pegasos OvR, linear)      ║");
  println!("╚══════════════════════════════════╝");
  let svm_report = svm_acc.report();
  svm_report.print_confusion_matrix(&label_fn);
  svm_report.print(&label_fn);

  // ── Summary table ────────────────────────────────────────────────────────
  println!("\n┌─────────────────┬──────────┬──────────┬──────────┬──────────┐");
  println!("│ Model           │ Mac.Prec │ Mac.Rec  │ Mac.F1   │ Jaccard  │");
  println!("├─────────────────┼──────────┼──────────┼──────────┼──────────┤");
  println!(
    "│ KNN             │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>8.4} │",
    knn_report.macro_precision,
    knn_report.macro_recall,
    knn_report.macro_f1,
    knn_report.jaccard_avg
  );
  println!(
    "│ Naive Bayes     │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>8.4} │",
    nb_report.macro_precision, nb_report.macro_recall, nb_report.macro_f1, nb_report.jaccard_avg
  );
  println!(
    "│ SVM             │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>8.4} │",
    svm_report.macro_precision,
    svm_report.macro_recall,
    svm_report.macro_f1,
    svm_report.jaccard_avg
  );
  println!("└─────────────────┴──────────┴──────────┴──────────┴──────────┘");

  Ok(())
}
