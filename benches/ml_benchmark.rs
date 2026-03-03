/// benches/ml_benchmark.rs
///
/// Run with:
///   cargo bench
///
/// Tune the limits below to control how long the benchmarks take.
use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rkyv::{Archive, Deserialize, Serialize};
use rustc_hash::FxBuildHasher;

use diploma_algos::{
    machine_learning::{
        knn_tfidf,
        mnnb::MultinomialNB,
        support_vector_machine::SupportVectorMachine,
    },
    TfidfTransformer, // re-exported from main or lib
};

// ─── Limits ─────────────────────────────────────────────────────────────────
/// Number of rows used to *train* the models in the benchmark.
const BENCH_TRAIN_LIMIT: usize = 2_000;
/// Number of rows used to *test* (predict) in the benchmark.
const BENCH_TEST_LIMIT: usize = 200;

// ─── rkyv data types (duplicated from main – or put them in lib.rs) ─────────

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[rkyv(compare(PartialEq), derive(Debug))]
struct DataU16 {
    matrix: Vec<Vec<u16>>,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[rkyv(compare(PartialEq), derive(Debug))]
struct DataU8 {
    matrix: Vec<Vec<u8>>,
}

type SparseVector = HashMap<u16, u16, FxBuildHasher>;
type TfidfBag    = HashMap<u16, f32, FxBuildHasher>;
type SvmVec      = HashMap<u16, f32, FxBuildHasher>;

// ─── helpers ─────────────────────────────────────────────────────────────────

fn vectorize_sparse(seq: &[u16]) -> SparseVector {
    let mut v = SparseVector::with_hasher(FxBuildHasher::default());
    for &w in seq { *v.entry(w).or_insert(0) += 1; }
    v
}

fn load_dataset(
    articles_path: &str,
    categories_path: &str,
    limit: usize,
) -> (Vec<SparseVector>, Vec<Vec<u8>>) {
    let cat_bytes = std::fs::read(categories_path).expect("categories file");
    let cat_arch  = rkyv::access::<ArchivedDataU8, rkyv::rancor::Error>(&cat_bytes).unwrap();
    let cat_data: DataU8 = rkyv::deserialize::<DataU8, rkyv::rancor::Error>(cat_arch).unwrap();

    let art_bytes = std::fs::read(articles_path).expect("articles file");
    let art_arch  = rkyv::access::<ArchivedDataU16, rkyv::rancor::Error>(&art_bytes).unwrap();
    let art_data: DataU16 = rkyv::deserialize::<DataU16, rkyv::rancor::Error>(art_arch).unwrap();

    art_data.matrix.into_iter()
        .zip(cat_data.matrix.into_iter())
        .take(limit)
        .map(|(article, label)| (vectorize_sparse(&article), label))
        .unzip()
}

/// Data fixture shared by all benchmarks (loaded once at startup).
struct BenchData {
    train_bow:   Vec<SparseVector>,
    train_tfidf: Vec<TfidfBag>,
    train_svm:   Vec<SvmVec>,
    train_labels: Vec<Vec<u8>>,

    test_bow:   Vec<SparseVector>,
    test_tfidf: Vec<TfidfBag>,
    test_svm:   Vec<SvmVec>,
}

impl BenchData {
    fn load() -> Self {
        // ── adjust paths to match your project layout ──
        let (train_bow, train_labels) = load_dataset(
            "out/albanian_train_1.rkyv",
            "out/albanian_train_categories_1.rkyv",
            BENCH_TRAIN_LIMIT,
        );
        let (test_bow, _test_labels) = load_dataset(
            "out/albanian_test_1.rkyv",
            "out/albanian_test_categories_1.rkyv",
            BENCH_TEST_LIMIT,
        );

        let mut tfidf = TfidfTransformer::new();
        tfidf.fit(&train_bow);

        let train_tfidf: Vec<TfidfBag> = train_bow.iter().map(|b| tfidf.transform(b)).collect();
        let test_tfidf:  Vec<TfidfBag> = test_bow.iter().map(|b| tfidf.transform(b)).collect();

        let to_svm = |bags: &[TfidfBag]| -> Vec<SvmVec> {
            bags.iter().map(|t| {
                let mut m = HashMap::with_hasher(FxBuildHasher::default());
                for (&k, &v) in t { m.insert(k, v); }
                m
            }).collect()
        };

        BenchData {
            train_svm:   to_svm(&train_tfidf),
            test_svm:    to_svm(&test_tfidf),
            train_tfidf,
            test_tfidf,
            train_labels,
            train_bow,
            test_bow,
        }
    }
}

// ─── KNN benchmarks ─────────────────────────────────────────────────────────

fn bench_knn_train(c: &mut Criterion) {
    let d = BenchData::load();
    c.bench_function("knn_train", |b| {
        b.iter(|| {
            let mut knn = knn_tfidf::KNearestNeighbors::new_with_k(5);
            knn.train(black_box(&d.train_tfidf), black_box(&d.train_labels)).unwrap();
        })
    });
}

fn bench_knn_predict(c: &mut Criterion) {
    let d = BenchData::load();
    let mut knn = knn_tfidf::KNearestNeighbors::new_with_k(5);
    knn.train(&d.train_tfidf, &d.train_labels).unwrap();

    c.bench_function("knn_predict_batch", |b| {
        b.iter(|| {
            for v in &d.test_tfidf {
                black_box(knn.predict_new(black_box(v)));
            }
        })
    });
}

// ─── Naive Bayes benchmarks ──────────────────────────────────────────────────

fn bench_nb_train(c: &mut Criterion) {
    let d = BenchData::load();
    c.bench_function("nb_train", |b| {
        b.iter(|| {
            let mut nb = MultinomialNB::new(1.0);
            nb.train(black_box(&d.train_bow), black_box(&d.train_labels));
        })
    });
}

fn bench_nb_predict(c: &mut Criterion) {
    let d = BenchData::load();
    let mut nb = MultinomialNB::new(1.0);
    nb.train(&d.train_bow, &d.train_labels);

    c.bench_function("nb_predict_batch", |b| {
        b.iter(|| {
            for v in &d.test_bow {
                black_box(nb.predict(black_box(v)));
            }
        })
    });
}

// ─── SVM benchmarks ─────────────────────────────────────────────────────────

fn bench_svm_train(c: &mut Criterion) {
    let d = BenchData::load();
    // SVM training is slow for large datasets; use a smaller subset
    let train_limit = BENCH_TRAIN_LIMIT;
    let small_data   = &d.train_svm[..train_limit];
    let small_labels = &d.train_labels[..train_limit];

    c.bench_function("svm_train", |b| {
        b.iter(|| {
            let mut svm = SupportVectorMachine::new(0.01, 15, 0.1, 4);
            svm.train(black_box(small_data), black_box(small_labels)).unwrap();
        })
    });
}

fn bench_svm_predict(c: &mut Criterion) {
    let d = BenchData::load();
    let mut svm = SupportVectorMachine::new(1e-4, 3, 0.0, 4);
    svm.train(&d.train_svm, &d.train_labels).unwrap();

    c.bench_function("svm_predict_batch", |b| {
        b.iter(|| {
            for v in &d.test_svm {
                black_box(svm.predict(black_box(v)));
            }
        })
    });
}

// ─── Criterion groups ────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_knn_train,
    bench_knn_predict,
    bench_nb_train,
    bench_nb_predict,
    bench_svm_train,
    bench_svm_predict,
);
criterion_main!(benches);
