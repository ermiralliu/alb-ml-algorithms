// --- MachineLearningModel Trait ---
// This trait defines the core functionality for a machine learning model.
// It is generic over the input data type `I` and the output prediction type `O`.
// Or data is are tokenized sentences (integer vectors), our classes/categories are tokenized categories
pub mod naive_bayes;
pub mod naive_bayes_incremental;
pub mod k_nearest_neighbors;
pub mod convolutional_neural_networks;

pub trait MachineLearningModel<I, O> {
    /// A function to create a new, uninitialized instance of the model.
    fn new() -> Self;

    /// The `fit` method for training the model.
    /// It takes a slice of training data and a slice of corresponding labels.
    /// The implementation of this method will be unique to each algorithm.
    /// For K-NN, it might just store the data. For Naive Bayes, it calculates
    /// probabilities. For a CNN, it runs a complex optimization loop.
    fn train(&mut self, data: &[Vec<I>], labels: &[Vec<O>])-> Result< (), String>;

    /// The `predict` method for making a prediction on a single data point.
    /// It takes a single input data point and returns the model's prediction.
    fn predict(&self, data_point: &Vec<I>) -> Vec<O>;

    // fn predict_multi(&self, data_point: &I) -> Vec<O>;

    /// A method to make predictions on a batch of data.
    /// It takes a slice of input data points and returns a vector of predictions.
    /// This is often more efficient than calling `predict` in a loop.
    fn predict_batch(&self, data: &[Vec<I>]) -> Vec<Vec<O>> {
        data.iter().map(|d| self.predict(d)).collect()
    }

}

// pub trait AlbanianTextCategorizationMLModel = MachineLearningModel<Vec<u16>, u16>;

pub trait AlbanianTextCategorizationMLModel: MachineLearningModel<u16, u16> {} // unfortunately i cannot use this directly

pub trait SaveAndLoad <T> where T: bincode::Encode{ // T is the type of the ML algorithm
    fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
    fn load_from_file(path: &str) -> Result<T, Box<dyn std::error::Error>>;
}
// I can only cast or sth
// frick Rust. C++ would've allowed me