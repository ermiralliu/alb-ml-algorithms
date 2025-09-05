// --- MachineLearningModel Trait ---
// This trait defines the core functionality for a machine learning model.
// It is generic over the input data type `I` and the output prediction type `O`.
// For simplicity, we'll assume our data is a vector of floating-point numbers,
// and our labels/predictions are also floating-point numbers (e.g., 0.0 or 1.0 for classes).
pub mod naive_bayes;

pub trait MachineLearningModel<I, O> {
    /// A function to create a new, uninitialized instance of the model.
    fn new() -> Self;

    /// The `fit` method for training the model.
    /// It takes a slice of training data and a slice of corresponding labels.
    /// The implementation of this method will be unique to each algorithm.
    /// For K-NN, it might just store the data. For Naive Bayes, it calculates
    /// probabilities. For a CNN, it runs a complex optimization loop.
    fn fit(&mut self, data: &[I], labels: &[O])-> Result< (), String>;

    /// The `predict` method for making a prediction on a single data point.
    /// It takes a single input data point and returns the model's prediction.
    fn predict(&self, data_point: &I) -> O;

    fn predict_multi(&self, data_point: &I) -> Vec<u32>;

    /// A method to make predictions on a batch of data.
    /// It takes a slice of input data points and returns a vector of predictions.
    /// This is often more efficient than calling `predict` in a loop.
    fn predict_batch(&self, data: &[I]) -> Vec<O> {
        data.iter().map(|d| self.predict(d)).collect()
    }

    fn finalize_training(&mut self);
}

