/// model_io.rs
/// Save and load ML models using rkyv zero-copy serialisation.
///
/// Each model (MultinomialNB, SupportVectorMachine, KNearestNeighbors) must
/// derive rkyv::{Archive, Serialize, Deserialize}.
///
/// Usage:
///   save_model(&my_nb, "models/nb.rkyv")?;
///   let nb: MultinomialNB = load_model("models/nb.rkyv")?;
use std::path::Path;

use rkyv::{
    api::high::{HighDeserializer, HighSerializer, HighValidator},
    bytecheck::CheckBytes,
    rancor::Error,
    ser::allocator::ArenaHandle,
    util::AlignedVec,
    Archive, Deserialize, Serialize,
};

// ─── generic helpers ────────────────────────────────────────────────────────

/// Serialise `value` to an `AlignedVec` using rkyv.
pub fn to_bytes<T>(value: &T) -> Result<AlignedVec, Error>
where
    T: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, Error>>,
{
    rkyv::to_bytes::<Error>(value)
}

/// Deserialise a `T` from raw bytes previously written by `to_bytes`.
pub fn from_bytes<T>(bytes: &[u8]) -> Result<T, Error>
where
    T: Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, Error>>
        + Deserialize<T, HighDeserializer<Error>>,
{
    rkyv::from_bytes::<T, Error>(bytes)
}

// ─── file-based API ─────────────────────────────────────────────────────────

/// Serialise any rkyv-capable model to a file.
///
/// ```rust,ignore
/// save_model(&my_svm, "models/svm.rkyv")?;
/// ```
pub fn save_model<T, P>(model: &T, path: P) -> Result<(), Box<dyn std::error::Error>>
where
    P: AsRef<Path>,
    T: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, Error>>,
{
    let bytes = to_bytes(model)?;
    // Create parent directories if needed
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes.as_slice())?;
    Ok(())
}

/// Deserialise a model from a file previously written by `save_model`.
///
/// ```rust,ignore
/// let svm: SupportVectorMachine = load_model("models/svm.rkyv")?;
/// ```
pub fn load_model<T, P>(path: P) -> Result<T, Box<dyn std::error::Error>>
where
    P: AsRef<Path>,
    T: Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, Error>>
        + Deserialize<T, HighDeserializer<Error>>,
{
    let bytes = std::fs::read(path)?;
    let model = from_bytes::<T>(&bytes)?;
    Ok(model)
}
