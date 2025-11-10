// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// Result type of Magika functions.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum RtenError {
    // Model inference failed.
    #[error("{0}")]
    RunError(#[from] rten::RunError),

    // Loading a model failed.
    #[error("{0}")]
    LoadError(#[from] rten::LoadError),

    // Converting a value failed.
    #[error("{0}")]
    ValueError(#[from] rten::TryFromValueError),
}

/// Errors returned by Magika functions.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Input/output errors reported by the standard library.
    #[error("{0}")]
    IOError(#[from] std::io::Error),

    /// Shape errors reported by the ndarray library.
    #[error("{0}")]
    ShapeError(#[from] ndarray::ShapeError),

    /// Errors reported by the rten library.
    #[error("{0}")]
    RtenError(#[from] RtenError),
}

impl From<rten::RunError> for Error {
    fn from(val: rten::RunError) -> Self {
        Self::RtenError(val.into())
    }
}

impl From<rten::LoadError> for Error {
    fn from(val: rten::LoadError) -> Self {
        Self::RtenError(val.into())
    }
}

impl From<rten::TryFromValueError> for Error {
    fn from(val: rten::TryFromValueError) -> Self {
        Self::RtenError(val.into())
    }
}
