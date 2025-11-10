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

use std::path::Path;
use std::sync::Arc;

use ndarray::ArrayD;

use crate::future::{exec, AsyncEnv, Env, SyncEnv};
use crate::input::AsyncInputApi;
use crate::{AsyncInput, Builder, Features, FeaturesOrRuled, FileType, Result, SyncInput};

/// A Magika session to identify files.
#[derive(Debug)]
pub struct Session {
    pub(crate) model: rten::Model,
    pub(crate) thread_pool: Arc<rten::ThreadPool>,
}

impl Session {
    /// Creates a default session.
    pub fn new() -> Result<Self> {
        Session::builder().build()
    }

    /// Initializes a new Magika session builder with default values.
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Identifies a single file (synchronously).
    pub fn identify_file_sync(&mut self, file: impl AsRef<Path>) -> Result<FileType> {
        exec(self.identify_file::<SyncEnv>(file.as_ref()))
    }

    /// Identifies a single file (asynchronously).
    pub async fn identify_file_async(&mut self, file: impl AsRef<Path>) -> Result<FileType> {
        self.identify_file::<AsyncEnv>(file.as_ref()).await
    }

    async fn identify_file<E: Env>(&mut self, file: &Path) -> Result<FileType> {
        let metadata = E::symlink_metadata(file).await?;
        if metadata.is_dir() {
            Ok(FileType::Directory)
        } else if metadata.is_symlink() {
            Ok(FileType::Symlink)
        } else {
            debug_assert!(metadata.is_file());
            self.identify_content::<E>(E::open(file).await?).await
        }
    }

    /// Identifies a single file from its content (synchronously).
    pub fn identify_content_sync(&mut self, file: impl SyncInput) -> Result<FileType> {
        exec(self.identify_content::<SyncEnv>(file))
    }

    /// Identifies a single file from its content (asynchronously).
    pub async fn identify_content_async(&mut self, file: impl AsyncInput) -> Result<FileType> {
        self.identify_content::<AsyncEnv>(file).await
    }

    async fn identify_content<E: Env>(&mut self, file: impl AsyncInputApi) -> Result<FileType> {
        match FeaturesOrRuled::extract(file).await? {
            FeaturesOrRuled::Ruled(content_type) => Ok(FileType::Ruled(content_type)),
            FeaturesOrRuled::Features(features) => self.identify_features::<E>(&features).await,
        }
    }

    /// Identifies a single file from its features (synchronously).
    pub fn identify_features_sync(&mut self, features: &Features) -> Result<FileType> {
        exec(self.identify_features::<SyncEnv>(features))
    }

    /// Identifies a single file from its features (asynchronously).
    pub async fn identify_features_async(&mut self, features: &Features) -> Result<FileType> {
        self.identify_features::<AsyncEnv>(features).await
    }

    async fn identify_features<E: Env>(&mut self, features: &Features) -> Result<FileType> {
        let results = self.identify_features_batch::<E>(std::slice::from_ref(features)).await?;
        let [result] = results.try_into().ok().unwrap();
        Ok(result)
    }

    /// Identifies multiple files in parallel from their features (synchronously).
    pub fn identify_features_batch_sync(&mut self, features: &[Features]) -> Result<Vec<FileType>> {
        exec(self.identify_features_batch::<SyncEnv>(features))
    }

    /// Identifies multiple files in parallel from their features (asynchronously).
    pub async fn identify_features_batch_async(
        &mut self, features: &[Features],
    ) -> Result<Vec<FileType>> {
        self.identify_features_batch::<AsyncEnv>(features).await
    }

    async fn identify_features_batch<E: Env>(
        &mut self, features: &[Features],
    ) -> Result<Vec<FileType>> {
        if features.is_empty() {
            return Ok(Vec::new());
        }
        let features_size = crate::model::CONFIG.features_size();

        let bytes_id = self.model.node_id("bytes")?;
        let output_id = self.model.node_id("target_label")?;
        let input = rten::Value::from_shape(
            [features.len(), features_size],
            features.iter().flat_map(|x| &x.0).cloned().collect(),
        )
        .expect("features shape should match data");

        // Use this session's thread pool instead of RTen's default global pool.
        let thread_pool = self.thread_pool.clone();
        let opts = rten::RunOptions::default().with_thread_pool(Some(thread_pool));

        // Upstream Magika uses the ort crate's built-in async support. RTen
        // doesn't have built-in async support so use `block_in_place` here
        // to allow other tasks to run concurrently.
        let [output] = tokio::task::block_in_place(|| {
            self.model.run_n(vec![(bytes_id, input.into())], [output_id], Some(opts))
        })?;

        let (out_shape, out_data) = output.into_shape_vec::<f32, 2>()?;
        let output = ArrayD::from_shape_vec(out_shape.as_slice(), out_data)?;
        Ok(FileType::convert(output.view()))
    }
}
