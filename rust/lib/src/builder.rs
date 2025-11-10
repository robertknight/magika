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

use crate::{Result, Session};

/// Configures and creates a Magika session.
#[derive(Debug, Default)]
pub struct Builder {
    inter_threads: Option<usize>,
    intra_threads: Option<usize>,
    parallel_execution: Option<bool>,
}

impl Builder {
    /// Configures the number of threads to parallelize the execution of the graph.
    pub fn with_inter_threads(mut self, num_threads: usize) -> Self {
        self.inter_threads = Some(num_threads);
        self
    }

    /// Configures the number of threads to parallelize the execution within nodes.
    pub fn with_intra_threads(mut self, num_threads: usize) -> Self {
        self.intra_threads = Some(num_threads);
        self
    }

    /// Configures the session parallel execution.
    pub fn with_parallel_execution(mut self, parallel_execution: bool) -> Self {
        self.parallel_execution = Some(parallel_execution);
        self
    }

    /// Consumes the builder to create a Magika session.
    pub fn build(self) -> Result<Session> {
        let bytes = include_bytes!("model.onnx");
        let model = rten::Model::load_static_slice(bytes.as_slice())?;

        // Create a per-session thread pool with the given number of threads.
        //
        // rten only supports parallelism within nodes, so the `inter_threads`
        // setting is ignored.
        let intra_threads = self.intra_threads.unwrap_or(1);
        let thread_pool = rten::ThreadPool::with_num_threads(intra_threads).into();

        Ok(Session { model, thread_pool })
    }
}
