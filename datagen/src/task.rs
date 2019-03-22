use std::{collections::HashMap, error::Error};

use serde::Serialize;

use lazy_static::lazy_static;
use parking_lot::RwLock;

use crate::{
    batch_gen::{BedExitGeneratorCnn, BedExitGeneratorLstm, Generator},
    config::{BatchConfig, GenSettings, GenType},
    types::Batch,
};

lazy_static! {
    static ref CONTEXT: RwLock<TaskContext> = RwLock::new(TaskContext::new());
}

pub fn get_manager() -> &'static RwLock<TaskContext> {
    &*CONTEXT
}

pub struct TaskContext {
    tasks: HashMap<String, TaskState>,
}

impl TaskContext {
    pub fn new() -> TaskContext {
        TaskContext {
            tasks: HashMap::new(),
        }
    }

    pub fn new_task(
        &mut self,
        id: String,
        settings: GenSettings,
    ) -> Result<TaskInfo, Box<dyn Error>> {
        let task = TaskState::new(id.clone(), settings)?;
        let info = task.info();
        self.tasks.insert(id, task);
        Ok(info)
    }

    pub fn list(&self) -> Vec<TaskInfo> {
        self.tasks.values().map(TaskState::info).collect()
    }

    pub fn get_task(&self, id: &str) -> Option<TaskInfo> {
        self.tasks.get(id).map(TaskState::info)
    }

    pub fn delete_task(&mut self, id: &str) -> bool {
        self.tasks.remove(id).is_some()
    }

    pub fn delete_all(&mut self) -> usize {
        let count = self.tasks.len();
        self.tasks.clear();
        count
    }

    pub fn update_task(&mut self, id: &str, step: usize) {
        if let Some(task) = self.tasks.get_mut(id) {
            while task.generated_steps() <= step {
                task.new_epoch();
            }
        }
    }

    pub fn get_batch(&self, id: &str, step: usize) -> Result<Batch, Box<dyn Error>> {
        self.tasks
            .get(id)
            .ok_or_else(|| format!("Task: {} does not exist", id))?
            .get_batch(step)
    }
}

#[derive(Serialize)]
pub struct TaskInfo {
    pub name: String,
    pub config: BatchConfig,
    pub step: usize,
    pub steps_per_epoch: usize,
}

struct TaskState {
    name: String,
    generator: Generator,
    entries: Vec<usize>,
}

impl TaskState {
    pub fn new(name: String, settings: GenSettings) -> Result<TaskState, Box<dyn Error>> {
        Ok(TaskState {
            name,
            generator: create_generator(settings)?,
            entries: vec![],
        })
    }

    /// Return metadata about the task
    pub fn info(&self) -> TaskInfo {
        TaskInfo {
            name: self.name.clone(),
            config: self.generator.config(),
            step: self.generated_steps(),
            steps_per_epoch: self.steps_per_epoch(),
        }
    }

    /// Returns the number of steps that have been prepared for generating data
    pub fn generated_steps(&self) -> usize {
        self.entries.len() / self.generator.config().batch_size
    }

    /// Returns the number of steps in a single epoch
    pub fn steps_per_epoch(&self) -> usize {
        let config = self.generator.config();
        config.num_windows / config.batch_size
    }

    /// Generates a new epoch worth of steps
    pub fn new_epoch(&mut self) {
        gen_windows(&mut self.entries, self.generator.config())
    }

    /// Gets batch at the specified step
    pub fn get_batch(&self, step: usize) -> Result<Batch, Box<dyn Error>> {
        let (start, end) = self
            .get_step_endpoints(step)
            .ok_or_else(|| format!("Step: {} has not been generated yet", step))?;

        self.generator.gen_batch(&self.entries[start..end])
    }

    /// Return the endpoints of the target step
    fn get_step_endpoints(&self, step: usize) -> Option<(usize, usize)> {
        if self.generated_steps() < step + 1 {
            return None;
        }

        let batch_size = self.generator.config().batch_size;

        let start = step * batch_size;
        let end = start + batch_size;

        Some((start, end))
    }
}

fn create_generator(settings: GenSettings) -> Result<Generator, Box<dyn Error>> {
    let data = crate::data::load(settings.loader)?;

    // Setting one_hot_metadata can be error prone, so we perform some basic validation that the
    // provided configuration is correct.
    if let Some(meta) = settings.data_config.one_hot_metadata {
        let max_tag = data.iter().map(|x| x.tag_id).max().unwrap();
        let max_antennas = data.iter().map(|x| x.antenna_id).max().unwrap();
        let max_frequencies = data.iter().map(|x| x.frequency_id).max().unwrap();

        if meta.num_tags <= max_tag {
            return Err(format!(
                "Too many tag_ids (expected <= {} got {})",
                meta.num_tags,
                max_tag + 1
            )
            .into());
        }
        if meta.num_antennas <= max_antennas {
            return Err(format!(
                "Too many antennas (expected <= {} got {})",
                meta.num_antennas,
                max_antennas + 1
            )
            .into());
        }
        if meta.num_frequencies <= max_frequencies {
            return Err(format!(
                "Too many frequencies (expected <= {} got {})",
                meta.num_frequencies,
                max_frequencies + 1
            )
            .into());
        }
    }

    match settings.gen_type {
        GenType::Lstm(lstm) => {
            if lstm.base.window_size == 0 {
                return Err("`window_size` must be > 0".into());
            }
            if lstm.inner_window_size == 0 {
                return Err("`inner_window_size` must be > 0".into());
            }

            Ok(Box::new(BedExitGeneratorLstm::new(
                data,
                lstm,
                settings.data_config,
            )))
        }

        GenType::Cnn(cnn) => {
            let num_windows = (data.len() - cnn.window_size) / cnn.window_stride;
            let config = BatchConfig {
                window_size: cnn.window_size,
                window_stride: cnn.window_stride,
                batch_size: match cnn.batch_size {
                    0 => num_windows,
                    x => x,
                },
                num_windows,
                data: settings.data_config,
            };
            Ok(Box::new(BedExitGeneratorCnn::new(data, config)))
        }
    }
}

fn gen_windows(output: &mut Vec<usize>, config: BatchConfig) {
    use rand::{seq::SliceRandom, thread_rng};

    let start = output.len();
    output.extend((0..config.num_windows).map(|x| x * config.window_stride));
    if config.data.shuffle {
        output[start..].shuffle(&mut thread_rng());
    }
}
