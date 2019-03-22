use std::error::Error;

use crate::{
    config::{BatchConfig, DataConfig, LstmBatchConfig, LstmSettings},
    types::{Batch, BedExitRecordWithFeatures, DataSet},
    utils::FlatTensor,
};

pub trait GenBatch {
    fn config(&self) -> BatchConfig;
    fn gen_batch(&self, offsets: &[usize]) -> Result<Batch, Box<dyn Error>>;
}

pub type Generator = Box<dyn GenBatch + Send + Sync>;

pub struct BedExitGeneratorCnn {
    data: DataSet,
    config: BatchConfig,
}

impl BedExitGeneratorCnn {
    pub fn new(data: DataSet, config: BatchConfig) -> BedExitGeneratorCnn {
        BedExitGeneratorCnn { data, config }
    }
}

impl GenBatch for BedExitGeneratorCnn {
    fn config(&self) -> BatchConfig {
        self.config
    }

    fn gen_batch(&self, offsets: &[usize]) -> Result<Batch, Box<dyn Error>> {
        Ok(cnn_batch(
            offsets,
            &self.data,
            |r, x_data| add_attrs(x_data, r, &self.config().data),
            |r| r.in_bed as f32,
            |r| {
                if self.config().data.classes == 3 {
                    r.three_class_label as f32
                } else {
                    r.original_label as f32
                }
            },
            |r| r.time_offset,
            &self.config,
        ))
    }
}

pub struct BedExitGeneratorLstm {
    data: DataSet,
    config: LstmBatchConfig,
}

impl BedExitGeneratorLstm {
    pub fn new(
        data: DataSet,
        settings: LstmSettings,
        data_cfg: DataConfig,
    ) -> BedExitGeneratorLstm {
        let base_config = BatchConfig {
            window_size: settings.base.window_size,
            window_stride: settings.base.window_stride,
            batch_size: settings.base.batch_size,
            num_windows: 0, // This is computed below
            data: data_cfg,
        };

        let mut config = LstmBatchConfig {
            lstm_steps: base_config.window_size,
            inner_window_size: settings.inner_window_size,
            inner_window_stride: settings.inner_window_stride,
            base: base_config,
        };

        let BatchConfig {
            window_size,
            window_stride,
            ..
        } = config.base;
        let LstmBatchConfig {
            inner_window_size,
            inner_window_stride,
            ..
        } = config;

        // Adjust the window size so that the window is large enough to generate
        // `lstm_steps` lots of inner windows
        config.base.window_size = inner_window_size + inner_window_stride * window_size;
        assert!(data.len() > config.base.window_size);
        config.base.num_windows = (data.len() - config.base.window_size) / window_stride;

        if config.base.batch_size == 0 {
            config.base.batch_size = config.base.num_windows;
        }

        BedExitGeneratorLstm { data, config }
    }
}

impl GenBatch for BedExitGeneratorLstm {
    fn config(&self) -> BatchConfig {
        self.config.base
    }

    fn gen_batch(&self, offsets: &[usize]) -> Result<Batch, Box<dyn Error>> {
        Ok(lstm_batch(
            offsets,
            &self.data,
            |r, x_data| add_attrs(x_data, r, &self.config().data),
            |r| r.in_bed as f32,
            |r| {
                if self.config().data.classes == 3 {
                    r.three_class_label as f32
                } else {
                    r.original_label as f32
                }
            },
            |r| r.time_offset,
            &self.config,
        ))
    }
}

/// Utility function for constructing a batch suitable for consumption by a Cnn-LSTM network.
///
/// x: [batch size * lstm steps, input window size, 1, num attributes]
/// y: [batch size, lstm steps]
/// t: [batch size, lstm steps] (Optional)
pub fn lstm_batch<T, F1, F2, F3, F4>(
    offsets: &[usize],
    records: &[T],
    mut select_attr: F1,
    mut select_class: F2,
    mut select_label: F3,
    mut select_t: F4,
    config: &LstmBatchConfig,
) -> Batch
where
    F1: FnMut(&T, &mut Vec<f32>),
    F2: FnMut(&T) -> f32,
    F3: FnMut(&T) -> f32,
    F4: FnMut(&T) -> f32,
{
    let &LstmBatchConfig {
        lstm_steps,
        inner_window_size,
        inner_window_stride,
        base: BatchConfig {
            data, window_size, ..
        },
    } = config;

    let num_attributes = data.num_attributes();
    let get_t = data.get_t;

    let batch_size = offsets.len();
    let num_labels = batch_size * lstm_steps;
    let num_values = num_labels * inner_window_size * num_attributes;

    let mut x_data: Vec<f32> = Vec::with_capacity(num_values);
    let mut y_data: Vec<f32> = Vec::with_capacity(num_labels);
    let mut labels: Vec<f32> = Vec::with_capacity(num_labels);
    let mut t_data: Vec<f32> = Vec::with_capacity(num_labels);

    for (batch_start, batch_end) in offsets.iter().map(|&i| (i, i + window_size)) {
        let initial_offset = batch_start;
        let final_offset = batch_end - inner_window_size;

        assert_eq!(
            (final_offset - initial_offset) / inner_window_stride,
            lstm_steps
        );

        for inner_start in (initial_offset..final_offset).step_by(inner_window_stride) {
            for i in inner_start..(inner_start + inner_window_size) {
                (select_attr)(&records[i], &mut x_data);
            }

            // We want the network to predict classes as soon as possible so we try to get it to
            // predict the final label in the window.
            let final_record = &records[inner_start + inner_window_stride - 1];
            y_data.push((select_class)(final_record));
            labels.push((select_label)(final_record));
            t_data.push((select_t)(final_record));
        }
    }

    assert_eq!(x_data.len(), num_values);
    assert_eq!(y_data.len(), num_labels);

    let mut batch = Batch::new();
    batch.insert(
        "x".into(),
        FlatTensor {
            shape: vec![
                (batch_size * lstm_steps) as u64,
                inner_window_size as u64,
                1,
                num_attributes as u64,
            ],
            data: x_data,
        },
    );
    batch.insert(
        "y".into(),
        FlatTensor {
            shape: vec![batch_size as u64, lstm_steps as u64],
            data: y_data,
        },
    );
    batch.insert(
        "label".into(),
        FlatTensor {
            shape: vec![batch_size as u64, lstm_steps as u64],
            data: labels,
        },
    );
    if get_t {
        batch.insert(
            "t".into(),
            FlatTensor {
                shape: vec![batch_size as u64, lstm_steps as u64],
                data: t_data,
            },
        );
    }

    batch
}

/// Utility function for constructing a batch suitable for consumption by a fully-connected Conv
/// network.
///
/// x: [batch size, input window size, num attributes]
/// y: [batch size, input window size]
/// t: [batch size, input window size] (Optional)
pub fn cnn_batch<T, F1, F2, F3, F4>(
    offsets: &[usize],
    records: &[T],
    mut select_attr: F1,
    mut select_class: F2,
    mut select_label: F3,
    mut select_t: F4,
    config: &BatchConfig,
) -> Batch
where
    F1: FnMut(&T, &mut Vec<f32>),
    F2: FnMut(&T) -> f32,
    F3: FnMut(&T) -> f32,
    F4: FnMut(&T) -> f32,
{
    let &BatchConfig {
        data, window_size, ..
    } = config;

    let num_attributes = data.num_attributes();
    let get_t = data.get_t;

    let mut count = 0;
    let mut x = vec![];
    let mut y = vec![];
    let mut labels = vec![];
    let mut t = vec![];

    for record in offsets
        .iter()
        .flat_map(move |&i| &records[i..i + window_size])
    {
        (select_attr)(record, &mut x);
        y.push((select_class)(record));
        labels.push((select_label)(record));
        t.push((select_t)(record));
        count += 1;
    }

    assert_eq!(x.len(), count * num_attributes);

    let mut batch = Batch::new();
    batch.insert(
        "x".into(),
        FlatTensor {
            data: x,
            shape: vec![
                (count / window_size) as u64,
                window_size as u64,
                num_attributes as u64,
            ],
        },
    );
    batch.insert(
        "y".into(),
        FlatTensor {
            data: y,
            shape: vec![(count / window_size) as u64, window_size as u64],
        },
    );
    batch.insert(
        "label".into(),
        FlatTensor {
            data: labels,
            shape: vec![(count / window_size) as u64, window_size as u64],
        },
    );
    if get_t {
        batch.insert(
            "t".into(),
            FlatTensor {
                shape: vec![(count / window_size) as u64, window_size as u64],
                data: t,
            },
        );
    }

    batch
}

fn add_attrs(data: &mut Vec<f32>, r: &BedExitRecordWithFeatures, config: &DataConfig) {
    if config.one_hot_metadata.is_some() {
        add_attrs_onehot(data, r, config);
        return;
    }

    data.extend_from_slice(&[r.antenna, r.rssi, r.phase]);

    if config.use_sensor_tag {
        data.push(r.tag);
    }

    if config.use_extra_features {
        data.extend_from_slice(&[
            r.mean_rssi,
            r.max_rssi,
            r.min_rssi,
            r.stddev_rssi,
            r.relative_read_count,
        ]);
    }

    if config.use_sensor_tag && config.use_extra_features {
        data.push(r.relative_id_count)
    }
}

fn add_attrs_onehot(data: &mut Vec<f32>, r: &BedExitRecordWithFeatures, config: &DataConfig) {
    let metadata = config.one_hot_metadata.unwrap();
    data.extend_from_slice(&[r.rssi, r.phase]);

    for id in 0..(metadata.num_antennas) {
        data.push(if r.antenna_id == id && !r.hole { 1.0 } else { 0.0 });
    }

    if config.use_sensor_tag {
        for id in 0..(metadata.num_tags) {
            data.push(if r.tag_id == id && !r.hole { 1.0 } else { 0.0 });
        }
    }

    for id in 0..(metadata.num_frequencies) {
        data.push(if r.frequency_id == id && !r.hole { 1.0 } else { 0.0 });
    }

    if config.use_extra_features {
        data.extend_from_slice(&[
            r.mean_rssi,
            r.max_rssi,
            r.min_rssi,
            r.stddev_rssi,
            r.relative_read_count,
        ]);
    }

    if config.use_sensor_tag && config.use_extra_features {
        data.push(r.relative_id_count)
    }
}
