use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenSettings {
    #[serde(flatten)]
    pub loader: LoaderConfig,

    #[serde(flatten)]
    pub gen_type: GenType,

    #[serde(flatten)]
    pub data_config: DataConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderConfig {
    pub source: String,

    pub patient_filter: Option<Vec<usize>>,

    #[serde(default)]
    pub normalize: Normalize,

    #[serde(default)]
    pub hole_filling: HoleFilling,

    #[serde(default)]
    pub end_padding: EndPadding,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct EndPadding {
    pub amount: f32,
    pub mode: HoleFillingMode,
    pub rate: f32,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct HoleFilling {
    pub mode: HoleFillingMode,
    pub rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HoleFillingMode {
    None,
    ZeroFill,
    PrevFill,
}

impl Default for HoleFillingMode {
    fn default() -> HoleFillingMode {
        HoleFillingMode::None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum GenType {
    Cnn(CnnSettings),
    Lstm(LstmSettings),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CnnSettings {
    pub window_size: usize,
    pub window_stride: usize,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmSettings {
    pub inner_window_size: usize,
    pub inner_window_stride: usize,
    #[serde(flatten)]
    pub base: CnnSettings,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub shuffle: bool,
    pub use_sensor_tag: bool,
    pub use_extra_features: bool,
    pub get_t: bool,
    #[serde(default)]
    pub one_hot_metadata: Option<OneHotMetadata>,
    #[serde(default)]
    pub classes: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normalize {
    pub rssi_center: f32,
    pub rssi_scale: f32,
    pub phase_center: f32,
    pub phase_scale: f32,
}

impl Default for Normalize {
    fn default() -> Normalize {
        Normalize {
            rssi_center: 0.0,
            rssi_scale: 1.0,
            phase_center: 0.0,
            phase_scale: 1.0,
        }
    }
}
#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize)]
pub struct OneHotMetadata {
    pub num_tags: usize,
    pub num_antennas: usize,
    pub num_frequencies: usize,
}

impl DataConfig {
    pub fn num_attributes(&self) -> usize {
        match self.one_hot_metadata {
            Some(metadata) => self.expand_categorical(metadata),
            None => self.regular(),
        }
    }

    fn regular(&self) -> usize {
        // antenna, rssi, phase
        let mut num_attributes = 3;

        if self.use_sensor_tag {
            num_attributes += 1;
        }

        if self.use_extra_features {
            // rssi_mean, rssi_max, rssi_min, rssi_stddev relative_read_count
            num_attributes += 5;
        }

        if self.use_sensor_tag && self.use_extra_features {
            // relative_id_count
            num_attributes += 1;
        }

        num_attributes
    }

    fn expand_categorical(&self, metadata: OneHotMetadata) -> usize {
        // rssi, phase
        let mut num_attributes = 2;

        // one_hot(antenna_id)
        num_attributes += metadata.num_antennas;

        // one_hot(frequency_id)
        num_attributes += metadata.num_frequencies;

        if self.use_sensor_tag {
            // one_hot(tag_id)
            num_attributes += metadata.num_tags;
        }

        if self.use_extra_features {
            // rssi_mean, rssi_max, rssi_min, rssi_stddev relative_read_count
            num_attributes += 5;
        }

        if self.use_sensor_tag && self.use_extra_features {
            // relative_id_count
            num_attributes += 1;
        }

        num_attributes
    }
}

#[derive(Debug, Copy, Clone)]
pub struct LstmBatchConfig {
    pub lstm_steps: usize,
    pub inner_window_size: usize,
    pub inner_window_stride: usize,
    pub base: BatchConfig,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    pub window_size: usize,
    pub window_stride: usize,
    pub batch_size: usize,
    pub num_windows: usize,
    pub data: DataConfig,
}
