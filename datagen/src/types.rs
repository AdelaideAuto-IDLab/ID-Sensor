use std::collections::BTreeMap;

use serde::{Serialize, Deserialize};

pub type Batch = BTreeMap<String, crate::utils::FlatTensor>;
pub type DataSet = Vec<BedExitRecordWithFeatures>;

#[derive(Debug, Deserialize)]
pub struct MetadataRecord {
    pub id: usize,
    pub patient_id: usize,
    pub offset: usize,
    pub len: usize,
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub struct BedExitRecord {
    pub time_offset: f32,
    pub hole: bool,
    pub tag_id: i64,
    pub antenna_id: i64,
    pub rssi: f32,
    pub phase: f32,
    pub frequency: f32,
    #[serde(rename = "label")]
    pub in_bed: u8,
    pub three_class_label: u8,
    pub original_label: u8,
}

#[derive(Debug, Copy, Clone, Serialize)]
pub struct BedExitRecordWithFeatures {
    pub time_offset: f32,
    pub hole: bool,
    pub tag: f32,
    pub tag_id: usize,
    pub antenna: f32,
    pub antenna_id: usize,
    pub rssi: f32,
    pub phase: f32,
    pub frequency: f32,
    pub frequency_id: usize,
    pub in_bed: u8,
    pub three_class_label: u8,
    pub original_label: u8,
    pub mean_rssi: f32,
    pub max_rssi: f32,
    pub min_rssi: f32,
    pub stddev_rssi: f32,
    pub int_phase: f32,
    pub relative_read_count: f32,
    pub relative_id_count: f32,
}
