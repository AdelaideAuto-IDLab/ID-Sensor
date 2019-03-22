use std::{
    collections::{HashMap, VecDeque},
    error::Error,
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::config::{EndPadding, HoleFilling, HoleFillingMode, LoaderConfig};
use crate::types::{BedExitRecord, BedExitRecordWithFeatures};

#[derive(Deserialize)]
struct AttributeRecord {
    time: f64,
    // These fields are contained in the original csv file, but are not used in this project
    #[allow(dead_code)] acc_frontal: f64,
    #[allow(dead_code)] acc_vertical: f64,
    #[allow(dead_code)] acc_lateral: f64,
    antenna: i64,
    rssi: f64,
    tag_id: i64,
    phase: f64,
    frequency: f64,
}

const LABEL_SEPARATION: u8 = 0;
const LABEL_SIT_ON_BED: u8 = 1;
const LABEL_SIT_ON_CHAIR: u8 = 2;
const LABEL_LYING_ON_BED: u8 = 3;
const LABEL_WALK: u8 = 4;
const LABEL_STAND_UP: u8 = 5;

fn is_out_of_bed(label: u8) -> bool {
    label != LABEL_SIT_ON_BED && label != LABEL_LYING_ON_BED
}

fn three_class_label(label: u8) -> u8 {
    match label {
        LABEL_SIT_ON_BED | LABEL_LYING_ON_BED => 0,
        LABEL_SIT_ON_CHAIR => 1,
        LABEL_WALK | LABEL_STAND_UP => 2,
        _ => panic!("Invalid label in dataset: {}", label),
    }
}

#[derive(Deserialize)]
struct LabelRecord {
    label: u8,
}

pub fn load(config: LoaderConfig) -> Result<Vec<BedExitRecordWithFeatures>, Box<dyn Error>> {
    let normalize = config.normalize;
    let (labels, attributes) = load_records(&config.source)?;

    let mut patient_id = 0;

    let mut runs = vec![];
    let mut current_patient: Vec<BedExitRecord> = vec![];

    for (label, attributes) in labels.into_iter().zip(attributes.into_iter()) {
        if label.label == LABEL_SEPARATION {
            if current_patient.len() > 0 {
                runs.push(current_patient.clone());
                current_patient.clear();
            }

            patient_id += 1;
            continue;
        }

        if let Some(filter) = &config.patient_filter {
            if !filter.contains(&patient_id) {
                continue;
            }
        }

        if attributes.tag_id == 1 {
            // Data was collected with an additional tag with accelerometer data, this is not used
            // for this analysis
            continue;
        }

        assert!(attributes.tag_id == 2 || attributes.tag_id == 3);
        assert!(attributes.antenna == 1 || attributes.antenna == 2 || attributes.antenna == 3);

        current_patient.push(BedExitRecord {
            time_offset: attributes.time as f32,
            hole: false,
            tag_id: (attributes.tag_id - 2),
            antenna_id: (attributes.antenna - 1),
            rssi: (attributes.rssi as f32 - normalize.rssi_center) / normalize.rssi_scale,
            phase: (attributes.phase as f32 - normalize.phase_center) / normalize.phase_scale,
            frequency: attributes.frequency as f32,
            in_bed: if is_out_of_bed(label.label) { 1 } else { 0 },
            three_class_label: three_class_label(label.label),
            original_label: label.label - 1,
        });
    }

    if current_patient.len() > 0 {
        runs.push(current_patient);
    }

    for (i, patient) in runs.iter_mut().enumerate() {
        println!("patient: {}, entries: {}", i, patient.len());
        *patient = fill_holes(&patient, &config.hole_filling);
        println!("patient: {}, after hole filling: {}", i, patient.len());

        apply_end_padding(patient, &config.end_padding);
        println!("patient: {}, after padding: {}", i, patient.len());

        let bed_exit_count = count_bed_exit_events(patient.iter().map(|x| x.in_bed));
        println!("patient: {}, bed exits: {}", i, bed_exit_count);
    }

    let data: Vec<_> = runs.iter().flat_map(|x| gen_features(30.0, x)).collect();

    println!("loaded: {} entries ({} patients)", data.len(), runs.len());
    println!(
        "total {} bed exit events",
        count_bed_exit_events(data.iter().map(|x| x.in_bed))
    );

    Ok(data)
}

fn load_records(
    source: impl AsRef<Path>,
) -> Result<(Vec<LabelRecord>, Vec<AttributeRecord>), Box<dyn Error>> {
    let base_path = source.as_ref();

    let labels: Result<Vec<LabelRecord>, _> = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(base_path.join("YL.csv"))?
        .deserialize()
        .collect();

    let attributes: Result<Vec<AttributeRecord>, _> = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(base_path.join("XL.csv"))?
        .deserialize()
        .collect();

    Ok((labels?, attributes?))
}
fn gen_features(window_size: f32, records: &[BedExitRecord]) -> Vec<BedExitRecordWithFeatures> {
    let mut window = VecDeque::new();
    let mut output = vec![];

    let mut frequency_map: HashMap<i64, usize> = HashMap::new();

    for record in records {
        let next_id = frequency_map.len();
        let frequency_id = *frequency_map
            .entry((record.frequency * 100.0).round() as i64)
            .or_insert(next_id);

        window.push_back(record);

        while let Some(entry) = window.front() {
            if record.time_offset - entry.time_offset < window_size {
                break;
            }
            window.pop_front();
        }

        let window_duration = (record.time_offset - window.front().unwrap().time_offset).max(0.1);

        assert!(record.tag_id >= 0);
        assert!(record.antenna_id >= 0);

        output.push(BedExitRecordWithFeatures {
            time_offset: record.time_offset,
            hole: record.hole,
            tag: record.tag_id as f32,
            tag_id: record.tag_id as usize,
            antenna_id: record.antenna_id as usize,
            antenna: record.antenna_id as f32,
            rssi: record.rssi,
            phase: record.phase,
            frequency: record.frequency,
            frequency_id,
            in_bed: record.in_bed,
            three_class_label: record.three_class_label,
            original_label: record.original_label,

            mean_rssi: stats::mean(window.iter().map(|x| x.rssi)) as f32,
            max_rssi: window
                .iter()
                .map(|x| x.rssi)
                .max_by(|a, b| a.partial_cmp(&b).unwrap())
                .unwrap() as f32,
            min_rssi: window
                .iter()
                .map(|x| x.rssi)
                .min_by(|a, b| a.partial_cmp(&b).unwrap())
                .unwrap() as f32,
            stddev_rssi: stats::stddev(window.iter().map(|x| x.rssi)) as f32,
            int_phase: window.iter().map(|x| x.phase).sum::<f32>() / (20.0 * window_duration),
            relative_read_count: (window.len() as f32 / (20.0 * window_duration)),
            relative_id_count: window.iter().map(|x| x.tag_id as f32).sum::<f32>()
                / window.len() as f32,
        });
    }

    output
}

fn zero_fill(t: f32, prev: BedExitRecord, _next: BedExitRecord) -> BedExitRecord {
    BedExitRecord {
        time_offset: t,
        // This is required to prevent the tag_id and antenna_id to be set when one_hot encoded
        hole: true,
        tag_id: 0,
        antenna_id: 0,
        rssi: 0.0,
        phase: 0.0,
        ..prev
    }
}

fn prev_fill(t: f32, prev: BedExitRecord, _next: BedExitRecord) -> BedExitRecord {
    BedExitRecord {
        time_offset: t,
        ..prev
    }
}

/// Fills holes in input data with zeroes
fn fill_holes(input: &[BedExitRecord], config: &HoleFilling) -> Vec<BedExitRecord> {
    let fill_mode = match config.mode {
        HoleFillingMode::None => return input.to_vec(),
        HoleFillingMode::ZeroFill => zero_fill,
        HoleFillingMode::PrevFill => prev_fill,
    };
    let rate = config.rate;

    let mut output = Vec::with_capacity(input.len());

    let mut prev_time = input[0].time_offset;
    let mut prev_sample = input[0];

    for &sample in input {
        let dt = sample.time_offset - prev_time;
        if dt > rate {
            let num_fake_samples = (dt / rate).floor() as usize;
            for i in 0..num_fake_samples {
                let t = prev_time + i as f32 * rate;
                output.push(fill_mode(t, prev_sample, sample));
            }
        }
        output.push(sample);
        prev_time = sample.time_offset;
        prev_sample = sample;
    }

    output
}

fn apply_end_padding(data: &mut Vec<BedExitRecord>, config: &EndPadding) {
    let end_padding_count = (config.amount / config.rate) as usize;
    let iter = (0..end_padding_count).map(|i| i as f32 * config.rate);

    let last_sample = *data.last().unwrap();

    match config.mode {
        HoleFillingMode::None => {}
        HoleFillingMode::ZeroFill => {
            data.extend(iter.map(|t| zero_fill(t, last_sample, last_sample)))
        }
        HoleFillingMode::PrevFill => {
            data.extend(iter.map(|t| prev_fill(t, last_sample, last_sample)))
        }
    }
}

fn count_bed_exit_events(mut labels: impl Iterator<Item = u8>) -> usize {
    let mut prev_label = labels.next().unwrap_or(0);
    let mut count = 0;

    for label in labels {
        if prev_label == 0 && label == 1 {
            count += 1;
        }
        prev_label = label;
    }

    count
}
