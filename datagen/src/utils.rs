use base64;
use byteorder::{ByteOrder, LittleEndian};
use serde::{Serialize, Deserialize, Deserializer, Serializer};

/// Flattened tensor data with shape information suitable for IPC.
#[derive(Serialize, Deserialize)]
pub struct FlatTensor {
    #[serde(serialize_with = "f32_as_base64", deserialize_with = "f32_from_base64")]
    pub data: Vec<f32>,
    pub shape: Vec<u64>,
}

/// A generic helper method for base64 deserialization of a f32 array
pub fn f32_from_base64<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;

    let bytes = String::deserialize(deserializer)
        .and_then(|string| base64::decode(&string).map_err(|err| Error::custom(err.to_string())))?;

    let mut result = vec![0.0; bytes.len() / 4];
    LittleEndian::read_f32_into(&bytes, &mut result);
    Ok(result)
}

/// A generic helper method for base64 serialization of a f32 array
pub fn f32_as_base64<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
where
    T: AsRef<[f32]>,
    S: Serializer,
{
    let mut buffer = vec![0; 4 * value.as_ref().len()];
    LittleEndian::write_f32_into(value.as_ref(), &mut buffer);
    serializer.serialize_str(&base64::encode(&buffer))
}

/// Debug function for outputing batch data in CSV format
#[allow(dead_code)]
pub fn gen_csv(batch: &crate::types::Batch) {
    let x = &batch["x"];
    let y = &batch["y"];

    let mut writer = csv::Writer::from_writer(std::fs::File::create("test.csv").unwrap());

    writer
        .write_record({
            let mut header = Vec::with_capacity(x.shape[2] as usize + 4);
            header.push("window".into());
            header.push("sample".into());
            header.extend((0..x.shape[2]).map(|i| format!("x{}", i)));
            header.push("y".into());
            header
        })
        .unwrap();

    for window in 0..y.shape[0] {
        for sample in 0..y.shape[1] {
            let i = (window * y.shape[1] + sample) as usize;
            let attributes = &x.data[i * x.shape[2] as usize..(i + 1) * x.shape[2] as usize];
            writer
                .serialize((window, sample, attributes, y.data[i]))
                .unwrap();
        }
    }
}
