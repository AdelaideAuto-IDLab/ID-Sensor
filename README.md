# ID-Sensor

_Super Low Resolution RF Powered Accelerometers for Alerting on Hospitalized Patient Bed Exits_

## Abstract

Falls have serious consequences and are prevalent in acute hospitals and nursing homes caring for older people. Most falls occur in bedrooms and near the bed. Technological interventions to mitigate the risk of falling aim to automatically monitor bed-exit events and subsequently alert healthcare personnel to provide timely supervisions. We observe that frequency-domain information related to patient activities exist predominantly in very low frequencies.
Therefore, we recognise the potential to employ a low resolution acceleration sensing modality in contrast to powering and sensing with a conventional MEMS (Micro Electro Mechanical System) accelerometer.
Consequently, we investigate a batteryless sensing modality with low cost wirelessly powered Radio Frequency Identification (RFID) technology with the potential for convenient integration into _clothing_, such as hospital gowns. We design and build a _passive_ accelerometer-based RFID sensor embodiment---_ID-Sensor_---for our study. The sensor design allows deriving ultra low resolution acceleration data  from the rate of change of unique RFID tag identifiers in accordance  with the movement of a patient's upper body. We investigate two convolutional neural network architectures for learning from raw _RFID-only_ data streams and compare performance with a traditional shallow classifier with engineered features. We evaluate performance with 23 hospitalized older patients.
We demonstrate, for the first time and to the best of knowledge, that: i) the low resolution acceleration data embedded in the RF powered _ID-Sensor_ data stream can provide a practicable method for activity recognition; and ii) highly discriminative features can be efficiently learned from the raw RFID-_only_ data stream using a fully convolutional network architecture.

## Training

We provide a data loader and preprocessing tool and docker image for a model. To train a new model, create a new config file (see the `config` directory for more examples) and do the following:

1. Build the data management tool: `cd datagen && cargo build --release`
2. Build the training docker image: `docker build -t idsensor-training ./docker`
3. Run the data management tool: `./datagen/target/release/datagen`
3. Run the docker image:
    ```bash
    docker run \
        --mount='type=bind,src=./src,dst=/src' \
        --mount='type=bind,src=./config,dst=/config' \
        --mount='type=bind,src=./log,dst=/log' \
        --net=host \
        -d \
        -e PYTHONUNBUFFERED=0 -e API_ENDPOINT="http://localhost:8000" \
        idsensor-training \
        python /src/main.py [train or cross] <path/to/config/file>
    ```

## Reference

This repository is provided as part of the following paper:

M. Chesser, A. Jayatilaka, R. Visvanathan, C. Fumeaux, A. Sample and D. C. Ranasinghe. 2019. "Super Low Resolution RF Powered Accelerometers for Alerting on Hospitalized Patient Bed Exits". *IEEE International Conference on Pervasive Computing and Communications (IEEE PerCom 2019). Kyoto, Japan.*

Cite using:

```
@conference {2019,
    title = {Super Low Resolution RF Powered Accelerometers for Alerting on Hospitalized Patient Bed Exits},
    booktitle = {IEEE International Conference on Pervasive Computing and Communications (IEEE PerCom 2019)},
    year = {2019},
    address = {Kyoto, Japan},
    author = {Michael Chesser and Asangi Jayatilaka and Renuka Visvanathan and Christophe Fumeaux and Alanson Sample and Damith C Ranasinghe}
}
```

## License

This project is licensed under the GPL-3.0 License.

See [LICENSE](./LICENSE) for details.
