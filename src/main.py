import argparse, json, uuid, copy, datetime, os
from pathlib import Path
import tensorflow as tf

import models, evaluate

from training import Trainer, train_model
from data import DataSet, TestSet, Api

PATIENT_LIST = set(range(0, 23))
API_ENDPOINT = os.environ.get("API_ENDPOINT", "http://localhost:8000")


def init_for(config):
    id = config.get("id")
    if id == None:
        id = "{:%d%b%Y_%H%M%S}".format(datetime.datetime.now())
        config["id"] = id
        config["is_existing"] = False
    else:
        config["is_existing"] = True

    group = config.get("group", "default")
    log_dir = (
        config["trainer"].get("log_dir", ".log/{group}/{id}").format(group=group, id=id)
    )
    config["trainer"]["log_dir"] = log_dir
    config["num_attributes"] = num_attributes(config)

    return config


def num_attributes(config):
    if config["train_data"].get("one_hot_metadata"):
        meta = config["train_data"]["one_hot_metadata"]

        num_attributes = 2
        num_attributes += meta["num_antennas"]
        num_attributes += meta["num_frequencies"]
        if config["train_data"]["use_sensor_tag"]:
            num_attributes += meta["num_tags"]
        if config["train_data"]["use_extra_features"]:
            num_attributes += 5
        if (
            config["train_data"]["use_extra_features"]
            and config["train_data"]["use_sensor_tag"]
        ):
            num_attributes += 1
        return num_attributes

    num_attributes = 3
    if config["train_data"]["use_sensor_tag"]:
        num_attributes += 1
    if config["train_data"]["use_extra_features"]:
        num_attributes += 5
    if (
        config["train_data"]["use_extra_features"]
        and config["train_data"]["use_sensor_tag"]
    ):
        num_attributes += 1
    return num_attributes


def save_config(config):
    print(json.dumps(config))

    log_dir = config["trainer"]["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        with open(log_dir + "/config.json", mode="w") as file:
            json.dump(config, file)


def configure_data_filters(config, test_patients):
    train_patients = list(PATIENT_LIST - set(test_patients))

    config["train_data"]["patient_filter"] = train_patients
    config["test_data"]["patient_filter"] = test_patients


def get_full_train_data(config):
    train_data_config = copy.deepcopy(config["test_data"])
    train_data_config["patient_filter"] = config["train_data"]["patient_filter"]
    return TestSet(train_data_config, Api(API_ENDPOINT))


def main(args):
    config = init_for(args["config"])
    configure_data_filters(config, [0])

    save_config(config)

    test_data = TestSet(config["test_data"], Api(API_ENDPOINT))
    full_train_data = get_full_train_data(config)

    if config["is_existing"] == False:
        with DataSet(config["train_data"], Api(API_ENDPOINT)) as train_data:
            train(config, train_data, full_train_data, test_data)
        models.save_model(config)

    test(config, test_data)


def train_and_eval_network(config, args):
    test_data = TestSet(config["test_data"], Api(API_ENDPOINT))
    full_train_data = get_full_train_data(config)

    if config["is_existing"] == False:
        with DataSet(config["train_data"], Api(API_ENDPOINT)) as train_data:
            train(config, train_data, full_train_data, test_data)
        models.save_model(config)

    # Find the best threshold for the training data
    train_data_config = copy.deepcopy(config["test_data"])
    train_data_config["patient_filter"] = config["train_data"]["patient_filter"]
    train_data = TestSet(train_data_config, Api(API_ENDPOINT))
    config["alert_threshold"], config["alert_threshold_low"] = find_best_threshold(
        config, train_data
    )

    save_config(config)

    result = test(config, test_data)
    result["default"] = result["alarm_fscore"]
    result["threshold"] = config["alert_threshold"]
    result["threshold_low"] = config["alert_threshold_low"]

    result["tag"] = args["tag"]
    if args["results_path"] is not None:
        with open(args["results_path"], "a") as file:
            json.dump(result, file)
            file.write("\n")

    return result

def cross_validation(args):
    from models.lr import cross_validation_lr

    for test_patient in PATIENT_LIST:
        config = init_for(copy.deepcopy(args["config"]))
        config["name"] = f"{test_patient}"
        configure_data_filters(config, [test_patient])

        save_config(config)

        if config["model"]["type"] == "lr":
            cross_validation_lr(config, args.get("results_path"), args.get("tag"))
        else:
            train_and_eval_network(config, args)


def train(config, train_data, full_train_data, test_data):
    model = models.create_trainable_model(config)
    trainer = Trainer(train_data, test_data, full_train_data, config["trainer"])
    train_model(model, trainer)


def test(config, test_data):
    sess, model = models.restore_model(config)
    result = evaluate.test_model(sess, model, config, test_data)

    print(json.dumps(result))
    with open(config["trainer"]["log_dir"] + "/result.json", mode="w") as file:
        json.dump(result, file)

    return result


def find_best_threshold(config, train_data):
    sess, model = models.restore_model(config)
    return evaluate.find_best_threshold(sess, model, config, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    train_parser = subparsers.add_parser("train")
    cross_parser = subparsers.add_parser("cross")

    def add_train_args(parser):
        parser.add_argument("config")
        parser.add_argument("--results-path")
        parser.add_argument("--tag")

    for p in (train_parser, cross_parser):
        add_train_args(p)

    def get_train_args(args):
        common = {
            "config": args.config,
            "results_path": args.results_path,
            "tag": args.tag,
        }
        with Path(common["config"]).open() as file:
            common["config"] = json.loads(file.read())
        return common

    args = parser.parse_args()
    if args.mode == "train":
        main(get_train_args(args))
    elif args.mode == "cross":
        cross_validation(get_train_args(args))
    else:
        main(args)
