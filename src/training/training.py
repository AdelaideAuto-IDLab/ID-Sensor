import time, json
import tensorflow as tf
import numpy as np

from data import DataSet, TestSet, combine_prediction_overlap
from training import Scheduler

from evaluate import compare_alarms, alarm_summary, alert_class, alert_class2, compute_alarm_metrics


class TrainableModel(object):
    def __init__(self):
        self.graph = None
        self.model = None

        self.x = None
        self.y = None
        self.expected_y = None
        self.is_training = None

        self.l2_loss_factor = None
        self.loss = None
        self.accuracy = None

        self.trainer = None
        self.training_rate = None

        self.train_summary = None
        self.test_summary = None

    def add_summaries(self):
        self.train_summary = tf.summary.merge(
            [
                tf.summary.scalar("train/loss", self.loss),
                tf.summary.scalar("train/accuracy", self.accuracy),
            ]
        )

        self.test_summary = tf.summary.merge(
            [
                tf.summary.scalar("test/loss", self.loss),
                tf.summary.scalar("test/accuracy", self.accuracy),
            ]
        )

    def unwrap(self):
        return self.model

    def num_classes(self):
        return self.model.config.num_classes()

    def train_step(self, sess: tf.Session, data, step_config):
        if self.num_classes() == 2:
            y = data["y"]
        else:
            y = data["label"]

        feed_dict = {
            self.x: data["x"],
            self.expected_y: y,
            self.is_training: True,
            self.training_rate: step_config["training_rate"],
            self.l2_loss_factor: step_config["l2_loss"],
        }
        fetch_dict = {"trainer": self.trainer, "summary": self.train_summary}
        return sess.run(fetch_dict, feed_dict)

    def test(self, sess: tf.Session, data: TestSet):
        feed_dict = {
            self.x: data.input,
            self.expected_y: data.label,
            self.is_training: False,
        }
        fetch_dict = {
            "accuracy": self.accuracy,
            "summary": self.test_summary,
            "y": self.y,
        }
        return sess.run(fetch_dict, feed_dict)


## TODO: Switch to a builtin tensorflow trainer
class Trainer(object):
    def __init__(
        self,
        train_data: DataSet,
        full_train_data: TestSet,
        test_data: TestSet,
        trainer_config,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.full_train_data = full_train_data

        self.config = trainer_config
        self.scheduler = Scheduler()

        self.summary_writer = None

    def start(self):
        self.scheduler.start(self.train_data, self.config["schedule"])

    def train_step(self, sess: tf.Session, model: TrainableModel):
        data = self.train_data.next_batch()
        result = model.train_step(sess, data, self.scheduler.current_config())
        self.summary_writer.add_summary(result["summary"], self.scheduler.step)
        return result

    def test(self, sess: tf.Session, model: TrainableModel, data: TestSet, tag="test"):
        result = model.test(sess, data)
        if tag == "test":
            self.summary_writer.add_summary(result["summary"], self.scheduler.step)

        y_pred = combine_prediction_overlap(
            result["y"], data.window_size, data.window_overlap
        )

        t = data.t
        if model.num_classes() == 2:
            y_true = data.combine_scalar_overlap(data.expected_output)
        else:
            y_true = data.combine_scalar_overlap(data.label)

        pred_class = alert_class(y_pred, num_classes=model.num_classes())
        alarms = compare_alarms(
            t, pred_class, alert_class2(y_true, model.num_classes())
        )
        self.summary_writer.add_summary(alarm_summary(alarms, tag), self.scheduler.step)

        result["alarms"] = alarms
        return result

    def log_dir(self):
        return self.config.get("log_dir", "./log")

    def checkpoint_dir(self):
        return self.log_dir() + "/checkpoints"

    def progress(self):
        return (self.scheduler.current_epoch(), self.scheduler.step)

    def current_step(self):
        return self.scheduler.step

    def should_generate_summary(self):
        summary_rate = self.config.get("summary_rate", self.train_data.steps_per_epoch)
        return self.scheduler.step % summary_rate == 0


def train_model(model: TrainableModel, trainer: Trainer):
    config = tf.ConfigProto()
    # pylint:disable=no-member
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=model.graph, config=config)

    current_best = 0.0
    with model.graph.as_default():
        trainer.summary_writer = tf.summary.FileWriter(trainer.log_dir(), model.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    # Start the trainer and generate an initial summary
    trainer.start()
    new_summary(sess, trainer, model)

    # Loop until we are finished training
    while trainer.scheduler.next_round():
        trainer.train_step(sess, model)

        if trainer.should_generate_summary():
            results = trainer.test(sess, model, trainer.full_train_data, "train")
            alarm_metrics = compute_alarm_metrics(results["alarms"])
            if alarm_metrics["alarm_fscore"] > current_best:
                new_summary(sess, trainer, model)
                checkpoint_path = trainer.checkpoint_dir() + "/best"
                model.saver.save(sess, checkpoint_path)

            new_summary(sess, trainer, model)
            checkpoint_path = trainer.checkpoint_dir() + "/model"
            model.saver.save(sess, checkpoint_path, global_step=trainer.current_step())

    # Generate a final checkpoint and summary
    model.saver.save(sess, trainer.checkpoint_dir() + "/final")
    new_summary(sess, trainer, model)

    trainer.summary_writer.close()

    return model


def new_summary(sess: tf.Session, trainer: Trainer, model: TrainableModel):
    (epoch, step) = trainer.progress()

    start_time = time.clock()
    results = trainer.test(sess, model, trainer.test_data)
    generate_intermediate_results(results)
    elapsed_time = time.clock() - start_time

    print(
        f"epoch: {epoch} ({step}), acc: {results['accuracy']}, summary_time: {elapsed_time}"
    )
    print(json.dumps(results["alarms"]))


def generate_intermediate_results(data):
    alarm_precision = 0
    if data["alarms"]["correct"] != 0:
        alarm_precision = data["alarms"]["correct"] / (
            data["alarms"]["correct"] + data["alarms"]["false"]
        )
    alarm_recall = data["alarms"]["correct"] / (
        data["alarms"]["correct"] + data["alarms"]["missed"]
    )

    alarm_fscore = 0
    if alarm_precision + alarm_recall > 0:
        alarm_fscore = (
            2 * (alarm_precision * alarm_recall) / (alarm_precision + alarm_recall)
        )

    results = {
        "acc": data["accuracy"],
        "alarm_precision": alarm_precision,
        "alarm_recall": alarm_recall,
        "alarm_fscore": alarm_fscore,
    }
    results["default"] = results["alarm_fscore"]
