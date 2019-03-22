import json, csv
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from main import API_ENDPOINT
from data.dataset import TestSet, Api
from evaluate import compare_alarms, compute_alarm_metrics, TWO_CLASS


class LogisticRegressionClassifier:
    def __init__(self, config):
        self.model = LogisticRegression(C=0.25)
        self.config = config

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, features):
        return self.model.predict_proba(features)

    def decision_function(self, features):
        return self.model.decision_function(features)


def evaluate(config, model: LogisticRegressionClassifier, test_data: TestSet):
    t = np.reshape(test_data.t, [-1])
    y_true = np.reshape(test_data.expected_output[:, -1], [-1])
    inference = model.predict(
        np.reshape(test_data.input, [test_data.input.shape[0], -1])
    )

    if config["alert_threshold"] > 0.5:
        y_pred = (inference[:, 1] > config["alert_threshold"]).astype(np.int32)
    else:
        y_pred = np.argmax(inference, axis=-1)

    with open(
        config["trainer"]["log_dir"] + "/predictions.csv", "w", newline=""
    ) as output_csv:
        writer = csv.writer(output_csv, delimiter=",")
        writer.writerow(["t", *TWO_CLASS, "y_pred", "y_true"])
        for i in range(0, len(y_pred)):
            writer.writerow(
                [t[i]]
                + list(inference[i])
                + [TWO_CLASS[y_pred[i]], TWO_CLASS[int(y_true[i])]]
            )

    alarm_results = compare_alarms(t, y_pred, y_true)
    alarm_metrics = compute_alarm_metrics(alarm_results)

    return {
        "name": config.get("name"),
        "id": config["id"],
        "group": config["group"],
        "acc": sklearn.metrics.accuracy_score(y_true, y_pred),
        **alarm_metrics,
        **alarm_results,
    }


def cross_validation_lr(config, results_path, tag):
    train_data = TestSet(config["train_data"], Api(API_ENDPOINT))
    test_data = TestSet(config["test_data"], Api(API_ENDPOINT))

    model = LogisticRegressionClassifier(config)
    model.train(
        np.reshape(train_data.input, [train_data.input.shape[0], -1]),
        train_data.expected_output[:, -1],
    )

    result = evaluate(config, model, test_data)

    print(json.dumps(result))
    with open(config["trainer"]["log_dir"] + "/result.json", mode="w") as file:
        json.dump(result, file)

    result["tag"] = tag

    if results_path is not None:
        with open(results_path, "a") as file:
            json.dump(result, file)
            file.write("\n")
