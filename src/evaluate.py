import csv
import numpy as np
import tensorflow as tf
import sklearn

from operator import itemgetter

from data import combine_prediction_overlap

TWO_CLASS = ["InBed", "OutOfBed"]
FIVE_CLASS = ["SIT_ON_BED", "SIT_ON_CHAIR", "LYING_ON_BED", "WALK", "STAND_UP"]
THREE_CLASS = ["InBed", "OnChair", "Ambulating"]


def get_class_map(num_classes):
    if num_classes == 3:
        return THREE_CLASS
    elif num_classes == 5:
        return FIVE_CLASS

    return TWO_CLASS


def find_best_threshold(sess: tf.Session, model, config, train_data):
    t = train_data.t

    inference = combine_prediction_overlap(
        model.inference(sess, train_data.input),
        train_data.window_size,
        train_data.window_overlap,
    )

    if config["model"]["classes"] == 2:
        y_true = train_data.combine_scalar_overlap(train_data.expected_output)
    else:
        y_true = train_data.combine_scalar_overlap(train_data.label)
    y_true = alert_class2(y_true, config["model"]["classes"])

    best_threshold = 0.5
    best_threshold_low = 0.5
    best_fscore = 0

    for i in range(0, 1):
        for j in range(i, 20):
            threshold_low = 0.5 + 0.5 * i / 20
            threshold = 0.5 + 0.5 * j / 20
            y_pred = alert_class(
                inference,
                num_classes=config["model"]["classes"],
                lower_threshold=threshold_low,
                upper_threshold=threshold,
            )
            results = evaluate_predictions(t, y_true, y_pred, config)
            print(f"threshold={threshold_low},{threshold} got fscore={results['alarm_fscore']}")
            if results["alarm_fscore"] > best_fscore:
                best_fscore = results["alarm_fscore"]
                best_threshold = threshold
                best_threshold_low = threshold_low

    return best_threshold, best_threshold_low


def test_model(sess: tf.Session, model, config, test_data):
    from pathlib import Path

    t = test_data.t

    inference = combine_prediction_overlap(
        model.inference(sess, test_data.input),
        test_data.window_size,
        test_data.window_overlap,
    )
    y_pred = alert_class(
        inference,
        num_classes=config["model"]["classes"],
        lower_threshold=config["alert_threshold_low"],
        upper_threshold=config["alert_threshold"],
    )

    classes = get_class_map(config["model"]["classes"])
    if config["model"]["classes"] == 2:
        y_true = test_data.combine_scalar_overlap(test_data.expected_output)
    else:
        y_true = test_data.combine_scalar_overlap(test_data.label)
    y_true = alert_class2(y_true, config["model"]["classes"])

    with open(
        config["trainer"]["log_dir"] + "/predictions.csv", "w", newline=""
    ) as output_csv:
        writer = csv.writer(output_csv, delimiter=",")
        writer.writerow(["t", *classes, "y_pred", "y_true"])
        for i in range(0, len(y_pred)):
            writer.writerow(
                [t[i]]
                + list(inference[i])
                + [TWO_CLASS[y_pred[i]], TWO_CLASS[int(y_true[i])]]
            )

    results = evaluate_predictions(t, y_true, y_pred, config)

    with open(config["trainer"]["log_dir"] + "/alarms.csv", "w", newline="") as out:
        writer = csv.writer(out, delimiter=",")
        writer.writerow(["time", "type"])
        for row in sorted(results["alarms"], key=itemgetter(0)):
            writer.writerow([row[0], row[1]])

    return results


def evaluate_predictions(t, y_true, y_pred, config):
    alarm_results = compare_alarms(t, y_pred, y_true)
    alarm_metrics = compute_alarm_metrics(alarm_results)

    return {
        "name": config.get("name"),
        "id": config["id"],
        "group": config["group"],
        "acc": sklearn.metrics.accuracy_score(y_true, y_pred),
        "f1_score": sklearn.metrics.f1_score(y_true, y_pred),
        **alarm_metrics,
        **alarm_results,
    }


def compute_alarm_metrics(alarm_results):
    alarm_precision = 0
    if alarm_results["correct"] != 0:
        alarm_precision = alarm_results["correct"] / (
            alarm_results["correct"] + alarm_results["false"]
        )
    alarm_recall = alarm_results["correct"] / (
        alarm_results["correct"] + alarm_results["missed"]
    )

    alarm_fscore = 0
    if alarm_precision + alarm_recall > 0:
        alarm_fscore = (
            2 * (alarm_precision * alarm_recall) / (alarm_precision + alarm_recall)
        )

    return {
        "alarm_precision": alarm_precision,
        "alarm_recall": alarm_recall,
        "alarm_fscore": alarm_fscore,
    }


def alert_class(predictions, num_classes=2, lower_threshold=0.5, upper_threshold=0.5):
    IN_BED = 0
    OUT_OF_BED = 1

    two_class_pred = predictions
    if num_classes == 5:
        in_bed = np.sum(predictions[:, [0, 2]], axis=-1)
        out_of_bed = np.sum(predictions[:, [1, 3, 4]], axis=-1)
        two_class_pred = np.stack([in_bed, out_of_bed], axis=1)
    elif num_classes == 3:
        in_bed = np.sum(predictions[:, [0]], axis=-1)
        out_of_bed = np.sum(predictions[:, [1, 2]], axis=-1)
        two_class_pred = np.stack([in_bed, out_of_bed], axis=1)

    if lower_threshold != upper_threshold:
        current_class = OUT_OF_BED if two_class_pred[0, 1] > upper_threshold else IN_BED
        classes = []
        for row in two_class_pred[:, 1]:
            if current_class == OUT_OF_BED and row <= lower_threshold:
                current_class = IN_BED
            elif current_class == IN_BED and row >= upper_threshold:
                current_class = OUT_OF_BED
            classes.append(current_class)
        return np.array(classes, np.int32)
    elif upper_threshold != 0.5:
        return (two_class_pred[:, 1] > upper_threshold).astype(np.int32)

    return np.argmax(two_class_pred, axis=-1)


def alert_class2(labels, num_classes=2):
    IN_BED = 0
    OUT_OF_BED = 1

    if num_classes == 5:
        class_mapper = np.array([IN_BED, OUT_OF_BED, IN_BED, OUT_OF_BED, OUT_OF_BED])
        return class_mapper[labels.astype(np.int32)]
    elif num_classes == 3:
        class_mapper = np.array([IN_BED, OUT_OF_BED, OUT_OF_BED])
        return class_mapper[labels.astype(np.int32)]

    return labels


def compare_alarms(time, y_pred, y_true, cooldown_period=30, early_gap=30):
    miss_gap = 30

    IN_BED = 0
    OUT_OF_BED = 1

    correct_alarms = 0
    missed_alarms = 0
    false_alarms = 0
    alarm_delays = []
    alarms = []

    prev_predicted_alarm = 0
    prev_true_alarm = 0

    label_changes = np.where(np.roll(y_true, 1) != y_true)[0]

    early_alarm = False

    out_of_bed_event = lambda i: y_pred[i - 1] == IN_BED and y_pred[i] == OUT_OF_BED
    pred_in_cooldown = (
        lambda i: (time[i] - time[prev_predicted_alarm]) < cooldown_period
    )
    true_in_cooldown = lambda i: (time[i] - time[prev_true_alarm]) < cooldown_period

    for i in range(len(label_changes)):
        region_start = label_changes[i]

        if i + 1 == len(label_changes):
            region_end = len(y_true) - 1
        else:
            region_end = label_changes[i + 1]

        if y_true[region_start] == IN_BED:
            # Check for false positives
            for j in range(region_start + 1, region_end):
                time_since_start = time[j] - time[region_start]
                time_until_end = time[region_end] - time[j]

                if not out_of_bed_event(j) or pred_in_cooldown(j):
                    continue

                # Check for early alarm
                if time_until_end < early_gap:
                    if early_alarm != True:
                        alarms.append([time[j].item(), f"ALARM [i={j}, t={time[j]}]"])
                        early_alarm = True
                    prev_predicted_alarm = j
                # Check for a false alarm
                elif time_since_start > early_gap:
                    prev_predicted_alarm = j
                    false_alarms += 1
                    alarms.append([time[j].item(), f"FALSE ALARM [i={j}, t={time[j]}]"])
        else:
            if true_in_cooldown(region_start):
                # Previous alarm also applies to this region
                continue
            prev_true_alarm = region_start

            if early_alarm:
                early_alarm = False
                missed_alarm = False
                alarm_delays += [0.0]
            else:
                missed_alarm = True

            # Check if there was a transition at any point within the out of bed region
            for j in range(region_start + 1, region_end):
                if not pred_in_cooldown(j) and out_of_bed_event(j):
                    if missed_alarm:
                        # Only count the first instance of the alarm
                        missed_alarm = False
                        alarm_delays += [time[j] - time[region_start]]
                        alarms.append([time[j].item(), f"ALARM [i={j}, t={time[j]}]"])
                    prev_predicted_alarm = j

            if missed_alarm:
                missed_alarms += 1
                alarms.append(
                    [
                        time[region_start].item(),
                        f"MISSED_ALARM [i={region_start}, t={time[region_start]}]",
                    ]
                )

    # We currently don't alerts to be "too late" to be considered "missed/late" alerts
    late_alarms = [x for x in alarm_delays if x > miss_gap]
    correct_alarms = [x for x in alarm_delays if x <= miss_gap]

    average_delay = np.mean(correct_alarms).item() if len(correct_alarms) > 0 else 0

    return {
        "total": len(correct_alarms) + missed_alarms + len(late_alarms),
        "correct": len(correct_alarms),
        "average_delay": average_delay,
        "missed": missed_alarms + len(late_alarms),
        "false": false_alarms,
        "alarms": alarms,
        "alarm_delays": [float(x) for x in alarm_delays],
    }


def alarm_summary(alarms, tag="test"):
    summary = tf.Summary()
    # pylint:disable=no-member
    summary.value.add(tag=f"{tag}/alarm/correct", simple_value=alarms["correct"])
    summary.value.add(tag=f"{tag}/alarm/delay", simple_value=alarms["average_delay"])
    summary.value.add(tag=f"{tag}/alarm/missed", simple_value=alarms["missed"])
    summary.value.add(tag=f"{tag}/alarm/false", simple_value=alarms["false"])
    return summary
