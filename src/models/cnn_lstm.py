import tensorflow as tf
import numpy as np

from training import TrainableModel


class ModelConfig(object):
    def __init__(self, config):
        self.config = config

    def num_attributes(self):
        return self.config["num_attributes"]

    def conv_window_size(self):
        return self.config["model"]["conv_window_size"]

    def conv_layers(self):
        return self.config["model"]["conv_layers"]

    def conv_filters(self, depth):
        filters_config = self.config["model"]["conv_filters"]
        if isinstance(filters_config, list):
            return filters_config[depth]
        return filters_config

    def kernel_size(self):
        return self.config["model"]["kernel_size"]

    def kernel_stride(self):
        return self.config["model"]["kernel_stride"]

    def activation(self):
        if self.config["model"]["activation"] == "relu":
            return tf.nn.relu
        if self.config["model"]["activation"] == "tanh":
            return tf.nn.tanh
        if self.config["model"]["activation"] == "elu":
            return tf.nn.elu
        if self.config["model"]["activation"] == "leaky_relu":
            return tf.nn.leaky_relu
        return None

    def lstm_layers(self):
        return self.config["model"]["lstm_layers"]

    def lstm_units(self):
        return self.config["model"]["lstm_units"]

    def lstm_steps(self):
        return self.config["model"]["lstm_steps"]

    def use_batch_norm(self):
        if self.config["model"].get("use_batch_norm") == True:
            return True
        return False

    def use_pooling(self):
        if self.config["model"].get("use_pooling") == True:
            return True
        return False

    def dropout_rate(self):
        return self.config["model"]["dropout_rate"]

    def num_classes(self):
        return 2


def lstm_cell(config, training):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=config.lstm_units())
    if training:
        keep_prob = 1.0 - config.dropout_rate()
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob
        )
    return cell


class CnnLstmModel(object):
    def __init__(self, config, training=True):
        if training == False:
            config["model"]["lstm_steps"] = 1

        self.config = ModelConfig(config)
        self.graph = tf.Graph()
        self.training = training
        self.output_node_names = "y,state"
        self.current_state = None

        with self.graph.as_default():  # pylint:disable=not-context-manager
            # A tensor with shape [batch size * lstm steps, conv window size, 1, num attributes]
            self.x = tf.placeholder(
                tf.float32,
                shape=[
                    None if training else 1,
                    self.config.conv_window_size(),
                    1,
                    self.config.num_attributes(),
                ],
                name="x",
            )

            if self.training:
                self.is_training = tf.placeholder_with_default(
                    tf.constant(False), shape=[], name="is_training"
                )

            self.build_cnn(training)
            self.build_lstm(training)

            logits = tf.layers.dense(self.output, self.config.num_classes())
            self.logits = tf.reshape(
                logits, [-1, self.config.lstm_steps(), self.config.num_classes()]
            )
            self.y = tf.nn.softmax(self.logits, name="y")

    def step(self, sess: tf.Session, x):
        feed_dict = {self.x: x}
        if self.current_state is not None:
            feed_dict[self.initial_state] = self.current_state

        result = sess.run({"y": self.y, "state": self.final_state}, feed_dict)
        self.current_state = result["state"]
        return result["y"]

    def inference(self, sess: tf.Session, x):
        results = []
        for batch in range(0, len(x)):
            results.append(self.step(sess, x[batch : batch + 1]))

            if (100 * batch) % len(x) == 0:
                print(f"{100 * batch / len(x)}%")

        return np.concatenate(results)

    def step_features(self, sess: tf.Session, x):
        feed_dict = {self.x: x}
        if self.current_state is not None:
            feed_dict[self.initial_state] = self.current_state

        result = sess.run({"output": self.output, "state": self.final_state}, feed_dict)
        self.current_state = result["state"]
        return result["output"]

    def feature_space(self, sess: tf.Session, x):
        results = []
        for batch in range(0, len(x)):
            results.append(self.step_features(sess, x[batch : batch + 1]))

            if (100 * batch) % len(x) == 0:
                print(f"{100 * batch / len(x)}%")

        return np.concatenate(results)

    def build_cnn(self, training):
        def conv(input, depth):
            return tf.layers.conv2d(
                input,
                self.config.conv_filters(depth),
                kernel_size=self.config.kernel_size(),
                strides=self.config.kernel_stride(),
                padding="SAME",
                activation=self.config.activation(),
            )

        def pool(input):
            return tf.layers.max_pooling2d(
                input, pool_size=[2, 1], strides=[1, 1], padding="SAME"
            )

        def batch_norm(input):
            if self.training:
                return tf.layers.batch_normalization(input, training=self.is_training)
            else:
                return tf.layers.batch_normalization(input)

        self.conv_layers = [self.x]

        with tf.name_scope("ConvLayers"):
            for i in range(0, self.config.conv_layers()):
                with tf.variable_scope(f"conv{i}"):
                    layer = self.conv_layers[-1]

                    if self.config.use_batch_norm():
                        layer = batch_norm(layer)

                    layer = conv(layer, i)

                    if self.config.use_pooling():
                        layer = pool(layer)

                    self.conv_layers.append(layer)

        d = tf.reduce_prod(self.conv_layers[-1].shape[1:])
        self.features = tf.reshape(
            self.conv_layers[-1], [-1, self.config.lstm_steps(), d]
        )

    def build_lstm(self, training):
        layers = self.config.lstm_layers()

        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(self.config, training) for _ in range(layers)]
        )

        self.initial_state = stacked_lstm.zero_state(
            tf.shape(self.features)[0], tf.float32
        )

        inputs = tf.unstack(self.features, num=self.config.lstm_steps(), axis=1)
        out, state = tf.nn.static_rnn(
            stacked_lstm, inputs, self.initial_state, dtype=tf.float32
        )

        self.final_state = state

        self.state = tf.identity(state, name="state")  # Need this to allow exporting
        self.output = tf.reshape(tf.concat(out, 1), [-1, self.config.lstm_units()])


class CnnLstmTrainingWrapper(TrainableModel):
    def __init__(self, model: CnnLstmModel):
        super().__init__()

        self.graph = model.graph
        self.model = model

        self.x = self.model.x
        self.y = self.model.y

        with self.graph.as_default():  # pylint:disable=not-context-manager
            self.saver = tf.train.Saver(tf.global_variables())

            self.is_training = tf.placeholder_with_default(
                tf.constant(False), shape=[], name="is_training"
            )

            # A tensor with shape [batch_size, lstm_steps]
            self.expected_y = tf.placeholder(
                tf.int32, shape=[None, None], name="expected_y"
            )

            with tf.name_scope("loss"):
                self.l2_loss_factor = tf.placeholder_with_default(
                    tf.constant(0.0), shape=[]
                )
                self.loss = self.get_loss()

            with tf.name_scope("evaluation"):
                self.accuracy = self.get_evaluation()

            with tf.name_scope("trainer"):
                self.training_rate = tf.placeholder_with_default(
                    tf.constant(0.001), shape=[]
                )
                self.trainer = self.get_trainer()

            super().add_summaries()

    def get_loss(self):
        sequence_loss = tf.contrib.seq2seq.sequence_loss(
            self.model.logits,
            self.expected_y,
            tf.cast(tf.ones_like(self.expected_y), tf.float32),
            average_across_timesteps=True,
            average_across_batch=True,
        )

        l2_loss = tf.reduce_sum(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
        )

        return sequence_loss + self.l2_loss_factor * l2_loss

    def get_evaluation(self):
        y_pred = tf.argmax(self.model.y, -1)
        y_true = tf.cast(self.expected_y, tf.int64)
        correct = tf.equal(y_pred, y_true)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    def get_trainer(self, max_grad_norm=5):
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            trainable_vars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.loss, trainable_vars), max_grad_norm
            )

            optimizer = tf.train.AdamOptimizer(self.training_rate)
            return optimizer.apply_gradients(zip(grads, trainable_vars))

