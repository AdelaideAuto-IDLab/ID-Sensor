import tensorflow as tf
import numpy as np

from training import TrainableModel


class ModelConfig(object):
    def __init__(self, config):
        self.config = config

    def num_attributes(self):
        return self.config["num_attributes"]

    def conv_layers(self):
        return self.config["model"]["conv_layers"]

    def filters(self, depth):
        filters_config = self.config["model"]["filters"]
        if isinstance(filters_config, list):
            return filters_config[depth]
        return filters_config

    def kernel_size(self):
        return self.config["model"]["kernel_size"]

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
        return self.config["model"]["classes"]


def dense_prediction_readout(input, num_classes):
    # As a result of applying this operation, the inner 2D tensor (consisting of different sensor
    # readings over a window of time) is squashed to a 1D tensor where there is a single value for
    # each moment in time.
    #
    # [batch_size, window_size, 1, num_filters] -> [batch_size, window_size, num_classes]
    layer = tf.layers.conv2d(
        input, num_classes, kernel_size=[1, 1], strides=[1, 1], name="1xD_conv"
    )
    return tf.squeeze(layer, [2])


class FcnModel(object):
    def __init__(self, config, training=True):
        self.config = ModelConfig(config)
        self.graph = tf.Graph()
        self.training = training
        self.output_node_names = "y"

        with self.graph.as_default():  # pylint:disable=not-context-manager
            # A tensor with shape [batch_size, window_size, num_attributes]
            self.x = tf.placeholder(
                tf.float32, shape=[None, None, self.config.num_attributes()], name="x"
            )
            self.features = tf.transpose(tf.expand_dims(self.x, -1), [0, 1, 3, 2])

            if self.training:
                self.is_training = tf.placeholder_with_default(
                    tf.constant(False), shape=[], name="is_training"
                )

            network = self.hidden_layers()

            if self.training:
                network = tf.layers.dropout(
                    network, self.config.dropout_rate(), training=self.is_training
                )

            self.logits = dense_prediction_readout(network, self.config.num_classes())
            self.y = tf.nn.softmax(self.logits, name="y")

    def inference(self, sess: tf.Session, x):
        feed_dict = {self.x: x}
        return sess.run(self.y, feed_dict)

    def feature_space(self, sess: tf.Session, x):
        feed_dict = {self.x: x}
        return sess.run(self.conv_layers[-1], feed_dict)

    def hidden_layers(self):
        def conv(input, depth):
            return tf.layers.conv2d(
                input,
                self.config.filters(depth),
                kernel_size=self.config.kernel_size(),
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

        self.conv_layers = [self.features]

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

        return self.conv_layers[-1]


class FcnTrainingWrapper(TrainableModel):
    def __init__(self, model: FcnModel):
        super().__init__()

        self.graph = model.graph
        self.model = model

        self.x = self.model.x
        self.y = self.model.y

        self.is_training = self.model.is_training

        with self.graph.as_default():  # pylint:disable=not-context-manager
            self.saver = tf.train.Saver(tf.global_variables())

            # A tensor with shape [batch_size, window_size]
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
        classes = self.model.config.num_classes()
        labels = tf.one_hot(self.expected_y, depth=classes)

        # Reshape the tensors to merge the windows and mini-batch
        logits = tf.reshape(self.model.logits, [-1, classes])
        labels = tf.reshape(labels, [-1, classes])

        cross_entropy = tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=logits
        )

        l2_loss = tf.reduce_sum(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
        )

        return cross_entropy + self.l2_loss_factor * l2_loss

    def get_evaluation(self):
        # The inference is correct if the category with the maximum value
        # matches the label
        correct = tf.equal(
            tf.argmax(self.model.y, 2), tf.cast(self.expected_y, tf.int64)
        )

        # Evaluate the accuracy over the batch and time domain
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    def get_trainer(self):
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            optimizer = tf.train.AdamOptimizer(self.training_rate)
            return optimizer.minimize(self.loss)

