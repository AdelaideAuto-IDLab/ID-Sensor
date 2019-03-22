import tensorflow as tf

# pylint:disable=no-name-in-module
from tensorflow.python.tools.freeze_graph import freeze_graph

from .fcn import FcnModel, FcnTrainingWrapper
from .cnn_lstm import CnnLstmModel, CnnLstmTrainingWrapper
from training import Trainer


def create_model(config, training=True):
    if config["model"]["type"] == "fcn":
        return FcnModel(config, training)
    elif config["model"]["type"] == "cnn_lstm":
        return CnnLstmModel(config, training)
    else:
        return None


def create_trainable_model(config):
    if config["model"]["type"] == "fcn":
        return FcnTrainingWrapper(FcnModel(config, True))
    elif config["model"]["type"] == "cnn_lstm":
        return CnnLstmTrainingWrapper(CnnLstmModel(config, True))
    else:
        return None


def save_model(config, input_graph=None, checkpoint_dir=None):
    _sess, model = restore_model(config, checkpoint_dir)

    if input_graph == None:
        trainer = Trainer(None, None, None, config["trainer"])
        tf.train.write_graph(model.graph, trainer.log_dir(), "final.pb", False)
        input_graph = trainer.log_dir() + "/final.pb"

    if checkpoint_dir == None:
        checkpoint_dir = trainer.checkpoint_dir()

    freeze_graph(
        input_graph=input_graph,
        input_checkpoint=checkpoint_dir + "/best",
        output_graph=trainer.log_dir() + "/final_frozen.pb",
        output_node_names=model.output_node_names,
        input_binary=True,
        input_saver="",
        restore_op_name="save/restore_all",
        filename_tensor_name="save/Const:0",
        clear_devices=True,
        initializer_nodes="",
        variable_names_blacklist="",
    )


def restore_model(config, checkpoint_dir=None, checkpoint_file=None):
    if checkpoint_dir == None:
        checkpoint_dir = Trainer(None, None, None, config["trainer"]).checkpoint_dir()
    if checkpoint_file == None:
        checkpoint_file = checkpoint_dir + "/best"
    model = create_model(config, training=False)
    print("checkpoint_file", checkpoint_file)

    config = tf.ConfigProto()
    # pylint:disable=no-member
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=model.graph, config=config)

    with model.graph.as_default():  # pylint:disable=not-context-manager
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, checkpoint_file)

    return sess, model
