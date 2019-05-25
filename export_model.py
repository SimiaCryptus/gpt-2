#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model

def export_model(
    model_name='345M',
    seed=None
):
    """
    Load the model and save it as a single archive
    """
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        lm_output = model.model(
            hparams=hparams,
            X=tf.placeholder(tf.int32, shape=(1, None), name="input_X"),
            reuse=tf.AUTO_REUSE
        )
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(os.path.join('models', model_name)))
        with tf.name_scope("output") as scope:
            logits = lm_output['logits'][:, :, :hparams.n_vocab][:, -1, :]
            presents = lm_output['present']
            presents_shape = model.past_shape(hparams=hparams, batch_size=None)
            presents.set_shape(presents_shape)
            #presents = tf.concat([None, presents], axis=-2)
            tf.io.write_graph(
                tf.graph_util.convert_variables_to_constants(
                    sess,
                    sess.graph_def,
                    [
                        logits.op.name,
                        presents.op.name
                    ],
                    None,
                    None
                ),
                "models",
                model_name + "_Init.pb",
                as_text=False
            )
    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        past = tf.placeholder(tf.float32, shape=(1, 24, 2, 16, None, 64), name="input_past")
        lm_output = model.model(
            hparams=hparams,
            X=tf.placeholder(tf.int32, shape=(1, None), name="input_X"),
            past=past,
            reuse=tf.AUTO_REUSE
        )
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(os.path.join('models', model_name)))
        with tf.name_scope("output") as scope:
            logits = lm_output['logits'][:, :, :hparams.n_vocab][:, -1, :]
            presents = lm_output['present']
            presents_shape = model.past_shape(hparams=hparams, batch_size=None)
            presents.set_shape(presents_shape)
            presents = tf.concat([past, presents], axis=-2)
            tf.io.write_graph(
                tf.graph_util.convert_variables_to_constants(
                    sess,
                    sess.graph_def,
                    [
                        logits.op.name,
                        presents.op.name
                    ],
                    None,
                    None
                ),
                "models",
                model_name + ".pb",
                as_text=False
            )


if __name__ == '__main__':
    fire.Fire(export_model)

