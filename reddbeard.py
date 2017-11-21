#!/usr/bin/env python3

import argh
import bz2
import json
import numpy as np
import tensorflow as tf


def get_sequences(records):
    with bz2.BZ2File(records, "rb") as dataset:
        for line in dataset:
            data = json.loads(line)
            yield data["body"]

def main(train_records):

    
    tf.reset_default_graph()

    # Create input data
    batch_size=1

    alpha_size = 1

    X = tf.placeholder(tf.float32, shape=[None, None, alpha_size])
    X_lengths = tf.placeholder(tf.int32, shape=[None])

    cell = tf.nn.rnn_cell.LSTMCell(num_units=4, state_is_tuple=True)

    outputs, last_states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float32,
        sequence_length=X_lengths,
        inputs=X)


    for comment in get_sequences(train_records):
        characters = [float(ord(character)) for character in comment]
        seq_length = len(characters)        
        x = np.reshape(characters, (batch_size, seq_length, alpha_size))
        print("x:", x)
        outputs = tf.contrib.learn.run_n(
            {"outputs": outputs, "last_states": last_states},
            n=1,
            feed_dict={X: x, X_lengths:[seq_length]})
        print(outputs)
        break


if __name__ == "__main__":
    argh.dispatch_command(main)
