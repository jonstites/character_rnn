#!/usr/bin/env python3

import argh
import tensorflow
from character_rnn import utils
import math
import random
import numpy as np
import tensorflow as tf

def build_graph():
    pass

def get_batch(values, batch_size=100, window_size=1):
    batch_targets = []
    batch_labels = []
    for _ in range(batch_size):
        target_char = random.randint(window_size, len(values) - window_size)
        context = [char for char in values[target_char-window_size:target_char]]
        context += [char for char in values[target_char+1:target_char+window_size]]
        for context_char in context:
            batch_targets.append(values[target_char])
            batch_labels.append(context_char)
    return np.reshape(batch_targets, [100]), np.reshape(batch_labels, [100,1])

def main(train_file, val_file):
    train = utils.Text.load_files([train_file])
    val = utils.Text.load_files([val_file])    
    texts, ids = utils.Text.to_ids([train[0], val[0]])
    train, val = texts


    vocabulary_size = len(ids.keys())
    embedding_size = 10
    batch_size = 100
    num_sampled=5
    
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

    
    iterations = 1000000
    session = tf.Session()
    session.run(tf.global_variables_initializer())    
    for iteration in range(iterations):
        inputs, labels = get_batch(train)

        feed_dict = {train_inputs: inputs, train_labels: labels}
        _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)
        if iteration % 1000 == 0:
            print(iteration, cur_loss)

if __name__ == "__main__":
    argh.dispatch_command(main)
