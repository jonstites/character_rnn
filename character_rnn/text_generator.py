#!/usr/bin/env python3

from character_rnn import utils
import argh
import json
import numpy as np
import random
import tensorflow as tf


def build_graph(rnn_size=250, alpha_size=256, num_layers=1):
    g = tf.Graph()
    with g.as_default():
        # batch, seq, alpha        
        char_integers = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        sequence = tf.one_hot(char_integers, alpha_size)    
        label_integers= tf.placeholder(tf.int32, shape=[None, None], name="labels")

        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(num_units=rnn_size) for _ in range(num_layers)])

        outputs, states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=sequence)

        logits = tf.contrib.layers.linear(outputs, alpha_size)
        _ = tf.nn.softmax(logits, name="logits")
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_integers))
        _ = tf.identity(loss, name="loss")
        optimizer = tf.train.AdamOptimizer().minimize(loss, name="optimizer")
        return g

def sample(sess, sample_length, vocabulary, start):
    sample_start_char = start
    sample_sequence = [sample_start_char]
    for _ in range(sample_length):
        feed = {"inputs:0":[sample_sequence]}
        logits = sess.run("logits:0", feed_dict=feed)
        next_char = np.random.choice(len(logits[0][-1]), p=logits[0][-1])
        sample_sequence.append(next_char)
    return ''.join([vocabulary[s] for s in sample_sequence])
    
def main(train_text, sequence_length=100, batch_size=32, rnn_size=128, num_layers=2):
    
    tf.reset_default_graph()

    dataset = utils.Dataset()
    dataset.add_file(train_text)
    dataset.preprocess()
    
    # Create input data
    alpha_size = dataset.get_vocabulary_size()
    graph = build_graph(rnn_size=rnn_size, alpha_size=alpha_size, num_layers=num_layers)
    
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        train_batch_generator = utils.batches(dataset.train_batches, dataset.train_sources)
        val_batch_generator = utils.batches(dataset.validation_batches, dataset.validation_sources)        

        for iteration in range(10000):

            batch_inputs, batch_labels, authors = next(train_batch_generator)
            feed = {"inputs:0":batch_inputs, "labels:0":batch_labels}
            loss, _ = sess.run(["loss:0", "optimizer"], feed_dict=feed)

            if iteration % 100 == 0:
                print(loss, flush=True)

            if iteration % 1000 == 0:
                print("iteration: {0}".format(iteration))
                print(sample(sess, 500, dataset.inverse_vocabulary, start=dataset.vocabulary[""]))

                val_loss = []
                for it in range(1000):
                    batch_inputs, batch_labels, authors = next(val_batch_generator)
                    feed = {"inputs:0":batch_inputs, "labels:0":batch_labels}
                    loss = sess.run(["loss:0"], feed_dict=feed)
                    val_loss.append(loss)
                print("validation loss: ", np.mean(val_loss))


    print("loss:", loss)
    print(sample(sess, 500, dataset.inverse_vocabulary, start=dataset.vocabulary[""]))    
    avg_loss = []
        
            
                    
if __name__ == "__main__":
    argh.dispatch_command(main)
