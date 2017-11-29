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

    alpha_size = 256

    # batch, seq, alpha
    pre_X = tf.placeholder(tf.int32, shape=[None, None])
    X = tf.one_hot(pre_X, alpha_size)

    X_lengths = tf.placeholder(tf.int32, shape=[None])
    pre_labels = tf.placeholder(tf.int32, shape=[None, None])
    labels = tf.one_hot(pre_labels, alpha_size)

    
    cell = tf.nn.rnn_cell.LSTMCell(num_units=alpha_size)

    outputs, states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float32,
        sequence_length=X_lengths,
        inputs=X)
    print(outputs)
    outputs_flat = tf.reshape(outputs, [-1, alpha_size])
    Ylogits = tf.contrib.layers.linear(outputs_flat, alpha_size)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    logits = Ylogits
    Yflat_ = tf.reshape(labels, [-1, alpha_size])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    print("yflat", Yflat_)
    print("logits", logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_))  # [ BATCHSIZE x SEQLEN ]
    
    #logits = tf.contrib.layers.fully_connected(outputs, alpha_size)
    print(logits)
    #loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=tf.contrib.layers.flatten(logits), labels=tf.contrib.layers.flatten(labels)))

    optimizer = tf.train.AdamOptimizer().minimize(loss)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(10000):
        for comment in get_sequences(train_records):
            characters = [ord(character) for character in comment]
            seq_length = len(characters) 


            x = np.reshape(characters, (batch_size, seq_length))
            x2 = x[:,:-1]
            y = x[:,1:]
            the_loss, _, the_logits, the_labels, the_outputs = sess.run([loss, optimizer, logits, labels, Yflat_], feed_dict={pre_X:x2, X_lengths:[seq_length-1], pre_labels:y})

            if epoch % 10 == 0:
                print("epoch:", epoch)
                print("loss:", the_loss)
                print("".join([chr(char) for char in characters[1:]]))
                print("".join([chr(np.argmax(the_logits[i])) for i in range(seq_length-1)]), flush=True)


if __name__ == "__main__":
    argh.dispatch_command(main)
