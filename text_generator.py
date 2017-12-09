#!/usr/bin/env python3

import argh
import json
import numpy as np
import random
import tensorflow as tf


def get_random_batch(train_text, batch_size=32, sequence_length=50):
    text = load_integer_text(train_text)
    characters_per_batch = batch_size * sequence_length
    for _ in range(0, len(text), characters_per_batch):
        batched_sequences = create_random_batch(text, batch_size, sequence_length)
        yield batched_sequences

def load_integer_text(train_text):
    text = load_text(train_text)
    return ascii_to_integers(text)

def load_text(train_text):
    with open(train_text) as handle:
        text = handle.read()
    return text

def create_random_batch(text, batch_size, sequence_length):
    batch = []
    for _ in range(batch_size):
        sequence = get_random_sequence(text, sequence_length)
        batch.append(sequence)
    return batch

# Biased against the last sequence_length characters. That's okay.
def get_random_sequence(text, sequence_length):
    text_length = len(text)
    assert text_length >= sequence_length + 1
    start = random.randint(0, text_length - sequence_length - 1)
    return text[start: start + sequence_length + 1]
        
def ascii_to_integers(seq):
    return [ord(letter) for letter in seq]

def integers_to_ascii(seq):
    return [chr(letter) for letter in seq]

def batch_to_labels(batch):
    labels = []
    sequences = []
    for seq in batch:
        label, sequence = sequence_to_label(seq)
        labels.append(label)
        sequences.append(sequence)
    return labels, sequences

def sequence_to_label(sequence):
    return sequence[1:], sequence[:-1]

def pad_ascii(seq, sequence_length):
    return seq + [0]*(sequence_length - len(seq))

    

def main(train_text):
    
    tf.reset_default_graph()

    # Create input data
    batch_size=16

    alpha_size = 256

    # batch, seq, alpha
    pre_X = tf.placeholder(tf.int32, shape=[None, None])
    X = tf.one_hot(pre_X, alpha_size)

    pre_labels = tf.placeholder(tf.int32, shape=[None, None])
    labels = tf.one_hot(pre_labels, alpha_size)

    
    cell1 = tf.nn.rnn_cell.GRUCell(num_units=alpha_size)
    cell2 = tf.nn.rnn_cell.GRUCell(num_units=alpha_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

    outputs, states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float32,
        inputs=X)
    #outputs = tf.contrib.layers.batch_norm(pre_outputs)
    outputs_flat = tf.reshape(outputs, [-1, alpha_size])
    Ylogits = tf.contrib.layers.linear(outputs_flat, alpha_size)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    logits = Ylogits
    soft_logits = tf.nn.softmax(logits)
    Yflat_ = tf.reshape(labels, [-1, alpha_size])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    print("yflat", Yflat_)
    print("logits", logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_))  # [ BATCHSIZE x SEQLEN ]
    
    print(logits)

    optimizer = tf.train.AdamOptimizer().minimize(loss)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    sequence_length = 50
    batch_size = 16

    for epoch in range(100):
        avg_loss = []
        for num, batch in enumerate(get_random_batch(train_text, sequence_length=sequence_length, batch_size=batch_size)):
            
            output_labels, input_sequences = batch_to_labels(batch)

            the_loss, _, the_logits, the_labels, the_outputs = sess.run([loss, optimizer, logits, labels, Yflat_], feed_dict={pre_X:input_sequences, pre_labels:output_labels})
            real_logits = np.reshape(the_logits, (len(batch), 50, alpha_size))
            avg_loss.append(the_loss)

            if num % 500 == 0 and num > 0:
                print(".", end="", flush=True)

        sample_start_char = random.randint(65, 90)
        sample_sequence = [sample_start_char]
        for _ in range(75):
            feed = {pre_X: [sample_sequence]}
            the_logits = sess.run(logits, feed_dict=feed)
            next_char = np.argmax(the_logits[-1])
            sample_sequence.append(next_char)
        print("\nsample max:", ''.join([chr(s) for s in sample_sequence]))

        sample_start_char = random.randint(65, 90)
        sample_sequence = [sample_start_char]
        for _ in range(75):
            feed = {pre_X: [sample_sequence]}
            the_logits = sess.run(soft_logits, feed_dict=feed)
            next_char = np.random.choice(alpha_size, p=the_logits[-1])
            sample_sequence.append(next_char)
        print("\nsample softmax:", ''.join([chr(s) for s in sample_sequence]))


                
        print("\nepoch:", epoch)
        print("loss:", np.mean(avg_loss))
        avg_loss = []

        
            
                    
if __name__ == "__main__":
    argh.dispatch_command(main)
