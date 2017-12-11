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

def build_graph(embedding=True):
    alpha_size = 256
    embedding_size = 50
    rnn_size = 40
    # batch, seq, alpha
    g = tf.Graph()
    with g.as_default():
        char_integers = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        if embedding:
            embeddings = tf.get_variable(
                "char_embeddings",
                [alpha_size, embedding_size])

            sequence = tf.nn.embedding_lookup(embeddings, char_integers)
        else:
            sequence = tf.one_hot(char_integers, alpha_size)    

        label_integers= tf.placeholder(tf.int32, shape=[None, None], name="labels")
        sequence_labels = tf.one_hot(label_integers, alpha_size)
    
        h1 = tf.nn.rnn_cell.GRUCell(num_units=rnn_size)
        h2 = tf.nn.rnn_cell.GRUCell(num_units=rnn_size)
        cell = tf.nn.rnn_cell.MultiRNNCell([h1, h2])

        outputs, states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=sequence)

        logits = tf.contrib.layers.linear(outputs, alpha_size)
        _ = tf.nn.softmax(logits, name="logits")
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=sequence_labels))
        loss_tensor = tf.identity(loss, name="loss")
        optimizer = tf.train.AdamOptimizer().minimize(loss, name="optimizer")
        return g

def main(train_text, sequence_length=100, sample_length=500, batch_size=32):
    
    tf.reset_default_graph()

    # Create input data
    graph = build_graph()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(100):
            avg_loss = []
            for num, batch in enumerate(get_random_batch(train_text, sequence_length=sequence_length, batch_size=batch_size)):
            
                output_labels, input_sequences = batch_to_labels(batch)
                
                feed = {"inputs:0":input_sequences, "labels:0":output_labels}
                loss, _ = sess.run(["loss:0", "optimizer"], feed_dict=feed)
                avg_loss.append(loss)

                if num % 100 == 0 and num > 0:
                    print(loss, flush=True)

            sample_start_char = random.randint(65, 90)
            sample_sequence = [sample_start_char]
            #sample_sequence = []
            for _ in range(sample_length):
                feed = {"inputs:0":[sample_sequence]}
                logits = sess.run("logits:0", feed_dict=feed)
                next_char = np.random.choice(len(logits[-1]), p=logits[-1])
                sample_sequence.append(next_char)
            print("\nsample softmax:", ''.join([chr(s) for s in sample_sequence]))
            print("\nepoch:", epoch)
            print("loss:", np.mean(avg_loss))
            avg_loss = []


    print("\nepoch:", epoch)
    print("loss:", np.mean(avg_loss))
    avg_loss = []
        
            
                    
if __name__ == "__main__":
    argh.dispatch_command(main)
