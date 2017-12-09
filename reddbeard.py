#!/usr/bin/env python3

import argh
import bz2
import json
import numpy as np
import random
import tensorflow as tf


ascii_pad_character = 256


def get_sequences(records):
    with bz2.BZ2File(records, "rb") as dataset:
        lines = random.shuffle(dataset.readlines())
        for line in lines:
            data = json.loads(line.decode("UTF-8"))
            yield data["body"]

def get_batch(records, batch_size=32, max_length=300):
    batch = []
    lengths = []
    for seq in get_sequences(records):
        if len(seq) >= max_length:
            continue
        if not is_ascii(seq):
            continue
        length = len(seq)
        processed = ascii_to_integers(seq)
        padded = pad_ascii(processed, max_length)
        batch.append(padded)
        lengths.append(length)
        if len(batch) >= batch_size:
            yield batch, lengths
            batch = []
            lengths = []
    if len(batch) > 0:
        yield batch, lengths

def is_ascii(seq):
    return len(seq) == len(seq.encode())

def ascii_to_integers(seq):
    return [ord(letter) for letter in seq]

def pad_ascii(seq, max_length):
    pad_num = max_length - len(seq) 
    return seq + [ascii_pad_character]*pad_num
        

def main(train_records):

    
    tf.reset_default_graph()

    # Create input data
    batch_size=16

    alpha_size = 256

    # batch, seq, alpha
    pre_X = tf.placeholder(tf.int32, shape=[None, None])
    X = tf.one_hot(pre_X, alpha_size)

    X_lengths = tf.placeholder(tf.int32, shape=[None])
    pre_labels = tf.placeholder(tf.int32, shape=[None, None])
    labels = tf.one_hot(pre_labels, alpha_size)

    
    cell1 = tf.nn.rnn_cell.GRUCell(num_units=alpha_size)
    cell2 = tf.nn.rnn_cell.GRUCell(num_units=alpha_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

    outputs, states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float32,
        sequence_length=X_lengths,
        inputs=X)
    #outputs = tf.contrib.layers.batch_norm(pre_outputs)
    outputs_flat = tf.reshape(outputs, [-1, alpha_size])
    Ylogits = tf.contrib.layers.linear(outputs_flat, alpha_size)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    logits = Ylogits
    #soft_logits = tf.nn.softmax(logits)
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

    
    for epoch in range(100):
        #for comment in get_sequences(train_records):
        avg_loss = []
        for num, batch in enumerate(get_batch(train_records, batch_size=batch_size)):
            sequences, lengths = batch
            input_sequences = [seq[:-1] for seq in sequences]
            output_sequences = [seq[1:] for seq in sequences]

            the_loss, _, the_logits, the_labels, the_outputs = sess.run([loss, optimizer, logits, labels, Yflat_], feed_dict={pre_X:input_sequences, X_lengths:lengths, pre_labels:output_sequences})
            real_logits = np.reshape(the_logits, (len(sequences), 299, alpha_size))
            avg_loss.append(the_loss)
            
            if num % 100 == 0 and num != 0:
                print("epoch:", epoch)
                print("batch:", num)
                print("loss:", np.mean(avg_loss))
                avg_loss = []

                
                for i in range(len(sequences)):
                    sequence = sequences[i]
                    length = lengths[i]
                    print("".join([chr(sequence[j]) for j in range(length)]))
                    print("".join([chr(np.argmax(real_logits[i][j])) for j in range(length)]), flush=True)

                """
                seq = ascii_to_integers(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")))
                seq = pad_ascii(seq, 300)
                for i in range(1, 299):
                    probs = sess.run(soft_logits, feed_dict={pre_X:[seq], X_lengths:[i]})
                    new_char = np.random.choice(alpha_size, p=probs[i-1])
                    seq[i] = new_char
                print("A sample:")
                print("".join([chr(s) for s in seq]), flush=True)
                """
                    
if __name__ == "__main__":
    argh.dispatch_command(main)
