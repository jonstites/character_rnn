#!/usr/bin/env python3

import argh
from character_rnn import utils
import numpy as np
import tensorflow as tf
from functools import partial
import json
import math
import os


def __parse__function(record, chunk_size):
    features = {"sequence": tf.FixedLenFeature(shape=chunk_size, dtype=tf.int64)}
    parsed_features = tf.parse_single_example(record, features)
    return parsed_features["sequence"]

def __choose_random_context(record, chunk_size):
    index = np.random.randint(chunk_size - 1)
    character = record[index]
    context = record[index + 1]
    
    if np.random.random() < 0.5:
        character, context = context, character
    return character, context

@argh.arg("filenames", nargs="+")
@argh.arg("-o", "--output-dir", required=True)
def preprocess(filenames, output_dir=None):
    v = utils.Vocabulary()
    v.learn_vocabulary(filenames)
    r = utils.RecordWriter(output_dir, v.vocabulary)
    r.process(filenames)
    r.dump_vocabulary()
    
@argh.arg("train-filenames", nargs="+")
@argh.arg("validation-filenames", nargs="+")
def create_embedding(train_filenames, validation_filenames, vocabulary_file, chunk_size=1000):
    batch_size = 256
    embedding_size = 10
    num_sampled = 10
    
    parse_function = partial(__parse__function, chunk_size=chunk_size)
    choose_random_context = partial(__choose_random_context, chunk_size=chunk_size)    

    train_dataset = tf.data.TFRecordDataset(train_filenames)
    train_dataset = train_dataset.map(parse_function)
    train_dataset = train_dataset.map(choose_random_context)    
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size)    

    validation_dataset = tf.data.TFRecordDataset(validation_filenames)
    validation_dataset = validation_dataset.map(parse_function)
    validation_dataset = validation_dataset.map(choose_random_context)
    train_dataset = train_dataset.shuffle(buffer_size=10000)    
    validation_dataset = validation_dataset.batch(batch_size)    
    
    train_iterator = train_dataset.make_initializable_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()    

    with open(vocabulary_file) as handle:
        vocabulary = json.load(handle)
    vocabulary_size = len(vocabulary.keys())
    
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    next_element = iterator.get_next()
    characters, contexts = next_element
    contexts = tf.reshape(contexts, shape=(-1, 1))


    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    embed = tf.nn.embedding_lookup(embeddings, characters)

    if mode == "train":
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=contexts,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))

    else:
        loss = tf.reduce_mean(
            logits = tf.matmul(inputs, tf.transpose(weights))
            logits = tf.nn.bias_add(logits, biases)
            labels_one_hot = tf.one_hot(labels, n_classes)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_one_hot,
                logits=logits)
            loss = tf.reduce_sum(loss, axis=1)
            )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

    train_iterator = train_dataset.make_initializable_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())    
    training_handle = sess.run(train_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    LOG_DIR = "tf_output"
    from tensorflow.contrib.tensorboard.plugins import projector
    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, config)

    
    num_epochs = 10
    for epoch in range(num_epochs):
        sess.run(train_iterator.initializer)
        losses = []
        while True:

            try:
                loss_value, _ = sess.run([loss, optimizer], feed_dict={handle: training_handle})
                losses.append(loss_value)
            except tf.errors.OutOfRangeError:
                print("train loss: ", np.mean(losses))
                break

        # Run one pass over the validation dataset.
        sess.run(validation_iterator.initializer)
        losses = []
        while True:

            try:
                val_loss = sess.run(loss, feed_dict={handle: validation_handle})
                losses.append(val_loss)
            except tf.errors.OutOfRangeError:
                print("val loss: ", np.mean(losses))
                break

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), epoch)

    


if __name__ == "__main__":
    parser = argh.ArghParser()
    parser.add_commands([preprocess, create_embedding])
    parser.dispatch()
