import numpy as np
import tensorflow as tf
from functools import partial
import json
import math
import os
from . import utils


class TextGenerator:

    def __parse__function(record, chunk_size):
        features = {"sequence": tf.FixedLenFeature(shape=chunk_size, dtype=tf.int64)}
        parsed_features = tf.parse_single_example(record, features)
        return parsed_features["sequence"]

    def __embedding_lookup(sequence, label, embeddings):
        embedded_sequence = tf.nn.embedding_lookup(embeddings, sequence, name="lookup")
        embedded_label = tf.nn.embedding_lookup(embeddings, label, name="lookup")
        return sequence, embedded_sequence, label, embedded_label

    def __pad_sequence(sequence, sequence_length):
        paddings = tf.constant([[sequence_length, 0]])
        return tf.pad(sequence, paddings, "CONSTANT")

    def __random_crop(sequence, sequence_length):
        cropped = tf.random_crop(sequence, [sequence_length+1])
        cropped_sequence = tf.slice(cropped, [0], [sequence_length])
        label = tf.slice(cropped, [1], [sequence_length])
        return cropped_sequence, label
    
    def make_dataset(filenames,  sequence_length, batch_size, chunk_size, repeat_count, shuffle_count):
        parse_function = partial(TextGenerator.__parse__function, chunk_size=chunk_size)
        random_crop = partial(TextGenerator.__random_crop,
                              sequence_length=sequence_length)
        dataset = (tf.data.TFRecordDataset(filenames)
                   .map(parse_function)
                   .repeat(repeat_count)
                   .shuffle(shuffle_count)
                   .map(random_crop)
                   .batch(batch_size)
                   .cache()
                   .prefetch(1))
        return dataset

    def input_fn(filenames, sequence_length, batch_size, chunk_size, repeat_count=10, shuffle_count=1024):
        dataset = TextGenerator.make_dataset(filenames, sequence_length, batch_size, chunk_size, repeat_count, shuffle_count)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()        

    def input_fn_predict(start_text):
        dataset = tf.data.Dataset.from_tensors([start_text])
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
        
    def sample(rnn, vocabulary, sequence_length, sample_length, start="The", temperature=1.0):
        full_text = [vocabulary.vocabulary[char] for char in start]
        while len(full_text) < sample_length:
            text = full_text[-sequence_length:]
            predictions = rnn.predict(input_fn = lambda: TextGenerator.input_fn_predict(text))
            for prediction in predictions:
                probabilities = prediction["probabilities"]
                if np.random.random() <= temperature:
                    new_char = np.random.choice(vocabulary.size, p=probabilities[-1])
                else:
                    new_char = prediction["class_ids"]
                full_text.append(new_char)
        return "".join([vocabulary.inverse_vocabulary[char] for char in full_text])

    
            
    def model_fn(features, labels, mode, params):

        with tf.name_scope("Model"):
            weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4)

            keep_prob = 1
            if mode == tf.estimator.ModeKeys.TRAIN:
                keep_prob = params["keep_prob"]

                """            cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.LSTMBlockCell(num_units=params["rnn_size"]),
                    output_keep_prob=keep_prob) for _ in range(params["num_layers"])])
            outputs, states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                inputs=tf.one_hot(features, params["vocabulary_size"]))
                """
            cell = tf.contrib.cudnn_rnn.CudnnLSTM(
                params["num_layers"], params["rnn_size"], dropout=1-keep_prob)

            outputs, states = cell(
                tf.transpose(
                    tf.one_hot(features, params["vocabulary_size"]),
                    perm=[1, 0, 2]))
            
            #outputs = tf.contrib.layers.linear(outputs, params["vocabulary_size"], activation_fn=None, weights_regularizer=weights_regularizer)
            outputs = tf.contrib.layers.linear(
                tf.transpose(outputs, perm=[1, 0, 2]),
                params["vocabulary_size"], activation_fn=None, weights_regularizer=weights_regularizer)
            logits = tf.nn.softmax(outputs, name="logits")
            predicted_classes = tf.argmax(outputs, -1)
            
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            
                
        with tf.name_scope("Cost"):
            ml_loss = tf.identity(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs), name="ml_loss"))
            tf.summary.scalar("ml_loss", ml_loss)
            reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            tf.summary.scalar("reg_loss", reg_loss)
            loss = ml_loss + reg_loss
            tf.summary.scalar("loss", loss)
            
        with tf.name_scope("Train"):
            # hyperparameters?
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer().minimize(loss,  global_step=tf.train.get_global_step(), name="optimizer")
        

        with tf.name_scope("Accuracies"):
             
            accuracy = tf.metrics.accuracy(labels, predicted_classes)
            
            tf.summary.scalar("accuracy", accuracy[1])

            in_top_3 = tf.metrics.recall_at_k(labels, logits, 3)
            tf.summary.scalar("in_top_3", in_top_3[1])
            
            metrics = {
                "accuracy": accuracy,
                "top 3": in_top_3
            }
            
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer)
            
        if mode == tf.estimator.ModeKeys.EVAL:            
            return tf.estimator.EstimatorSpec(
                 mode, loss=loss, eval_metric_ops=metrics)
