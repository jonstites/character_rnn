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
    
    def make_dataset(filenames,  sequence_length, batch_size, chunk_size):
        parse_function = partial(TextGenerator.__parse__function, chunk_size=chunk_size)
        random_crop = partial(TextGenerator.__random_crop,
                              sequence_length=sequence_length)
        dataset = (tf.data.TFRecordDataset(filenames)
                   .map(parse_function)
                   .map(random_crop)
                   .cache()
                   .shuffle(batch_size * 10)
                   .repeat(None)
                   .batch(batch_size)
                   .prefetch(1))
        return dataset

    def input_fn(filenames, sequence_length, batch_size, chunk_size):
        dataset = TextGenerator.make_dataset(filenames, sequence_length, batch_size, chunk_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()    

    def input_fn_predict(vocabulary, sequence_length, sample_length):
        start = "The"
        start_text = None
        pass
    
    def model_fn(features, labels, mode, params):

        with tf.name_scope("Model"):
            weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            num_layers = 1
            rnn_size = 48

            keep_prob = 1 #tf.cond(is_training, lambda: 0.5, lambda: 1.0)
            cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.contrib.rnn.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(num_units=rnn_size),
                    output_keep_prob=keep_prob) for _ in range(num_layers)])

            outputs, states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                inputs=tf.one_hot(features, params["vocabulary_size"]))
                
            outputs = tf.contrib.layers.linear(outputs, params["vocabulary_size"], activation_fn=None, weights_regularizer=weights_regularizer)            
            logits = tf.nn.softmax(outputs, name="logits")

                
        with tf.name_scope("Cost"):
            print("labels", labels)
            print("outputs", outputs)
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
            
            # replace with tf metrics?
            predicted_classes = tf.argmax(outputs, -1)
            print("pc", predicted_classes)
            print("labels", labels)
            accuracy = tf.metrics.accuracy(predicted_classes, labels)

            tf.summary.scalar("accuracy", accuracy[0])

            in_top_3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(
                tf.reshape(outputs, shape=(-1, params["vocabulary_size"])),
                tf.reshape(labels, shape=[-1]), 3), "float"), name="in_top_3")
            tf.summary.scalar("in_top_3", in_top_3)
            
            metrics = {
                "accuracy": accuracy
            }
            
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer)
            
        if mode == tf.estimator.ModeKeys.EVAL:            
            return tf.estimator.EstimatorSpec(
                 mode, loss=loss, eval_metric_ops=metrics)


        if mode == tf.estimate.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(logits),
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    def generate_text(vocabulary_file, model_file, embedding_size=10, sequence_length=100, start_sequence="The ", conv=False):
        vocabulary = utils.Vocabulary.load_from_file(vocabulary_file)
        start_text = [vocabulary.vocabulary[i] for i in start_sequence]
        text = start_text
        sess=tf.Session()
        
        saver = tf.train.import_meta_graph(model_file)
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_file)))
        graph = tf.get_default_graph()
        sequences = graph.get_tensor_by_name("Input/sequences:0")
        logits = graph.get_tensor_by_name("Model/logits:0")
        is_training = graph.get_tensor_by_name("Input/is_training:0")
        _, old_sequence_length = sequences.get_shape().as_list()
        
        while len(text) < sequence_length:
            text_to_use = text[-old_sequence_length:]
            padded_text = np.pad(text_to_use, (old_sequence_length - len(text_to_use), 0), mode="constant")
            resh = np.reshape(padded_text, (1, -1))
            logts = sess.run(logits, feed_dict={sequences: resh, is_training:False})

            if conv:
                print(logts)
                next_char_int = np.random.choice(len(logts[0]), p=logts[0])
            else:
                next_char_int = np.random.choice(len(logts[0][-1]), p=logts[0][-1])
            text.append(next_char_int)
        converted_text = "".join(vocabulary.inverse_vocabulary[s] for s in text)
        print(converted_text)
        
    def create_text_generator(train_filenames, validation_filenames, vocabulary_file,
                              output_dir, chunk_size=1000, embedding_size=10, max_steps=500000,
                              sequence_length=100):
        batch_size = 32

        vocabulary = utils.Vocabulary.load_from_file(vocabulary_file)
        vocabulary_size = vocabulary.size            

        train_dataset = TextGenerator.make_dataset(train_filenames, sequence_length, batch_size, chunk_size, conv)
        train_iterator = train_dataset.make_initializable_iterator()

        validation_dataset = TextGenerator.make_dataset(validation_filenames, sequence_length, batch_size, chunk_size, conv)
        validation_iterator = validation_dataset.make_initializable_iterator()    

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))        
        training_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        with tf.name_scope("Input"):
            is_training = tf.placeholder(tf.bool, name="is_training")
            handle = tf.placeholder(tf.string, shape=[], name="handle")

            if handle is None:
                sequences = tf.placeholder(tf.int32, shape=[None, None], name="sequences")
                labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
            else:
                iterator = tf.data.Iterator.from_string_handle(
                    handle, train_dataset.output_types, train_dataset.output_shapes)
                sequences, labels = iterator.get_next()
                sequences = tf.identity(sequences, name="sequences")
                labels = tf.identity(labels, name="labels")

        with tf.name_scope("Embeddings"):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embeddings")
            embedded_sequences = tf.nn.embedding_lookup(embeddings, sequences, name="lookup")        
            
        with tf.name_scope("Model"):
            weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            num_layers = 4
            rnn_size = 1024

            keep_prob = tf.cond(is_training, lambda: 0.5, lambda: 1.0)
            cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.contrib.rnn.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(num_units=rnn_size),
                    output_keep_prob=keep_prob) for _ in range(num_layers)])

            outputs, states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                inputs=embedded_sequences)
                
            outputs = tf.contrib.layers.linear(outputs, vocabulary_size, activation_fn=None, weights_regularizer=weights_regularizer)
            
            logits = tf.nn.softmax(outputs, name="logits")

                
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
                optimizer = tf.train.AdamOptimizer().minimize(loss, name="optimizer")

        with tf.name_scope("Accuracies"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs, -1), labels) , "float"), name="accuracy")
            #accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels) , "float"), name="accuracy")
            tf.summary.scalar("accuracy", accuracy)

            if conv:
                in_top_3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(tf.reshape(outputs, shape=(-1, vocabulary_size)), labels, 3), "float"), name="in_top_3")
            else:

                in_top_3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(
                    tf.reshape(outputs, shape=(-1, vocabulary_size)),
                    tf.reshape(labels, shape=[-1]), 3), "float"), name="in_top_3")
            tf.summary.scalar("in_top_3", in_top_3)
            
        summaries_tensor = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
        train_writer = tf.summary.FileWriter(os.path.join(output_dir, "train"), sess.graph)
        validation_writer = tf.summary.FileWriter(os.path.join(output_dir, "validation"), sess.graph)        


        sess.run(tf.global_variables_initializer())        
        sess.run(train_iterator.initializer)
        sess.run(validation_iterator.initializer)

        for step in range(1, max_steps+1):
            if step % 10 == 0:
                summary = sess.run( summaries_tensor, feed_dict={handle: validation_handle, is_training:False})
                validation_writer.add_summary(summary, global_step=step)

            else:
                _, summary = sess.run([optimizer, summaries_tensor], feed_dict={handle: training_handle, is_training:True})
                train_writer.add_summary(summary, global_step=step)                

            if step % 100 == 0:                
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(output_dir, "model.ckpt"), step)

