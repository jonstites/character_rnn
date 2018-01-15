import numpy as np
import tensorflow as tf
from functools import partial
import json
import math
import os
from . import utils


class Embedding:

    def __parse__function(record, chunk_size):
        features = {"sequence": tf.FixedLenFeature(shape=chunk_size, dtype=tf.int64)}
        parsed_features = tf.parse_single_example(record, features)
        return parsed_features["sequence"]

    def __choose_random_context(record, chunk_size):
        index = np.random.randint(chunk_size - 1)
        character = record[index]

        # reshape context here
        context = record[index + 1]
        return character, context

    def make_dataset(filenames, batch_size, chunk_size):
        parse_function = partial(Embedding.__parse__function, chunk_size=chunk_size)
        choose_random_context = partial(Embedding.__choose_random_context, chunk_size=chunk_size)    
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parse_function)
        dataset = dataset.map(choose_random_context)
        dataset = dataset.repeat(None)
        dataset = dataset.shuffle(buffer_size=batch_size * 5)
        dataset = dataset.batch(batch_size)
        return dataset

    def create_embedding(train_filenames, validation_filenames, vocabulary_file, output_dir, chunk_size=1000, embedding_size=10, max_steps=10000):
        batch_size = 64
        num_sampled = 10
    
        train_dataset = Embedding.make_dataset(train_filenames, batch_size, chunk_size)
        train_iterator = train_dataset.make_initializable_iterator()

        validation_dataset = Embedding.make_dataset(validation_filenames, batch_size, chunk_size)
        validation_iterator = validation_dataset.make_initializable_iterator()    

        vocabulary = utils.Vocabulary.load_from_file(vocabulary_file)
        vocabulary_size = vocabulary.size

        sess = tf.Session()
        training_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        with tf.name_scope("Input"):
            handle = tf.placeholder(tf.string, shape=[], name="handle")
            iterator = tf.data.Iterator.from_string_handle(
                handle, train_dataset.output_types, train_dataset.output_shapes)
            next_element = iterator.get_next()
            characters, pre_contexts = next_element
            contexts = tf.reshape(pre_contexts, shape=(-1, 1))

        with tf.name_scope("Model"):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
            embed = tf.nn.embedding_lookup(embeddings, characters)

        with tf.name_scope("Cost"):
            loss = tf.identity(tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=contexts,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size)), name="loss")
            tf.summary.scalar("loss", loss)
            
        with tf.name_scope("Train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss, name="optimizer")

        with tf.name_scope("Accuracies"):
            logits = tf.matmul(embed, tf.transpose(nce_weights)) + nce_biases
            probs = tf.reduce_sum(tf.multiply(tf.nn.softmax(logits), tf.one_hot(pre_contexts, vocabulary_size)), axis=1)
            perplexity = tf.divide( tf.reduce_mean(tf.log(probs)), -tf.log(2.0), name="batch_perplexity")
            tf.summary.scalar("perplexity", perplexity)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), pre_contexts, name="correct_prediction"), "float"))
            tf.summary.scalar("accuracy", accuracy)

        with tf.name_scope("Projections"):
            from tensorflow.contrib.tensorboard.plugins import projector
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embeddings.name
            embedding.metadata_path = os.path.join(output_dir, 'metadata.tsv')


        summaries_tensor = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
        train_writer = tf.summary.FileWriter(os.path.join(output_dir, "train"), sess.graph)
        validation_writer = tf.summary.FileWriter(os.path.join(output_dir, "validation"), sess.graph)        
        
        sess.run(tf.global_variables_initializer())        
        sess.run(train_iterator.initializer)
        sess.run(validation_iterator.initializer)

        for step in range(max_steps):
            if step % 10 == 0:
                summary = sess.run(summaries_tensor, feed_dict={handle: validation_handle})
                validation_writer.add_summary(summary, global_step=step)
            else:
                _, summary = sess.run([optimizer, summaries_tensor], feed_dict={handle: training_handle})
                train_writer.add_summary(summary, global_step=step)                

            if step % 1000 == 0:
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(output_dir, "model.ckpt"), step)
                projector.visualize_embeddings(train_writer, config)
