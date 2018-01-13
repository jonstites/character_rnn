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
        context = record[index + 1]
        return character, context

    def make_dataset(filenames, batch_size, buffer_size, chunk_size):
        parse_function = partial(Embedding.__parse__function, chunk_size=chunk_size)
        choose_random_context = partial(Embedding.__choose_random_context, chunk_size=chunk_size)    

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parse_function)
        dataset = dataset.map(choose_random_context)    
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        return dataset
    
    def create_embedding(train_filenames, validation_filenames, vocabulary_file, chunk_size=1000, embedding_size=10):
        batch_size = 128
        buffer_size = 4000
        num_sampled = 3
    
        train_dataset = Embedding.make_dataset(train_filenames, batch_size, buffer_size, chunk_size)
        train_iterator = train_dataset.make_initializable_iterator()

        validation_dataset = Embedding.make_dataset(validation_filenames, batch_size, buffer_size, chunk_size)
        validation_iterator = validation_dataset.make_initializable_iterator()    

        vocabulary = utils.Vocabulary.load_from_file(vocabulary_file)
        vocabulary_size = vocabulary.size
    
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
        is_training = tf.placeholder(dtype=bool, shape=None)

        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=contexts,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

        train_iterator = train_dataset.make_initializable_iterator()
        validation_iterator = validation_dataset.make_initializable_iterator()


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
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())    
        training_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
        
        num_epochs = 10
        for epoch in range(num_epochs):
            sess.run(train_iterator.initializer)
            losses = []
        
            while True:

                try:
                    loss_value, _ = sess.run([loss, optimizer], feed_dict={handle: training_handle, is_training: True})
                    losses.append(loss_value)
                except tf.errors.OutOfRangeError:
                    print("train loss: ", np.mean(losses))
                    break
        
            # Run one pass over the validation dataset.
            sess.run(validation_iterator.initializer)
            losses = []
            while True:
                try:
                    val_loss = sess.run(loss, feed_dict={handle: validation_handle, is_training: False})
                    losses.append(val_loss)
                except tf.errors.OutOfRangeError:
                    print("val loss: ", np.mean(losses))
                    break
        
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), epoch)
