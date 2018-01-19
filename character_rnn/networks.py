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
        cropped = tf.random_crop(record, [10])
        character = tf.slice(cropped, [8], [1])
        next_character = tf.slice(cropped, [9], [1])
        return tf.reshape(cropped, shape=[10]), tf.reshape(character, shape=[]), tf.reshape(next_character, shape=[])

    def make_dataset(filenames, batch_size, chunk_size):
        parse_function = partial(Embedding.__parse__function, chunk_size=chunk_size)
        choose_random_context = partial(Embedding.__choose_random_context, chunk_size=chunk_size)    
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parse_function)
        dataset = dataset.map(choose_random_context)
        dataset = dataset.shuffle(buffer_size=batch_size * 5)
        dataset = dataset.repeat(None)        
        dataset = dataset.batch(batch_size)
        return dataset

    def create_embedding(train_filenames, validation_filenames, vocabulary_file, output_dir, chunk_size=1000, embedding_size=10, max_steps=200000):
        batch_size = 256
        num_sampled = 2
    
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
            cropped, characters, pre_contexts = next_element
            contexts = tf.reshape(pre_contexts, shape=(-1, 1))

        with tf.name_scope("Model"):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embeddings")
            embed = tf.nn.embedding_lookup(embeddings, characters, name="lookup")        
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)), name="nce_weights")
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="nce_biases")

        # Have an evaluation loss function??
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
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name="optimizer")

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
            #embedding.metadata_path = os.path.join(output_dir, 'metadata.tsv')


        summaries_tensor = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
        train_writer = tf.summary.FileWriter(os.path.join(output_dir, "train"), sess.graph)
        validation_writer = tf.summary.FileWriter(os.path.join(output_dir, "validation"), sess.graph)        
        
        sess.run(tf.global_variables_initializer())        
        sess.run(train_iterator.initializer)
        sess.run(validation_iterator.initializer)

        for step in range(1, max_steps+1):
            if step % 10 == 0:
                summary, crop, chars, next_chars = sess.run([summaries_tensor, cropped, characters, pre_contexts], feed_dict={handle: validation_handle})
                validation_writer.add_summary(summary, global_step=step)
                #for i in range(batch_size):
                #    print("".join(vocabulary.inverse_vocabulary[j] for j in crop[i]), vocabulary.inverse_vocabulary[chars[i]], vocabulary.inverse_vocabulary[next_chars[i]])
                
            else:
                _, summary = sess.run([optimizer, summaries_tensor], feed_dict={handle: training_handle})
                train_writer.add_summary(summary, global_step=step)                

            if step % 1000 == 0:
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(output_dir, "model.ckpt"), step)
                projector.visualize_embeddings(train_writer, config)


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

    def __random_crop(sequence, sequence_length, conv):
        cropped = tf.random_crop(sequence, [sequence_length+1])
        cropped_sequence = tf.slice(cropped, [0], [sequence_length])
        if conv:
            label = tf.reshape(tf.slice(cropped, [sequence_length], [1]), shape=[])
        else:
            label = tf.slice(cropped, [1], [sequence_length])
        return cropped_sequence, label
    
    def make_dataset(filenames,  sequence_length, batch_size, chunk_size, conv):
        parse_function = partial(TextGenerator.__parse__function, chunk_size=chunk_size)
        #embedding_lookup = partial(TextGenerator.__embedding_lookup, embeddings=embeddings)
        #pad_sequence = partial(TextGenerator.__pad_sequence, sequence_length=sequence_length)
        random_crop = partial(TextGenerator.__random_crop,
                              sequence_length=sequence_length, conv=conv)
        
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parse_function)
        #dataset = dataset.map(pad_sequence)
        dataset = dataset.map(random_crop)
        #dataset = dataset.map(embedding_lookup)
        dataset = dataset.repeat(None)
        dataset = dataset.shuffle(buffer_size=batch_size * 5)
        dataset = dataset.batch(batch_size)
        return dataset

    def generate_text(vocabulary_file, model_file, embedding_size=10, sequence_length=100):
        vocabulary = utils.Vocabulary.load_from_file(vocabulary_file)
        start_text = [vocabulary.vocabulary[i] for i in "And "]
        text = start_text
        sess=tf.Session()
        
        saver = tf.train.import_meta_graph(model_file)
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_file)))
        graph = tf.get_default_graph()
        embedding = graph.get_tensor_by_name("Model/normed_embeddings:0")
        closest_characters = graph.get_tensor_by_name("Accuracies/closest_characters:0")
        embedded_sequences = graph.get_tensor_by_name("Input/embedded_sequences:0")
        is_sampling = graph.get_tensor_by_name("Input/is_sampling:0")
        is_training = graph.get_tensor_by_name("Input/is_training:0")
        
        
        while len(text) < sequence_length:
            embedded_text = tf.nn.embedding_lookup(embedding, text, name="lookup")
            seq_len, _ = embedded_text.shape

            next_char = sess.run(closest_characters, feed_dict={embedded_sequences: embedded_text.eval(session=sess), is_sampling:True, is_training:False})
            text.append(next_char)
        converted_text = "".join(vocabulary.inverse_vocabulary[s] for s in text)
        print(converted_text)
        
    def create_text_generator(train_filenames, validation_filenames, vocabulary_file,
                              output_dir, chunk_size=1000, embedding_size=10, max_steps=500000,
                              sequence_length=100, conv=False):
        batch_size = 128

        vocabulary = utils.Vocabulary.load_from_file(vocabulary_file)
        vocabulary_size = vocabulary.size

            

        train_dataset = TextGenerator.make_dataset(train_filenames, sequence_length, batch_size, chunk_size, conv)
        train_iterator = train_dataset.make_initializable_iterator()

        validation_dataset = TextGenerator.make_dataset(validation_filenames, sequence_length, batch_size, chunk_size, conv)
        validation_iterator = validation_dataset.make_initializable_iterator()    

        sess = tf.Session()        
        training_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        with tf.name_scope("Input"):
            is_training = tf.placeholder(tf.bool, name="is_training")
            handle = tf.placeholder(tf.string, shape=[], name="handle")
            iterator = tf.data.Iterator.from_string_handle(
                handle, train_dataset.output_types, train_dataset.output_shapes)
            #sequences, embedded_sequences, labels, embedded_labels = iterator.get_next()

            sequences, labels = iterator.get_next()

        with tf.name_scope("Embeddings"):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embeddings")
            embedded_sequences = tf.nn.embedding_lookup(embeddings, sequences, name="lookup")        
            
        with tf.name_scope("Model"):

            if conv:
                weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                h = tf.contrib.layers.conv2d(embedded_sequences, kernel_size=10, num_outputs=80, weights_regularizer=weights_regularizer)
                h = tf.contrib.layers.batch_norm(h, is_training=is_training)
                h = tf.contrib.layers.conv2d(h, kernel_size=5, num_outputs=80, weights_regularizer=weights_regularizer, stride=2)
                h = tf.contrib.layers.batch_norm(h, is_training=is_training)
                h = tf.contrib.layers.conv2d(h, kernel_size=5, num_outputs=160, weights_regularizer=weights_regularizer, stride=2)
                h = tf.contrib.layers.batch_norm(h, is_training=is_training)
                h = tf.contrib.layers.conv2d(h, kernel_size=5, num_outputs=320, weights_regularizer=weights_regularizer, stride=2)
                h = tf.contrib.layers.batch_norm(h, is_training=is_training)
                h = tf.contrib.layers.flatten(h)
                outputs = tf.identity(tf.contrib.layers.fully_connected(h, vocabulary_size, activation_fn=None), name="outputs")

     
            else:
                num_layers = 4
                rnn_size = 512
                cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(num_units=rnn_size) for _ in range(num_layers)])

                outputs, states = tf.nn.dynamic_rnn(
                    cell=cell,
                    dtype=tf.float32,
                    inputs=embedded_sequences)
                outputs = tf.contrib.layers.linear(outputs, vocabulary_size, activation_fn=None)

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
                """
                seqq, predd, actt, cdd, summary = sess.run([sequences, closest_characters, labels, cosine_similarity, summaries_tensor], feed_dict={handle: validation_handle, is_training:False})

                for i in range(batch_size):
                    seq = seqq[i]
                    act = actt[i]
                    pred = predd[i]
                    cd = cdd[i]
                    print("seq: ", ''.join([vocabulary.inverse_vocabulary[s] for s in seq]), vocabulary.inverse_vocabulary[act], sep="")
                    print("pred: ", vocabulary.inverse_vocabulary[pred])
                    print("was correct: ", pred == act)
                    #for i, distance in enumerate(cd):
                    #print("cosine distance: ", distance, vocabulary.inverse_vocabulary[i])
                """
            else:
                _, summary = sess.run([optimizer, summaries_tensor], feed_dict={handle: training_handle, is_training:True})
                train_writer.add_summary(summary, global_step=step)                

            if step % 1000 == 0:
                
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(output_dir, "model.ckpt"), step)

