#!/usr/bin/env python3

import argh
from character_rnn import networks, utils
import tensorflow as tf

@argh.arg("filenames", nargs="+")
@argh.arg("-o", "--output-dir", required=True)
def preprocess(filenames, output_dir=None, chunk_size=250):
    v = utils.Vocabulary()
    v.learn_vocabulary(filenames)
    r = utils.RecordWriter(output_dir, v.vocabulary)
    r.chunk_size = chunk_size
    r.process(filenames)
    r.dump_vocabulary()
    
def create_text_generator(train_filenames, validation_filenames, vocabulary_file, chunk_size=1000, sequence_length=100):
    #networks.TextGenerator.create_text_generator(train_filenames, validation_filenames, vocabulary_file, output_dir, chunk_size=chunk_size, sequence_length=sequence_length, embedding_size=embedding_size)

    vocabulary = utils.Vocabulary.load_from_file(vocabulary_file)
    vocabulary_size = vocabulary.size            

    rnn = tf.estimator.Estimator(
        model_fn = networks.TextGenerator.model_fn,
        params={
            "vocabulary_size": vocabulary_size
            }
        )

    #config=

    #tf.estimator.RunConfig(session_config=tf.ConfigProto(log_device_placement=True))

    for _ in range(100):
        rnn.train(
            input_fn=lambda: networks.TextGenerator.input_fn(train_filenames, sequence_length, 32, chunk_size),
            steps=200)

        rnn.evaluate(
            input_fn=lambda: networks.TextGenerator.input_fn(train_filenames, sequence_length, 32, chunk_size),
            steps=1, name="train")
        
        rnn.evaluate(
            input_fn=lambda: networks.TextGenerator.input_fn(validation_filenames, sequence_length, 32, chunk_size),
            steps=1, name="validation")


        #predictions = rnn.predict(
        #    input_fn=lambda: networks.TextGenerator.input_fn(validation_filenames, sequence_length, 32, chunk_size))

        #for prediction in predictions:
        #    print(prediction)


def generate_text(vocabulary_file, model_file, embedding_size=10, sequence_length=100, conv=False):
    networks.TextGenerator.generate_text(vocabulary_file, model_file, embedding_size=embedding_size, sequence_length=sequence_length, conv=conv)
    
if __name__ == "__main__":
    parser = argh.ArghParser()
    parser.add_commands([preprocess, create_text_generator, generate_text])
    parser.dispatch()
