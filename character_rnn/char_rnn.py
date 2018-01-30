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



def create_text_generator(
        train_filenames, validation_filenames, vocabulary_file, output_dir=None,
        chunk_size=1000, sequence_length=100, batch_size=32,
        rnn_size=256, num_layers=1):
    
    vocabulary = utils.Vocabulary.load_from_file(vocabulary_file)
    vocabulary_size = vocabulary.size            

    config= tf.estimator.RunConfig(
        save_checkpoints_steps=10000,
        save_checkpoints_secs=None
    )

    rnn = tf.estimator.Estimator(
        model_fn = networks.TextGenerator.model_fn,
        model_dir=output_dir,
        config=config,
        params={
            "vocabulary_size": vocabulary_size,
            "rnn_size": rnn_size,
            "num_layers": num_layers,
            "keep_prob": 1
            }
        )


    train_input = lambda: networks.TextGenerator.input_fn(train_filenames, sequence_length, batch_size, chunk_size, repeat_count=None, shuffle_count=1024)
    validation_input = lambda: networks.TextGenerator.input_fn(validation_filenames, sequence_length, batch_size, chunk_size, repeat_count=None, shuffle_count=1024)

    experiment = tf.contrib.learn.Experiment(
        rnn,
        train_input,
        validation_input,
        checkpoint_and_export=True,
        eval_steps=100
        )
    
    experiment.train_and_evaluate()



def generate_text(
        vocabulary_file, output_dir, sequence_length=100,
        sample_length=100, start="The", temperature=1.0):
    
    vocabulary = utils.Vocabulary.load_from_file(vocabulary_file)
    vocabulary_size = vocabulary.size            

    rnn = tf.estimator.Estimator(
        model_fn = networks.TextGenerator.model_fn,
        model_dir=output_dir,
        params={
            "vocabulary_size": vocabulary_size,
            "num_layers":1,
            "rnn_size":24
        }
    )

    text = networks.TextGenerator.sample(rnn, vocabulary, sequence_length, sample_length, start, temperature)
    print(text)
                                  
if __name__ == "__main__":
    parser = argh.ArghParser()
    parser.add_commands([preprocess, create_text_generator, generate_text])
    parser.dispatch()
