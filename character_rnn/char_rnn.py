#!/usr/bin/env python3

import argh
from character_rnn import networks, utils


@argh.arg("filenames", nargs="+")
@argh.arg("-o", "--output-dir", required=True)
def preprocess(filenames, output_dir=None, chunk_size=250):
    v = utils.Vocabulary()
    v.learn_vocabulary(filenames)
    r = utils.RecordWriter(output_dir, v.vocabulary)
    r.chunk_size = chunk_size
    r.process(filenames)
    r.dump_vocabulary()
    
@argh.arg("train-filenames", nargs="+")
@argh.arg("validation-filenames", nargs="+")
@argh.arg("-o", "--output-dir", required=True)
def create_embedding(train_filenames, validation_filenames, vocabulary_file, output_dir=None, chunk_size=1000, embedding_size=10):
    networks.Embedding.create_embedding(train_filenames, validation_filenames, vocabulary_file, output_dir, chunk_size=chunk_size, embedding_size=embedding_size)

def create_text_generator(train_filenames, validation_filenames, vocabulary_file, embedding_model_file, output_dir=None, chunk_size=1000, embedding_size=10, sequence_length=100):
    networks.TextGenerator.create_text_generator(train_filenames, validation_filenames, vocabulary_file, embedding_model_file, output_dir, chunk_size=chunk_size, embedding_size=embedding_size, sequence_length=sequence_length)


def generate_text(vocabulary_file, model_file, embedding_size=10, sequence_length=100):
    networks.TextGenerator.generate_text(vocabulary_file, model_file, embedding_size=embedding_size, sequence_length=sequence_length)
    
if __name__ == "__main__":
    parser = argh.ArghParser()
    parser.add_commands([preprocess, create_embedding, create_text_generator, generate_text])
    parser.dispatch()
