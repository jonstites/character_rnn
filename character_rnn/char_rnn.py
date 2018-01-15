#!/usr/bin/env python3

import argh
from character_rnn import networks, utils


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
@argh.arg("-o", "--output-dir", required=True)
def create_embedding(train_filenames, validation_filenames, vocabulary_file, output_dir=None, chunk_size=1000, embedding_size=10):
    networks.Embedding.create_embedding(train_filenames, validation_filenames, vocabulary_file, output_dir, chunk_size=chunk_size, embedding_size=embedding_size)


if __name__ == "__main__":
    parser = argh.ArghParser()
    parser.add_commands([preprocess, create_embedding])
    parser.dispatch()
