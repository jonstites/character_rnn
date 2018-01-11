from collections import Counter
import math
import numpy as np
import os
import tensorflow as tf
import random


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class Vocabulary:

    def __init__(self):
        pass

    def learn_vocabulary(self, files):
        counts = self._get_character_counts(files)
        self.vocabulary = self._create_ids(counts)
        self.inverse_vocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))

    def _create_ids(self, counts):
        sorted_characters = self._sort_keys_by_freq_and_lex(counts)
        character_ids = {}
        for index, character in enumerate(sorted_characters):
            character_ids[character] = index
        return character_ids        
        
    def _sort_keys_by_freq_and_lex(self, counts):
        return sorted(counts.keys(), key=lambda x: (-counts[x], x))        

    def _get_character_counts(self, files):
        counts = Counter()
        for filename in files:
            text = self.load_file(filename)
            count = Counter(text)
            counts.update(count)
        return counts

    def load_file(self, text_file):
        with open(text_file) as handle:
            return handle.read()
    

class RecordWriter:

    def __init__(self, output_dir, vocabulary, chunk_size=1000):
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.validation_fraction = 0.1
        self.test_fraction = 0.1
        self.vocabulary = vocabulary
        
    def process(self, files):
        train_file = os.path.join(self.output_dir, "train.tfrecords")
        validation_file = os.path.join(self.output_dir, "validation.tfrecords")
        test_file = os.path.join(self.output_dir, "test.tfrecords")
        
        with tf.python_io.TFRecordWriter(train_file) as train_handle,\
             tf.python_io.TFRecordWriter(validation_file) as validation_handle,\
             tf.python_io.TFRecordWriter(test_file) as test_handle:
                    
            for chunk in self.read_files(files):
                converted_chunk = self.convert(chunk)
                record = self.create_tf_record(converted_chunk)
                handle = self.choose_handle(train_handle, validation_handle, test_handle)
                handle.write(record.SerializeToString())

    def choose_handle(self, train_handle, validation_handle, test_handle):
        handle = train_handle
        random_value = random.random()
        if random_value < self.test_fraction:
            handle = test_handle
        elif random_value < (self.validation_fraction + self.test_fraction):
            handle = validation_handle
        return handle
    
    def create_tf_record(self, chunk):
        record = tf.train.Example(features=tf.train.Features(feature={
            "sequence": _int64_feature(chunk)}))            
        return record

    def read_files(self, files):
        truncated = 0
        chunk_size = self.chunk_size
        for filename in files:
            with open(filename, 'r', encoding="utf-8") as file_handle:
                text = file_handle.read()
                for start in range(0, len(text) - chunk_size, chunk_size):
                    end = start + chunk_size
                    yield text[start: end]
                truncated += len(text) % chunk_size
        print("Truncated ", truncated, " characters due to chunk size.")

            
    def convert(self, chunk):
        return [self.vocabulary[char] for char in chunk]

