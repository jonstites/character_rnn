#!/usr/bin/env python3

import argh
from collections import Counter
import glob
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.models import load_model
import random
import math
import os
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import json
import yaml


def process_text_file(text_file, chunk_size, min_count, validation_split=0.1, random_seed=0):
    with open(text_file) as handle:
        text = handle.read()

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i: i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
            
    random.seed(random_seed)
    random.shuffle(chunks)

    train_chunks = chunks[: int(validation_split * len(chunks))]
    counts = Counter()
    for chunk in train_chunks:
        counts.update(chunk)
    
    oov_char = "<OOV>"            
    invalid_chars = [char for char, count in counts.items() if count < min_count]
    counts[oov_char] = sum([counts[char] for char in invalid_chars])


    print("Training set had ", len(invalid_chars), " OOV char types representing ", counts[oov_char], " characters.")
    for char in invalid_chars:
        del counts[char]
    
    # deterministic sorting, highest counts first
    chars = sorted(counts.keys(), key=lambda x: (counts[x], x), reverse=True)

    ids = {}
    for char in chars:
        ids[char] = len(ids.keys())

    converted_chunks = []
    for chunk in chunks:
        converted = np.array([ids.get(char, ids[oov_char]) for char in chunk])
        converted_chunks.append(converted)
        
    converted_chunks = np.stack(converted_chunks)
    return np.array(converted_chunks), ids

def generate(model, ids, start_text="And so ", sample_length=500, sequence_length=100, temperature=0.5, beam_width=5):
    text = start_text
    reverse_ids = dict(zip(ids.values(), ids.keys()))
    oov_id = ids["<OOV>"]
    
    converted_text = np.array([ids.get(i, oov_id) for i in text])
    tiled = np.tile(converted_text, (1, 1))
    print(tiled.shape)

    parent_likelihoods = np.tile([0], (1))
    for _ in range(sample_length - len(text)):
        predictions = model.predict(tiled)
        last_char_predictions = predictions[:, -1, :]
        log_last_char_predictions = np.log(last_char_predictions)
        num_predictions = log_last_char_predictions.shape[-1]
        tiled_parent_likelihoods = np.tile(parent_likelihoods, (num_predictions, 1))
        last_char_likelihoods = tiled_parent_likelihoods.T + log_last_char_predictions

        best_indexes = np.unravel_index(np.argsort(last_char_likelihoods, axis=None)[-beam_width:], last_char_likelihoods.shape)

        parent_likelihoods = last_char_likelihoods[best_indexes]
        tiled = np.append(
            tiled[best_indexes[:-1]],
            np.expand_dims(best_indexes[-1], axis=1), axis=-1)

        
    for sample_batch in tiled:
        text = ''.join(
            [reverse_ids[char_id] for char_id in sample_batch])
        print(text)

    best_batch_index = np.unravel_index(np.argsort(parent_likelihoods, axis=None)[-1:], tiled.shape)
    best_batch = np.squeeze(tiled[best_batch_index[:-1]])
    return "".join(
            [reverse_ids[char_id] for char_id in best_batch])



def find_weights_file(model_dir, weights_filename):
    weights_regex = os.path.join(model_dir, "weights*.hdf5")
    weights_files = glob.glob(weights_regex)
    best_weights_file = None
    best_epoch = 0

    for weights_file in weights_files:
        base_name = os.path.basename(weights_file)
        epoch = int(base_name.split("-")[1])
        if epoch > best_epoch:
            best_epoch = epoch
            best_weights_file = weights_file
    return best_weights_file, best_epoch

def initialize(text_file, data_output="text_data.npy",
               ids_output="text_ids.json", min_count=10, sequence_length=100):

    data, ids = process_text_file(text_file, sequence_length, min_count)
    np.save(data_output, data)

    with open(ids_output, 'w') as ids_handle:
        json.dump(ids, ids_handle, indent=4)

def build_model(num_words, embedding_size, rnn_size, num_layers, dropout, use_cudnn):
    model = Sequential()
    model.add(keras.layers.Embedding(num_words, embedding_size, input_shape=(None,)))
    model.add(keras.layers.BatchNormalization())
    for _ in range(num_layers):

        if use_cudnn:
            model.add(keras.layers.CuDNNLSTM(rnn_size, return_sequences=True))        
        else:
            model.add(keras.layers.LSTM(rnn_size, return_sequences=True))
            
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout, noise_shape=(None, 1, None), seed=None))
            
    model.add(keras.layers.TimeDistributed(
        Dense(num_words, activation="softmax")))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model

def train(data_file, ids_file, model_dir, epochs=10, batch_size=32,
          use_cudnn=False, num_layers=3, rnn_size=256, embedding_size=10,
          dropout=0.5, save_period=10):

    data = np.load(data_file)
    with open(ids_file) as ids_handle:
        ids = json.load(ids_handle)

    num_words = len(ids.keys())
        
    model = build_model(num_words, embedding_size, rnn_size, num_layers, dropout, use_cudnn)
    print(model.summary())


    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    weights_filepath = os.path.join(model_dir, "weights-{epoch:02d}-{val_loss:.2f}.hdf5")
    last_weights_file, last_epoch = find_weights_file(model_dir, weights_filepath)
    if last_weights_file:
        model.load_weights(last_weights_file)
    

    train_x, train_y = data[:, :-1], data[:, 1:, np.newaxis]
    checkpointer = ModelCheckpoint(filepath=weights_filepath, verbose=1, save_best_only=False, period=save_period, save_weights_only=True)
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, callbacks=[checkpointer], initial_epoch=last_epoch, validation_split=0.1)
    


def generate_text(ids_file, model_dir, start_text="And so ", sample_length=500, temperature=0.5, use_cudnn=False, num_layers=3, rnn_size=256, embedding_size=10,
                  dropout=0.5, sequence_length=100):

    with open(ids_file) as ids_handle:
        ids = json.load(ids_handle)

    num_words = len(ids.keys())
        
    model = build_model(num_words, embedding_size, rnn_size, num_layers, dropout, use_cudnn)

    weights_filepath = os.path.join(model_dir, "weights-{epoch:02d}-{val_loss:.2f}.hdf5")
    last_weights_file, last_epoch = find_weights_file(model_dir, weights_filepath)
    if last_weights_file:
        model.load_weights(last_weights_file)

    result = generate(model, ids, sample_length=sample_length, start_text=start_text, temperature=temperature, sequence_length=sequence_length)
    print(result)
    
if __name__ == "__main__":
    argh.dispatch_commands([initialize, train, generate_text])
