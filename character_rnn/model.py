#!/usr/bin/env python3


import argh
from collections import Counter
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
import math


def load_text(text_file):
    with open(text_file) as handle:
        text = handle.read()
    return text

def split_text(text):
    chunks = []
    chunk_size=100
    for i in range(0, len(text), chunk_size):
        chunk = text[i: i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
    return chunks

def train_val_split(text):

    chunks = split_text(text)

    random.seed(0)
    randomized_chunks = random.shuffle(chunks)

    validation_fraction = 0.1
    pivot = math.ceil( len(chunks) * validation_fraction)

    validation = chunks[:pivot]
    train = chunks[pivot:]

    return train, validation

def get_ids(train, min_count):
    counts = Counter()
    for chunk in train:
        counts.update(chunk)
    valid_chars = [k for k,v in counts.items() if v >= min_count]

    # deterministic sorting, highest counts first
    chars = sorted(valid_chars, key=lambda x: (counts[x], x), reverse=True)

    # skip zero: special padding id
    ids = {k: v + 1 for v, k in enumerate(chars) if counts[k] >= min_count}
    return ids

def convert(chunks, ids):
    converted_x = []
    converted_y = []    
    oov_id = len(ids.keys()) + 1
    for chunk in chunks:
        converted_chunk = [ids.get(c, oov_id) for c in chunk]
        x = converted_chunk[:-1]
        y = [[i] for i in converted_chunk[1:]]
        converted_x.append(x)
        converted_y.append(y)
    return keras.preprocessing.sequence.pad_sequences(converted_x, padding="post"), keras.preprocessing.sequence.pad_sequences(converted_y, padding="post")


def converted_train_val_split(text_file, min_count):
    text = load_text(text_file)
    train, validation = train_val_split(text)
    ids = get_ids(train, min_count)
    return convert(train, ids), convert(validation, ids), ids


def generate(model, ids, start_text="And so ", length=200, temperature=0.5):
    text = start_text
    reverse_ids = dict(zip(ids.values(), ids.keys()))
    
    while len(text) < length:
        converted_text = np.expand_dims(np.array([ids.get(i, len(ids.keys())+1) for i in text]), axis=0)

        prediction = model.predict(converted_text)
        probs = prediction[0][-1]

        if np.random.random() <= temperature:
            new_char_id = np.random.choice(len(ids.keys())+1, p=probs) 
        else:
            new_char_id = np.argmax(probs)
        new_char = reverse_ids.get(new_char_id, "<OOV>")

        # inefficient...
        text = text + new_char
    return text
        

def main(text_file, min_count=10, epochs=10):

    train, validation, ids = converted_train_val_split(text_file, min_count)
    train_x, train_y = train
    validation_x, validation_y = validation
    batch_size = 64
    num_words = len(ids) + 1
    print(train_y.shape)
    print(train_x.shape)
    

    model = Sequential()
    model.add(keras.layers.Embedding(num_words, 5, input_shape=(None,)))
    model.add(keras.layers.CuDNNLSTM(128, return_sequences=True))
    model.add(keras.layers.TimeDistributed(
        Dense(num_words, activation="softmax")))
    print(model.summary())
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
    loss_and_metrics = model.evaluate(validation_x, validation_y)    
    print(loss_and_metrics)
    print(generate(model, ids))

if __name__ == "__main__":
    argh.dispatch_command(main)
