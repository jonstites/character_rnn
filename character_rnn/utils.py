from collections import Counter
import math
import numpy as np


class Text:

    def load_file(text_file):
        with open(text_file) as handle:
            return handle.read()
    
    def load_files(text_files):
        texts = []
        for text_file in text_files:
            text = Text.load_file(text_file)
            texts.append(text)
        return texts

    def to_ids(texts, ids):
        as_ids = []
        for text in texts:
            as_id = [ids[char] for char in text]
            as_ids.append(as_id)
        return as_ids
    
    def _texts_to_ids(texts):
        counts = Text._multi_count_characters(texts)
        sorted_characters = Text._sort_keys_by_freq_and_lex(counts)
        character_ids = {}
        for index, character in enumerate(sorted_characters):
            character_ids[character] = index
        return character_ids

    def _sort_keys_by_freq_and_lex(counts):
        return sorted(counts.keys(), key=lambda x: (-counts[x], x))
    
    def _count_characters(text):
        counts = Counter(text)
        counts[""] = 0        
        return counts

    def _multi_count_characters(texts):
        counts = Counter()
        for text in texts:
            counts.update(text)
        counts[""] = 0
        return counts

    def chunk(text, sequence_length, pad=None):
        chunks = []
        for start in range(0, len(text), sequence_length):
            chunk = text[start: start + sequence_length]
            if len(chunk) < sequence_length:
                break
            if pad:
                chunk = [pad] + chunk
            chunks.append(chunk)
        return chunks
        
    
class Dataset:

    def __init__(self):
        self.files = []
        self.sources = []
        self.texts = []

    def add_file(self, filename, source=None):
        self.files.append(filename)
        self.sources.append(source)

    def add_files(self, filenames, sources=None):
        if sources is None:
            sources = [None for _ in len(filenames)]
        
        assert len(filenames) == len(sources)

        for filename, source in zip(filenames, sources):
            self.add_file(filename, source)        

    def preprocess(self):
        self.load_files()
        self.create_vocabulary()
        self.train_val_test_split()
            
    def load_files(self):
        texts = Text.load_files(self.files)
        self.texts = texts
        
    def create_vocabulary(self):
        self.vocabulary = Text._texts_to_ids(self.texts)
        self.inverse_vocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))
        self.texts = Text.to_ids(self.texts, self.vocabulary)
    
    def train_val_test_split(self, val_fraction=0.1, test_fraction=0.1, sequence_length=100):
        batches = []
        sources = []
        truncated_sequence = 0
        for source, text in zip(self.sources, self.texts):
            chunks = Text.chunk(text, sequence_length, pad=self.vocabulary[""])
            batches += chunks
            truncated_sequence += len(text) % sequence_length
            sources += [source] * len(chunks)
            
        print("Truncated ", truncated_sequence, " characters.")
        np.random.seed(0)
        np.random.shuffle(batches)
        np.random.seed(0)
        np.random.shuffle(sources)

        num_val = math.ceil(len(batches) * val_fraction)
        num_test = math.ceil(len(batches) * test_fraction)


        # convert to np here?
        self.validation_batches = np.asarray(batches[:num_val])
        self.validation_sources = np.asarray(sources[:num_val])
        self.test_batches = np.asarray(batches[num_val:num_val+num_test])
        self.test_sources = np.asarray(sources[num_val:num_val+num_test])
        self.train_batches = np.asarray(batches[num_val+num_test:])
        self.train_sources = np.asarray(sources[num_val+num_test:])

def batches(batches, sources, batch_size=128):

    seed = 0

    if len(batches) < batch_size:
        print("not enough batches for batch size")
        return
    
    while True:
        np.random.seed(seed)
        np.random.shuffle(batches)
        np.random.seed(seed)
        np.random.shuffle(sources)

        for start in range(0, len(batches), batch_size):
            batch = batches[start: start + batch_size]
            if len(batch) != batch_size:
                break
            source = sources[start: start + batch_size]
            yield batch[:-1], batch[1:], source
                
        seed += 1
