from collections import Counter

class Text:

    def load_text_file(text_file):
        with open(text_file) as handle:
            return handle.read()
    
    def load_text_files(text_files):
        texts = []
        for text_file in text_files:
            text = Text.load_text_file(text_file)
            texts.append(text)
        return texts

    def to_ids(texts):
        as_ids = []
        ids = Text._texts_to_ids(texts)
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
        return counts

    def _multi_count_characters(texts):
        counts = Counter()
        for text in texts:
            counts.update(text)
        return counts
    
