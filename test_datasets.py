from character_rnn import datasets
import io
import os
import pytest

@pytest.fixture()
def small_text():
    s = "a\nba"
    return s

@pytest.fixture()
def small_text_2():
    s = "zAa"
    return s

@pytest.fixture()
def small_file(tmpdir):
    temp_file = os.path.join(tmpdir, "small_file.txt")
    with open(temp_file, 'w') as handle:
        handle.write(small_text())
    return temp_file

@pytest.fixture()
def small_texts():
    texts = [small_text(), small_text_2()]
    return texts


class TestText:

    def test_load_text_file(self, tmpdir):
        target = small_file(tmpdir)
        text = datasets.Text.load_text_file(target)
        assert text == small_text()

    def test_load_text_files(self, tmpdir):
        target = small_file(tmpdir)
        targets = [target, target]
        texts = datasets.Text.load_text_files(targets)
        assert texts == [small_text(), small_text()]

    def test_count_characters(self):
        text = small_text()
        counts = {"a": 2, "b": 1, "\n": 1}
        assert datasets.Text._count_characters(text) == counts
        
    def test_multi_count_characters(self):
        texts = small_texts()
        counts = {"a": 3, "b": 1, "\n": 1, "z": 1, "A": 1}
        assert datasets.Text._multi_count_characters(texts) == counts

    def test_sort_keys_by_freq_and_lex(self):
        texts = small_texts()
        counts = datasets.Text._multi_count_characters(texts)
        sorted_keys = ["a", "\n", "A", "b", "z"]
        assert datasets.Text._sort_keys_by_freq_and_lex(counts) == sorted_keys
        
    def test_texts_to_ids(self):
        texts = small_texts()
        ids = {"a": 0, "\n": 1, "A": 2, "b": 3, "z": 4}
        assert datasets.Text._texts_to_ids(texts) == ids

    def test_to_ids(self):
        texts = small_texts()
        as_ids = [[0, 1, 3, 0], [ 4, 2, 0]]
        assert datasets.Text.to_ids(texts) == as_ids
