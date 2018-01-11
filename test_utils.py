from character_rnn import utils
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
def small_file_2(tmpdir):
    temp_file = os.path.join(tmpdir, "small_file_2.txt")
    with open(temp_file, 'w') as handle:
        handle.write(small_text_2())
    return temp_file

@pytest.fixture()
def small_files(tmpdir):
    return [small_file(tmpdir), small_file_2(tmpdir)]

@pytest.fixture()
def small_texts():
    texts = [small_text(), small_text_2()]
    return texts

@pytest.fixture()
def small_dataset(tmpdir):
    text_file = small_file(tmpdir)
    text_file_2 = small_file_2(tmpdir)
    dataset = utils.Dataset()

    dataset.add_file(text_file)
    dataset.add_file(text_file_2, source="atest")
    return dataset


class TestVocabulary:

    def test_load_text_file(self, tmpdir):
        filename = small_file(tmpdir)
        text = utils.Vocabulary().load_file(filename)
        assert text == small_text()
    
    def test_get_character_counts(self, tmpdir):
        filenames = small_files(tmpdir)
        counts = {"a": 3, "b": 1, "\n": 1, "z": 1, "A": 1}
        assert utils.Vocabulary()._get_character_counts(filenames) == counts

    def test_sort_keys_by_freq_and_lex(self, tmpdir):
        filenames = small_files(tmpdir)
        counts = utils.Vocabulary()._get_character_counts(filenames)
        sorted_keys = ["a", "\n", "A", "b", "z"]
        assert utils.Vocabulary()._sort_keys_by_freq_and_lex(counts) == sorted_keys

    def test_texts_to_ids(self, tmpdir):
        filenames = small_files(tmpdir)
        counts = utils.Vocabulary()._get_character_counts(filenames)
        ids = {"a": 0, "\n": 1, "A": 2, "b": 3, "z": 4}
        
        assert utils.Vocabulary()._create_ids(counts) == ids


