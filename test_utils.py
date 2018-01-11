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
    
class TestText:

    def test_load_text_file(self, tmpdir):
        target = small_file(tmpdir)
        text = utils.Text.load_file(target)
        assert text == small_text()

    def test_load_text_files(self, tmpdir):
        target = small_file(tmpdir)
        targets = [target, target]
        texts = utils.Text.load_files(targets)
        assert texts == [small_text(), small_text()]

    def test_count_characters(self):
        text = small_text()
        counts = {"": 0, "a": 2, "b": 1, "\n": 1}
        assert utils.Text._count_characters(text) == counts
        
    def test_multi_count_characters(self):
        texts = small_texts()
        counts = {"a": 3, "b": 1, "\n": 1, "z": 1, "A": 1, "": 0}
        assert utils.Text._multi_count_characters(texts) == counts

    def test_sort_keys_by_freq_and_lex(self):
        texts = small_texts()
        counts = utils.Text._multi_count_characters(texts)
        sorted_keys = ["a", "\n", "A", "b", "z", ""]
        assert utils.Text._sort_keys_by_freq_and_lex(counts) == sorted_keys
        
    def test_texts_to_ids(self):
        texts = small_texts()
        ids = {"a": 0, "\n": 1, "A": 2, "b": 3, "z": 4, "": 5}
        assert utils.Text._texts_to_ids(texts) == ids

    def test_to_ids(self):
        texts = small_texts()
        as_ids = [[0, 1, 3, 0], [ 4, 2, 0]]
        assert utils.Text.to_ids(texts, utils.Text._texts_to_ids(texts)) == as_ids

    def test_chunk_text(self):
        text = small_text()
        chunks = utils.Text.chunk(text, 3)
        assert chunks == ["a\nb"]

class TestDataset:

    def test_init(self):
        dataset = utils.Dataset()
        assert dataset.files == []
        assert dataset.sources == []
        
    def test_add_file(self, tmpdir):
        dataset = small_dataset(tmpdir)
        
        assert dataset.files[0] == small_file(tmpdir)
        assert dataset.files[1] == small_file_2(tmpdir)
        assert dataset.sources[0] == None
        assert dataset.sources[1] == "atest"

    def test_add_files(self, tmpdir):
        dataset = small_dataset(tmpdir)

        text_file = small_file(tmpdir)
        text_file_2 = small_file_2(tmpdir)
        dataset = utils.Dataset()
        dataset.add_files([text_file, text_file_2], [None, "atest"])
        
        assert dataset.files[0] == small_file(tmpdir)
        assert dataset.files[1] == small_file_2(tmpdir)
        assert dataset.sources[0] == None
        assert dataset.sources[1] == "atest"
        
    def test_load_files(self, tmpdir):
        dataset = small_dataset(tmpdir)
        dataset.load_files()
        
        assert dataset.texts == small_texts()
        
    def test_train_val_test_split(self, tmpdir):
        dataset = small_dataset(tmpdir)
        dataset.load_files()
        dataset.create_vocabulary()
        dataset.train_val_test_split(test_fraction=0.0, sequence_length=1)

        assert len(dataset.validation_batches) == 1
        assert len(dataset.train_batches) == 6
        assert len(dataset.test_batches) == 0
