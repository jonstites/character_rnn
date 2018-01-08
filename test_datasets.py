from character_rnn import datasets
import io
import os
import pytest

@pytest.fixture()
def small_text():
    s = "test\nstring"
    return s

@pytest.fixture()
def small_file(tmpdir):
    temp_file = os.path.join(tmpdir, "small_file.txt")
    with open(temp_file, 'w') as handle:
        handle.write(small_text())
    return temp_file
    

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


