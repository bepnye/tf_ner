import os
import data_utils
from pathlib import Path

top_path = Path(os.path.dirname(os.path.abspath(__file__)))
TRAIN_WORDS, TRAIN_TAGS = data_utils.read_file(top_path / 'train.txt')
TEST_WORDS,  TEST_TAGS  = data_utils.read_file(top_path / 'test.txt')

def train_words():
  return TRAIN_WORDS
def train_tags():
  return TRAIN_TAGS
def test_words():
  return TEST_WORDS
def test_tags():
  return TEST_TAGS

def word_embeddings():
  return ((top_path / '..' / 'embeddings' / 'glove.840B.300d.txt').open(), 300)
