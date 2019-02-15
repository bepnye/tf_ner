import os
import data_utils
from pathlib import Path

top_path = Path(os.path.dirname(os.path.abspath(__file__)))
TRAIN_WORDS, TRAIN_TAGS = data_utils.read_file(top_path / 'wnut17train.conll')
TEST_WORDS,  TEST_TAGS  = data_utils.read_file(top_path / 'emerging.test.annotated')

def train_words():
  return TRAIN_WORDS
def train_tags():
  return TRAIN_TAGS
def test_words():
  return TEST_WORDS
def test_tags():
  return TEST_TAGS

#def word_embeddings():
#  return  ((top_path / '..' / 'embeddings' / 'glove.twitter.27B' / 'glove.twitter.27B.200d.txt').open(), 200)
def word_embeddings():
  return  ((top_path / '..' / 'embeddings' / 'glove.840B.300d.txt').open(), 300)
