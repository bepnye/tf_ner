import os
import data_utils
from pathlib import Path

top_path = Path(os.path.dirname(os.path.abspath(__file__)))
EBM_NLP = Path('/Users/ben/Desktop/ebm_nlp/repo/ebm_nlp_2_00/')
NO_LABEL = '0'

def overwrite_tags(new_tags, tags):
  for i, t in enumerate(new_tags):
    if t != NO_LABEL:
      tags[i] = t

def get_tags(d):
  pmid_tags = {}
  for e in ['participants', 'interventions', 'outcomes']:
    for a in (EBM_NLP / 'annotations' / 'aggregated' / 'starting_spans' / e / d).glob('*.ann'):
      pmid = a.stem.split('.')[0]
      tags = a.open().read().split()
      tags = [e[0] if t == '1' else NO_LABEL for t in tags]
      if pmid not in pmid_tags:
        pmid_tags[pmid] = tags
      else:
        overwrite_tags(tags, pmid_tags[pmid])
  return pmid_tags

def get_words(pmids):
  return { pmid: (EBM_NLP / 'documents' / '{}.tokens'.format(pmid)).open().read().split() for pmid in pmids }

def get_seqs(tag_d, word_d, keys):
  tag_seqs = []
  word_seqs = []
  for k in keys:
    words, tags = data_utils.generate_seqs(word_d[k], tag_d[k])
    tag_seqs += tags
    word_seqs += words
  return word_seqs, tag_seqs

TRAIN_TAG_D = get_tags(Path('train/'))
TRAIN_PMIDS = sorted(TRAIN_TAG_D.keys())
TRAIN_WORD_D = get_words(TRAIN_PMIDS)

TRAIN_WORDS, TRAIN_TAGS = get_seqs(TRAIN_TAG_D, TRAIN_WORD_D, TRAIN_PMIDS)

TEST_TAG_D = get_tags(Path('test/gold/'))
TEST_PMIDS = sorted(TEST_TAG_D.keys())
TEST_WORD_D = get_words(TEST_PMIDS)

TEST_WORDS, TEST_TAGS = get_seqs(TEST_TAG_D, TEST_WORD_D, TEST_PMIDS)

def train_words():
  return TRAIN_WORDS
def train_tags():
  return TRAIN_TAGS
def test_words():
  return TEST_WORDS
def test_tags():
  return TEST_TAGS

def word_embeddings():
  return  ((top_path / '..' / 'embeddings' / 'glove.840B.300d.txt').open(), 300)
