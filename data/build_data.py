"""Script to build words, chars and tags vocab"""

__author__ = "Guillaume Genthial"

from collections import Counter
from pathlib import Path
from argparse import ArgumentParser
from importlib import import_module
import sys
import numpy as np

# TODO: modify this depending on your needs (1 will work just fine)
# You might also want to be more clever about your vocab and intersect
# the GloVe vocab with your dataset vocab, etc. You figure it out ;)
MINCOUNT = 1

if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument('data_path', help='path to data directory')
  parser.add_argument('--clean', action='store_true', help='remove existing outputs first')
  args = parser.parse_args()
  data_dir = Path(args.data_path)

  stream = import_module('{}.stream'.format(data_dir))

  vocab_words_file = data_dir / 'vocab.words.txt'
  vocab_chars_file = data_dir / 'vocab.chars.txt'
  vocab_tags_file  = data_dir / 'vocab.tags.txt'
  embeddings_file  = data_dir / 'embeddings.npz'

  if args.clean:
    for f in [vocab_words_file, vocab_chars_file, vocab_tags_file, embeddings_file]:
      try:
        f.unlink()
        print('Clean existing file {}'.format(f))
      except FileNotFoundError:
        print('Unable to clean missing file {}'.format(f))

  need_vocab_words = not (vocab_chars_file.is_file() and embeddings_file.is_file())

  for f, seq_fn in [ \
    (data_dir / 'train.words.txt', stream.train_words),
    (data_dir / 'train.tags.txt',  stream.train_tags),
    (data_dir / 'test.words.txt',  stream.test_words),
    (data_dir / 'test.tags.txt',   stream.test_tags),
  ]:
    if f.is_file() and args.clean:
      print('Clean existing file {}'.format(f))
      f.unlink()

    if not f.is_file():
      print('Generating {}'.format(f))
      with f.open('w') as f:
        for seq in seq_fn():
          f.write('{}\n'.format(' '.join(seq)))

    else:
      print('Skip {} (already done)'.format(f))
  
  # 1. Words
  # Get Counter of words on the training set, filter by min count, save

  if not vocab_words_file.is_file():
    print('Build vocab words (may take a while)')
    counter_words = Counter()
    for sent in stream.train_words():
      counter_words.update(sent)
    vocab_words = [w for w, c in counter_words.items() if c >= MINCOUNT]
    vocab_words = sorted(vocab_words)

    with vocab_words_file.open('w') as f:
      for w in vocab_words:
        f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(
      len(vocab_words), len(counter_words)))

  elif need_vocab_words:
    print('Read vocab words from file')
    vocab_words = [l.strip() for l in vocab_words_file.open().readlines()]
    print('- done. Read {}'.format(len(vocab_words)))

  else:
    print('Skip vocab words (already done)')


  # 2. Chars
  # Get all the characters from the vocab words

  if not vocab_chars_file.is_file():
    print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
      vocab_chars.update(w)

    with vocab_chars_file.open('w') as f:
      for c in sorted(list(vocab_chars)):
        f.write('{}\n'.format(c))
    print('- done. Found {} chars'.format(len(vocab_chars)))

  else:
    print('Skip vocab chars (already done)')


  # 3. Tags
  # Get all tags from the training set

  if not vocab_tags_file.is_file():
    print('Build vocab tags (may take a while)')
    vocab_tags = set()
    for tags in stream.train_tags():
      vocab_tags.update(tags)

    with vocab_tags_file.open('w') as f:
      for t in sorted(list(vocab_tags)):
        f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))

  else:
    print('Skip vocab tags (already done)')


  # 4. Embeddings
  # Get all word embeddings from the resource file

  if not embeddings_file.is_file():
    # Array of zeros
    word_to_idx = { word: idx for idx, word in enumerate(vocab_words) }
    size_vocab = len(word_to_idx)
    embedding_lines, embedding_size = stream.word_embeddings()

    embeddings = np.zeros((size_vocab, embedding_size))

    # Get relevant glove vectors
    found = 0
    print('Reading embedding file (may take a while)')

    for line_idx, line in enumerate(embedding_lines):
      if line_idx % 100000 == 0:
        print('- At line {}'.format(line_idx))
      line = line.strip().split()
      if len(line) != embedding_size + 1:
        continue
      word = line[0]
      embedding = line[1:]
      if word in word_to_idx:
        found += 1
        word_idx = word_to_idx[word]
        embeddings[word_idx] = embedding

    print('- done. Found {} vectors for {} words ({:.2f})'.format(found, size_vocab, found/size_vocab))

    # Save np.array to file
    np.savez_compressed(embeddings_file, embeddings=embeddings)
