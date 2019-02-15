def read_file(fpath, word_idx = 0, tag_idx = -1, filter_words = ['-DOCSTART-']):
  words = []
  tags = []
  with fpath.open() as f:
    for l in f:
      fields = l.strip().split()
      if fields:
        word = fields[word_idx]
        tag = fields[tag_idx]
      else:
        word = ''
        tag = None
      if word not in filter_words:
        words.append(word)
        tags.append(tag)

  return generate_seqs(words, tags, [''], False)

def generate_seqs(words, tags, terminal_words = ['', '.', '?', '!'], keep_terminal = True):
  word_seqs = []
  tag_seqs = []

  cur_word_seq = []
  cur_tag_seq = []

  def push_seq():
    if len(cur_word_seq) > 0:
      word_seqs.append(tuple(cur_word_seq))
      tag_seqs.append(tuple(cur_tag_seq))
      cur_word_seq.clear()
      cur_tag_seq.clear()
    else:
      pass # don't push empty seqs

  for word, tag in zip(words, tags):
    if word in terminal_words:
      if keep_terminal:
        cur_word_seq.append(word)
        cur_tag_seq.append(tag)
      push_seq()
    else:
      cur_word_seq.append(word)
      cur_tag_seq.append(tag)
  push_seq() # push final seq in case file doesn't end with a blank line

  return word_seqs, tag_seqs
