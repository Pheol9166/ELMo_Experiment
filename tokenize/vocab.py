from typing import List


class Vocab:

  def __init__(self):
    self.word2idx = {"<PAD>": 0, "<UNK>": 1}
    self.idx2word = {0: "<PAD>", 1: "<UNK>"}
    self.word_freq = {}

  def add_word(self, word: str) -> None:
    """
    단어가 사전에 있는지 확인 후 없으면 추가하고, 있으면 빈도를 증가시킵니다.

    Args:
      word(str): 추가할 단어
    """
    if word in self.word2idx:
      self.word_freq[word] += 1
    else:
      idx = len(self.word2idx)
      self.word2idx[word] = idx
      self.idx2word[idx] = word
      self.word_freq[word] = 1

  def __call__(self, tokens: List[str]):
    for token in tokens:
      self.add_word(token)

  def __len__(self):
    return len(self.word2idx)
