from typing import List, Optional
from type_hint import SentTokens
from tokenize.tokenizer import Tokenizer
from tokenize.vocab import Vocab
from torchtext.vocab import GloVe, vocab


class WordEncoder:

  def __init__(self, vocab: Optional[Vocab] = None):
    self.tokenizer = Tokenizer(stopword=True)
    if vocab:
      self.vocab = vocab
    else:
      self.vocab = Vocab()

  def __call__(self,
               data: List[str],
               build_vocab: bool = False) -> List[List[int]]:
    """
    텍스트 데이터를 받아 전처리와 인코딩을 진행합니다. vocab이 만들어져 있지 않다면, 만듭니다.

    Args:
      data(List[str]): 텍스트 데이터
      build_vocab(bool): vocab을 만드는지 여부

    Returns:
      List[List[int]]: 인코딩된 데이터
    """
    if build_vocab:
      self.build_vocab(data)

    tokens = self.preprocess(data)
    encoded_data = [self.encode(sent) for sent in tokens]
    return encoded_data

  def build_vocab(self, data: List[str]):
    """
    텍스트 데이터를 받아 토큰화하고 vocab을 만듭니다.

    Args:
      data(List[str]): 텍스트 데이터
    """
    for text in data:
      tokens = self.tokenizer(text)
      self.vocab(tokens)

  def preprocess(self, data: List[str]) -> SentTokens:
    return self.padding([self.tokenizer(text) for text in data])

  def padding(self, tokens: SentTokens) -> SentTokens:
    """
    패딩을 진행합니다. 길이가 가장 긴 문장의 길이를 기준으로 패딩합니다.

    Args:
      tokens(SentTokens): 토큰화된 텍스트 데이터

    Returns:
      SentTokens: 패딩된 텍스트 데이터
    """
    max_len = max(len(token) for token in tokens)
    padded_sents = []

    for sent in tokens:
      pad_len = max_len - len(sent)
      padded_sent = sent + ["<PAD>"] * pad_len
      padded_sents.append(padded_sent)

    return padded_sents

  def encode(self, tokens: List[str]) -> List[int]:
    """
    Vocab 클래스를 활용하여 단어를 인코딩합니다. 모르는 단어는 <UNK>로 처리합니다.

    Args:
      tokens(List[str]): 토큰화된 텍스트 데이터

    Returns:
      List[int]: 인코딩된 데이터
    """
    encoded = [
        self.vocab.word2idx.get(token, self.vocab.word2idx["<UNK>"])
        for token in tokens
    ]
    return encoded

  def decode(self, tokens: List[int]) -> List[str]:
    """
    Vocab 클래스를 활용하여 단어를 디코딩합니다. 모르는 단어는 <UNK>로 처리합니다.

    Args:
      tokens(List[int]): 인코딩된 데이터

    Returns:
      List[str]: 디코딩된 데이터
    """
    decoded = [self.vocab.idx2word.get(token, "<UNK>") for token in tokens]
    return decoded


class GloveEncoder(WordEncoder):

  def __init__(self,
               name: str = '6B',
               dim: int = 200,
               special_tokens: List[str] = ['<PAD>', '<UNK>']):
    self.tokenizer = Tokenizer()
    self.glove = GloVe(name=name, dim=dim)
    self.vocab = vocab(self.glove.stoi, specials=special_tokens).get_stoi()
    self.special_tokens = special_tokens

  def encode(self, tokens: List[str]) -> List[int]:
    return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

  def decode(self, tokens):
    return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
