from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import nltk

nltk.download('punkt')


class Tokenizer:

  def __init__(self, stopword=True):
    if stopword:
      nltk.download('stopwords')
      self.stopword_list = stopwords.words('english')
    else:
      self.stopword_list = None

  def __call__(self, text: str) -> List[str]:
    text = self.clean_text(text)
    tokens = self.tokenize(text)
    return tokens

  def clean_text(self, text: str) -> str:
    """
    괄호 안의 내용을 제거하고 a-zA-Z0-9 ,.!?를 제외한 문자를 제거합니다.

    Args:
      text(str): 처리할 텍스트

    Returns:
      str: 처리된 텍스트
    """
    text = text.lower()
    text = re.sub(r'\([^)]*\)', "", text)
    text = re.sub(r'[^a-zA-Z0-9 ,.!?]', "", text)
    return text

  def tokenize(self, text: str) -> List[str]:
    """
    텍스트를 불용어처리한 후 토큰화합니다.

    Args:
      text(str): 처리할 텍스트
      
    Returns:
      List[str]: 토큰화된 텍스트
    """
    tokens = word_tokenize(text)
    if self.stopword_list:
      tokens = [token for token in tokens if token not in self.stopword_list]
    return tokens
