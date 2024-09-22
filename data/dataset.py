from typing import Any, List, Tuple
from type_hint import Tensor
from tokenize.encoder import WordEncoder
import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):

  def __init__(self,
               data: Any,
               encoder: WordEncoder,
               build_vocab: bool = False,
               elmo_mode: bool = False):
    self.data = data
    self.encoder = encoder
    self.elmo_mode = elmo_mode
    self.encoded_data = self.encoder(self.data['title'],
                                     build_vocab=build_vocab)

    if elmo_mode:
      self.token_data = self.encoder.preprocess(self.data['title'])

  def __len__(self):
    return len(self.encoded_data)

  def __getitem__(
      self,
      idx: int) -> Tuple[Tensor, Tensor, List[str]] | Tuple[Tensor, Tensor]:
    """
    데이터셋으로부터 인덱스에 맞는 데이터를 얻습니다.
    elmo mode에 따라 패딩 처리된 토큰들이 같이 반환됩니다.

    Args:
      idx(int): 인덱스

    Returns:
      Tuple[Tensor, Tensor, List[str]]: 인덱스에 맞는 텍스트 데이터와 라벨 데이터, elmo mode에 따라 패딩 처리된 토큰들
      Tuple[Tensor, Tensor]: 인덱스에 맞는 텍스트 데이터와 라벨 데이터
    """
    text = self.encoded_data[idx]
    label = self.data.iloc[idx]['category']
    text_tensor = torch.tensor(text, dtype=torch.long)
    label_tensor = torch.tensor(label, dtype=torch.long)

    if self.elmo_mode:
      token = self.token_data[idx]
      return text_tensor, label_tensor, token
    else:
      return text_tensor, label_tensor
