from typing import Optional
from type_hint import SentTokens, Tensor
from models.elmo import ELMo
import torch
import torch.nn as nn


class NewsEmbedding(nn.Module):

  def __init__(self,
               vocab_size: int,
               embedding_dim: int,
               dropout: float = 0.3,
               elmo_mode: bool = False,
               elmo_option: Optional[str] = None):
    super(NewsEmbedding, self).__init__()
    self._embedding = nn.Embedding(vocab_size, embedding_dim)
    self.dropout = nn.Dropout(dropout)
    self.elmo_mode = elmo_mode
    if self.elmo_mode:
      assert elmo_option, "Elmo option must be required for Elmo mode"
      self.elmo = ELMo(elmo_option)
      self.elmo_dim = self.elmo.elmo.get_output_dim()

    self.init_weight()

  def init_weight(self):
    for name, param in self.named_parameters():
      if 'weight' in name and 'embedding' in name:
        nn.init.uniform_(param.data, -0.05, 0.05)

  def forward(self,
              x: torch.Tensor,
              elmo_tokens: Optional[SentTokens] = None) -> Tensor:
    """
    elmo mode에 따라 데이터를 임베딩합니다.(nn.Embedding으로 eval loss와 함께 학습)
      True: nn.Embedding + elmo representation
      False: nn.Embedding

    Args:
      x(torch.Tensor): 데이터, shape: (batch_size, seq_len)
      elmo_tokens(Optional[SentTokens]): elmo에 입력할 토큰

    Returns:
      Tensor: 임베딩된 데이터, shape: (batch_size, seq_len, embedding_dim)
    """
    if self.elmo_mode:
      assert elmo_tokens, "Elmo tokens must be provided"
      base_embedding = self._embedding(x)
      base_embedding = base_embedding.permute(1, 0, 2)
      combined_embedding = self.elmo.combine_elmo_representations(
          elmo_tokens, base_embedding)
      combined_embedding = self.dropout(combined_embedding)
      return combined_embedding.permute(1, 0, 2)

    else:
      embedding = self._embedding(x)
      embedding = self.dropout(embedding)
      return embedding


class GloveEmbedding(nn.Module):

  def __init__(self,
               pretrained_embeddings: Tensor,
               dropout: float = 0.3,
               freeze: bool = True,
               elmo_mode: bool = False,
               elmo_option: Optional[str] = None):
    super(GloveEmbedding, self).__init__()
    # pretrained_embeddings: (num_words, embedding_dim)
    self._embedding = nn.Embedding.from_pretrained(pretrained_embeddings,
                                                   freeze=freeze)
    self.dropout = nn.Dropout(dropout)
    self.elmo_mode = elmo_mode
    if self.elmo_mode:
      if elmo_option:
        self.elmo = ELMo(elmo_option)
        self.elmo_dim = self.elmo.elmo.get_output_dim()
      else:
        raise ValueError("elmo option must be provided")

    self.init_weight()

  def init_weight(self):
    for name, param in self.named_parameters():
      if 'weight' in name and 'embedding' in name:
        nn.init.uniform_(param.data, -0.05, 0.05)
      else:
        nn.init.xavier_uniform_(param.data)

  def forward(self, x: Tensor, elmo_tokens: Optional[SentTokens] = None):
    """
    elmo mode에 따라 데이터를 임베딩합니다. (사전 학습된 Glove 사용)
      True: nn.Embedding + elmo representation
      False: nn.Embedding

    Args:
      x(Tensor): 데이터, shape: (batch_size, seq_len)
      elmo_tokens(Optional[SentTokens]): elmo에 입력할 토큰

    Returns:
      Tensor: 임베딩된 데이터, shape: (batch_size, seq_len, embedding_dim)
    """
    if self.elmo_mode:
      assert elmo_tokens is not None, "elmo_tokens must be provided"
      base_embedding = self._embedding(x)
      base_embedding = base_embedding.permute(1, 0, 2)
      combined_embedding = self.elmo.combine_elmo_representations(
          elmo_tokens, base_embedding)
      combined_embedding = self.dropout(combined_embedding)
      return combined_embedding.permute(1, 0, 2)

    else:
      embedding = self._embedding(x)
      embedding = self.dropout(embedding)
      return embedding


class LSTMLayer(nn.Module):

  def __init__(self,
               input_dim: int,
               hidden_dim: int,
               output_dim: int,
               num_layers: int = 2,
               dropout: float = 0.1):
    super(LSTMLayer, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_dim,
                        hidden_dim // 2,
                        num_layers=self.num_layers,
                        batch_first=True,
                        bidirectional=True)
    self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
    self.fc1 = nn.Linear(hidden_dim, output_dim)
    self.dropout = nn.Dropout(dropout)

  def init_weights(self):
    for name, param in self.named_parameters():
      if 'weight' in name:
        if 'lstm' in name:
          for i in range(self.num_layers):
            # LSTM Layer를 순회하며 각각 맞는 weight를 초기화합니다.
            nn.init.xavier_uniform_(getattr(self.lstm, f'weight_ih_l{i}'))
            nn.init.xavier_uniform_(getattr(self.lstm, f'weight_hh_l{i}'))
            if self.lstm.bidirectional:
              nn.init.xavier_uniform_(
                  getattr(self.lstm, f'weight_ih_l{i}_reverse'))
              nn.init.xavier_uniform_(
                  getattr(self.lstm, f'weight_hh_l{i}_reverse'))
        else:
          nn.init.xavier_uniform_(param.data)
      elif 'bias' in name:
        if 'lstm' in name:
          for i in range(self.num_layers):
            nn.init.constant_(getattr(self.lstm, f'bias_ih_l{i}'), 0)
            nn.init.constant_(getattr(self.lstm, f'bias_hh_l{i}'), 0)
            if self.lstm.bidirectional:
              nn.init.constant_(getattr(self.lstm, f'bias_ih_l{i}_reverse'), 0)
              nn.init.constant_(getattr(self.lstm, f'bias_hh_l{i}_reverse'), 0)
        else:
          # 이외의 가중치는 0으로 초기화합니다.
          nn.init.constant_(param.data, 0)

  def forward(self, x: Tensor) -> Tensor:
    """
    LSTM layer를 사용하여 데이터를 학습합니다.
    배치 정규화를 거친 후 mean pooling을 사용합니다.

    Args:
      x(Tensor): 데이터, shape: (batch_size, seq_len, embedding_dim)
    
    Returns:
      Tensor: 학습된 데이터, shape: (batch_size, output_dim)
    """
    lstm_out, (_, _) = self.lstm(x)
    lstm_out = lstm_out.contiguous().view(
        -1, lstm_out.shape[2])  # (batch_size * seq_len, hidden_dim)
    lstm_out = self.batch_norm1(lstm_out)
    lstm_out = lstm_out.view(
        x.size(0), -1, self.hidden_dim)  # (batch_size, seq_len, hidden_dim)

    # mean pooling
    lstm_out = lstm_out.mean(1)  # (batch_size, hidden_dim)

    lstm_out = self.dropout(lstm_out)

    fc_out = self.fc1(lstm_out)

    return fc_out


class NewsClassifier(nn.Module):

  def __init__(self,
               input_dim: int,
               embedding_dim: int,
               hidden_dim: int,
               output_dim: int,
               elmo_mode: bool = False,
               elmo_option: Optional[str] = None):
    super(NewsClassifier, self).__init__()
    self.embedding = NewsEmbedding(input_dim,
                                   embedding_dim,
                                   elmo_mode=elmo_mode,
                                   elmo_option=elmo_option)
    # elmo representation과 결합 후 lstm layer에 입력될 차원을 계산합니다.
    lstm_input_dim = embedding_dim + self.embedding.elmo_dim if elmo_mode else embedding_dim
    self.hidden = LSTMLayer(lstm_input_dim, hidden_dim, output_dim)

  def forward(self,
              x: Tensor,
              elmo_tokens: Optional[SentTokens] = None) -> Tensor:
    embedding = self.embedding(x, elmo_tokens)
    output = self.hidden(embedding)
    return output


class GloveClassifier(nn.Module):

  def __init__(self,
               pretrained_embeddings: Tensor,
               hidden_dim: int,
               output_dim: int,
               freeze: bool = True,
               elmo_mode: bool = False,
               elmo_option: Optional[str] = None):
    super(GloveClassifier, self).__init__()
    self.embedding = GloveEmbedding(pretrained_embeddings,
                                    freeze=freeze,
                                    elmo_mode=elmo_mode,
                                    elmo_option=elmo_option)
    lstm_input_dim = pretrained_embeddings.shape[
        1] + self.embedding.elmo_dim if elmo_mode else pretrained_embeddings.shape[
            1]
    self.hidden = LSTMLayer(lstm_input_dim, hidden_dim, output_dim)

  def forward(self,
              x: Tensor,
              elmo_tokens: Optional[SentTokens] = None) -> Tensor:
    embedding = self.embedding(x, elmo_tokens)
    output = self.hidden(embedding)
    return output
