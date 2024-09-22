from type_hint import SentTokens, Tensor
from allennlp.modules.elmo import Elmo, batch_to_ids
from option.config import Config
import torch


class ELMo:

  def __init__(self, cfg_path: str):
    """
    config로부터 설정을 받아 Elmo 모델을 불러옵니다.

    Args:
      cfg_path (str): config 파일 경로
    """
    self._config = Config(cfg_path)
    self.elmo = Elmo(
        self._config.get('OPTION'),
        self._config.get('WEIGHTS'),
        num_output_representations=self._config.get('NUM_OUT_REP'),
        dropout=self._config.get('DROPOUT'))

  def __call__(self, input_sents: SentTokens) -> Tensor:
    """
    패딩처리된 토큰들을 입력받아 elmo representation으로 변환합니다.

    Args:
      input_sents (SentTokens): 입력 문장들
    
    Returns:
      Tensor: elmo representation
    """
    input_ids = batch_to_ids(input_sents)
    output = self.elmo(input_ids)
    return output['elmo_representations'][0]

  def combine_elmo_representations(self, batch_sents: SentTokens,
                                   base_embeddings: Tensor) -> Tensor:
    """
    기존 임베딩과 elmo representation을 concat을 사용하여 연결합니다.
    
    Args:
      batch_sents (SentTokens): 입력 문장들
      base_embeddings (Tensor): 기존 임베딩
    
    Returns:
      Tensor: 기존 임베딩과 elmo representation을 연결한 임베딩
    """
    device = base_embeddings.device
    elmo_representation = self(batch_sents).to(device)
    assert elmo_representation.shape[:
                                     2] == base_embeddings.shape[:
                                                                 2], "Shape mismatch between ELMo and existing embeddings"
    combined_embeddings = torch.cat((base_embeddings, elmo_representation),
                                    dim=-1)
    return combined_embeddings
