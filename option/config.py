from typing import Any
from type_hint import Json
import json


def load_json(path: str) -> Json:
  with open(path, 'r') as f:
    data = json.load(f)
  return data


class Config:

  def __init__(self, config_path: str) -> None:
    self.path = config_path
    self._config = load_json(config_path)

  def get(self, key: str, default: Any = None, sep: str = '.') -> Any:
    """
    config로부터 값을 얻습니다. 계층 구분은 sep으로 구분하고 기본값은 '.'입니다.

    Args:
      key(str): config에서 얻고자 하는 값
      default(Any): 값이 존재하지 않을 시 반환될 값, 기본값은 None
      sep(str): 구분자, 기본값은 '.'
    """
    keys = key.split(sep)
    value = self._config
    for k in keys:
      value = value.get(k, default)
      if value == default:
        break
    return value

  def save_json(self):
    """
    현재 config를 json 파일로 저장합니다.
    """
    with open(self.path, 'w') as f:
      json.dump(self._config, f)
