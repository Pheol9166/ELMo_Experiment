from typing import Tuple
from type_hint import Model, Optim
import pandas as pd
import torch


def load_data(path: str,
              data_type: str = 'csv',
              sep: str = ',',
              encoding: str = 'utf-8') -> pd.DataFrame:
    """
    파일로부터 데이터프레임을 불러옵니다.

    Args:
        path (str): 불러올 파일 경로
        data_type (str): 데이터 형식. 기본값은 csv
        sep (str): 구분자. 기본값은 ','
        encoding (str): 인코딩 방식. 기본값은 'utf-8'
    
    Returns:
        pd.DataFrame: 불러온 데이터프레임
    """
    if data_type == 'csv':
        df = pd.read_csv(path, encoding=encoding, sep=sep)
    elif data_type == 'excel':
        df = pd.read_excel(path, encoding=encoding)
    elif data_type == 'table':
        df = pd.read_table(path, encoding=encoding, sep=sep)
    else:
        df = None
        print("Invalid data type specified.")
    return df


def get_info(df: pd.DataFrame, option: str = 'both') -> None:
    """
    데이터프레임의 정보를 불러옵니다.

    Args:
        df (pd.DataFrame): 정보를 불러올 데이터프레임
        option (str): 불러올 방법. info와 describe, both로 되어있고, 기본값은 both.

    Raises:
        ValueError: option값이 없을 떄 발생
    """
    if option == 'info':
        df.info()
    elif option == 'describe':
        df.describe()
    elif option == 'both':
        df.info()
        print()
        print(df.describe())
    else:
        raise ValueError("It is wrong option value")

    nulls = df.isnull().sum()
    print("Missing values sum per column:")
    for col, missing_count in zip(nulls.index, nulls.values):
        if missing_count:
            print(f"Column {col}: {missing_count}")


def load_model(model_path: str, model: Model,
               optim: Optim) -> Tuple[Model, Optim]:
    """
    파일 경로로부터 모델을 불러옵니다.

    Args:
        model_path (str): 불러올 경로
        model (Model): 덮어쓸 모델
        optim (Optim): 덮어쓸 옵티마이저

    Returns:
        Tuple[Module, Optim]
    """
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])

    return model, optim
