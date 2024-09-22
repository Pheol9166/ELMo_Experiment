from type_hint import Model, Optim
import torch
import numpy as np


class EarlyStopping:

  def __init__(self,
               path: str,
               patience: int = 5,
               verbose: bool = False,
               delta: float = 0.0):
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.delta = delta
    self.val_loss_min = np.Inf
    self.path = path

  def __call__(self, epoch: int, model: Model, optim: Optim, val_loss: float,
               acc: float):
    """
    Early stopping의 동작 알고리즘입니다. 만약 best score + delta보다 score가 작으면 카운트를 하고 patience에 이르면 학습이 조기 종료됩니다.
    best score가 경신될 때마다 모델과 옵티마이저의 state를 저장합니다.

    Args:
      epoch(int): 현재 epoch
      model(Model): 모델
      optim(Optim): 옵티마이저
      val_loss(float): 현재 eval loss
      acc(float): 현재 eval accuracy
    """

    score = -val_loss
    if self.best_score is None:
      self.best_score = score
      self.save_checkpoint(epoch, model, optim, val_loss, acc)
    elif score < self.best_score + self.delta:
      self.counter += 1
      print(f"Early Stopping counter: {self.counter} out of {self.patience}")
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_score = score
      self.save_checkpoint(epoch, model, optim, val_loss, acc)
      self.counter = 0

  def save_checkpoint(self, epoch: int, model: Model, optim: Optim,
                      val_loss: float, acc: float):
    """
    best score가 경신될 시 checkpoint를 만들고 모델, 옵티마이저, 손실, 정확도를 저장하는 함수입니다.

    Args:
      epoch(int): 현재 epoch
      model(Model): 모델
      optim(Optim): 옵티마이저
      val_loss(float): 현재 eval loss 
      acc(float): 현재 eval accuracy
    """
    if self.verbose:
      print(
          f"Validation loss decreased ({self.val_loss_min: 6f} --> {val_loss: 6f}). Saving model ..."
      )
    torch.save(
        {
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'loss': val_loss,
            'acc': acc
        }, f"{self.path}/checkpoint_model_epoch{epoch}_{val_loss}_{acc}%.pth")
    self.val_loss_min = val_loss
