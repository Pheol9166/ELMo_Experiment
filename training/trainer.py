from typing import Callable, Tuple
from type_hint import Tensor, Model, Optim
from tqdm import tqdm
from training.early_stopping import EarlyStopping
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import math
import torch
import matplotlib.pyplot as plt
import numpy as np


class Trainer:

    def __init__(self,
                 model: Model,
                 criterion: Callable[[Tensor, Tensor], Tensor],
                 optimizer: Optim,
                 scheduler: _LRScheduler,
                 device: torch.device,
                 early_stopping: EarlyStopping,
                 elmo_mode: bool = False):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.elmo_mode = elmo_mode
        self.early_stopping = early_stopping
        self.loss_list = []
        self.metric_list = []

    def fit(self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int = 10) -> None:
        """
        학습을 진행하는 함수입니다. epoch 별로 train, eval loss, accuracy, f1-score를 기록하고 early stopping을 수행합니다.

        Args:
            train_loader (DataLoader): 훈련 데이터로더
            test_loader (DataLoader): 평가 데이터로더
            epochs (Optional[int]): 학습 횟수. 기본값은 10
        """
        assert train_loader and test_loader, "train_loader and test_loader must be provided"
        for epoch in tqdm(range(epochs)):
            print(f"\nEpoch {epoch + 1}:")
            train_loss = self.train(train_loader)
            test_loss, accuracy, score = self.eval(test_loader)
            self.loss_list.append((train_loss, test_loss))
            self.metric_list.append((accuracy, score))
            self.early_stopping(epoch, self.model, self.optimizer, test_loss,
                                accuracy)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            self.scheduler.step(test_loss)

    def train(self, dataloader: DataLoader) -> float:
        """
        모델에 데이터를 훈련하는 함수입니다. elmo mode인 경우 elmo representation을 위한 token을 train dataloader에서 가져옵니다.

        Args:
         dataloader(DataLoader): 훈련 데이터로더

        Returns:
            float: train loss
        """
        self.model.train()
        total_loss = 0
        if self.elmo_mode:
            for inputs, labels, tokens in tqdm(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs, tokens)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        else:
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        train_loss = total_loss / len(dataloader)
        print(
            f"Train Loss {train_loss:.4f}, Train PPL: {math.exp(train_loss):7.3f}"
        )

        return train_loss

    def eval(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        모델을 평가하는 함수입니다. elmo mode인 경우 elmo representation을 위한 token을 eval dataloader에서 가져옵니다.

        Args:
            dataloader(DataLoader): 평가 데이터로더

        Returns:
            Tuple[float, float, float]: eval loss, accuracy, f1-score
        """
        test_loss = 0.0
        correct = 0
        total = 0
        prediction_lst = []
        label_lst = []

        self.model.eval()
        with torch.no_grad():
            if self.elmo_mode:
                for inputs, labels, tokens in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(
                        self.device)
                    outputs = self.model(inputs, tokens)
                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            else:
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(
                        self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    prediction_lst.extend(predicted.cpu().numpy())
                    label_lst.extend(labels.cpu().numpy())

            test_loss = test_loss / len(dataloader)
            accuracy = 100 * correct / total
            prediction_lst = np.array(prediction_lst)
            label_lst = np.array(label_lst)
            score = f1_score(label_lst, prediction_lst, average='macro')
            print(
                f"Eval Loss {test_loss:.4f}, Eval PPL: {math.exp(test_loss):7.3f}"
            )
            print(
                f"Accuracy: {accuracy:.2f}%, Err rate: {(100 - accuracy):.2f}%"
            )
            print(f"F1-score: {score:.2f}")

            return test_loss, accuracy, score

    def loss_graph(self):
        if self.loss_list:
            # train, eval loss 분리
            train_list = [x[0] for x in self.loss_list]
            eval_list = [x[1] for x in self.loss_list]
            plt.plot(train_list, label='Train loss', color='blue')
            plt.plot(eval_list, label='Eval loss', color='orange')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Graph')
            plt.show()

    def metric_graph(self):
        if self.metric_list:
            # accuracy, f1-score 분리
            acc_list = [x[0] for x in self.metric_list]
            score_list = [x[1] for x in self.metric_list]

            plt.subplot(2, 1, 1)
            plt.plot(acc_list,
                     label='Accuracy',
                     color='green',
                     linestlye='--',
                     marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Graph')

            plt.subplot(2, 1, 2)
            plt.plot(score_list,
                     label='F1-score',
                     color='red',
                     linestyle='-',
                     marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('F1-score')
            plt.title('F1-score Graph')

            plt.tight_layout()
            plt.show()
