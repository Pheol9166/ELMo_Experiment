# ELMo_Experiment

## TOC
- [서론](#서론)
- [실험 설계](#실험-설계)
- [실험 결과](#실험-결과)
- [결론](#결론)

## 서론
ELMo(Embeddings from Language Model)는 2018년에 발표된 Deep Contextualized Word Embeddings(Matthew, 2018)에서 고안된 임베딩 기법이다. 기존에 사용되던 워드 임베딩과는 달리, 문맥 정보를 반영한 word representation을 제공한다. 이는 동음이의어뿐만 아닌 문장에 따라 미묘하게 달라지는 단어의 뜻까지 구분할 수 있다는 점에서 다른 임베딩 기법보다 높은 정확도를 보인다. 이름에서 알 수 있듯, ELMo는 Language Model, 즉 언어 모델로부터 임베딩을 얻는 방식으로 진행된다. 이를 위해 사전학습한 biLM을 사용한다.

- biLM
ELMo의 bi-LSTM은 Forward LM과 Backward LM으로 이루어진다.  다만, 기존의 양방향 RNN과는 다른 점을 보인다. 양방향 RNN의 경우 순방향 RNN과 역방향 RNN의 hidden state를 연결했다면, ELMo의 bi-LSTM은 Forward LM, Backward LM을 별개로 학습한다.
    
    각각의 LM은 다음과 같이 이루어져있다.
    
    <img src=https://github.com/user-attachments/assets/d958d1f2-ef11-423b-9644-ed8e81c324b9 width=500 height=500>

    
    - CharCNN
        
        CharCNN은 CNN을 활용한 임베딩 기법으로 필터를 구성하여 주변 단어를 볼 수 있다는 점은 Word Embedding과 비슷하지만, CNN이기에 연산 속도가 빠르다. 또한 CharCNN은 단어 대신 문자를 기준으로 다양한 크기의 Convolution filter를 적용한다.  Max Pooling이 완료된 후엔 Highway Network를 거쳐 불필요한 연산을 줄인다.
        Highway Network는 데이터가 어떤 레이어를 지나갈 때 수행될 연산을 거치지 않고 지나가는 경로가 있는 구조이다. 학습이 이미 다 이루어졌으면 연산을 거치지 않고 지나갈 수 있어 연산량을 줄일 수 있다.
        
        ![image](https://github.com/user-attachments/assets/12206299-b7a5-448e-83ff-a0644f6f3411)

        
    - Bidirectional LSTM
        
        N개의 sequential token (t<sub>1</sub>, t<sub>2</sub>, … ,t<sub>n</sub>)이 있을 때 Forward LM이 계산하는 확률은 다음과 같다. 특정 시점에서의 토큰 t<sub>k</sub>가 등장할 확률은 t<sub>1</sub>부터 t<sub>k-1</sub>까지의 확률의 곱과도 같다. 
        
        ![image](https://github.com/user-attachments/assets/7663ec5a-41ba-46a8-8ba8-e50e9c6e7461)

        
        입력 위치를 k, 현재 층의 높이를 j라 할 때, Forward LM에서 나온 context-dependent representation 은 ![image](https://github.com/user-attachments/assets/218f7a6e-fa29-466c-993c-9ec52b167887)
과 같이 표현한다. 
        
        biLSTM은 LSTM 층과 층을 사이 Skip connection을 적용한다. Skip connection이란 기울기 소실을 방지하기 위해 이전 층의 입력 정보를 연결하여 계산하는 방법이다. 
        
        ![image](https://github.com/user-attachments/assets/f0a568ec-f6f2-4722-a463-d67aa503a7c5)

        
        Backward LM은 Forward LM과 동일하지만 역방향으로 확률을 계산한다. 
        
       ![image](https://github.com/user-attachments/assets/0fdb82f2-7d40-4c23-8657-a4b90515af17)

        
        마찬가지로 Backward LM에서 나온 context-dependent representation은 ![image](https://github.com/user-attachments/assets/60830fcb-7fe2-4f7a-8a5f-8fdd4aba3b17)
과 같이 표현한다.  최종적으로 biLM의 학습은 Forward LM과 Backward LM을 합친 것의 log likelihood를 최대화하는 방향으로 진행된다.
        
        ![image](https://github.com/user-attachments/assets/92f9099b-a061-4a1d-9a5d-ad27c6986e95)

- ELMo
    
    ELMo는 중간 레이어의 representation들을 task에 맞게 결합한 것을 말한다. 레이어 수를 L이라 할 때, 입력 벡터 ![image](https://github.com/user-attachments/assets/473e2349-742f-4f73-a98b-c9a344d3e330)
을 포함하여 2L + 1개의 representation들을 가중치를 곱한 후 가중합하여 계산한다.
    
    ![image](https://github.com/user-attachments/assets/2e7545ff-b5a0-42be-8f53-30677abc0705)

    
    ![image](https://github.com/user-attachments/assets/6026e98c-dc87-4223-9d57-aa55cc670d30)

    
    ![image](https://github.com/user-attachments/assets/bd93b6bf-58dd-4541-ba3e-436525c10892)

    
    Rk의 경우는 모든 레이어를 representation으로 압축한 것이고, task에 따라 특정 레이어만 선택하거나 변형될 수 있다.
    
    ![image](https://github.com/user-attachments/assets/a744fcef-0a0f-4673-96a3-26939d64ce4c)

    
    특정 task에 따라 계산된 elmo representation은 기존 임베딩 값에 연결(concat)되어 문맥 정보를 함께 넘겨준다.
    

이처럼 ELMo는 사전 학습된 모델을 통해 문맥 정보를 반영하였다는 점에서 강한 성능을 보일 수 있다.  때문에 간단한 성능 비교 실험을 통해 ELMo가 어느 정도의 성능 향상을 보이는지 확인해보기로 하였다.

# 실험 설계
실험은 ELMo를 통해 생성된 elmo representaton이 기존 임베딩 기법과 결합하였을 때의 성능 향상을 측정하는 것이 목적이다. 
때문에 기존 임베딩 기법만 사용했을 때의 평가 척도, 엘모와 결합하여 사용했을 때의 평가 척도를 비교하는 방식으로 실험을 진행하였다. 한 가지 기법만으로 실험을 진행하게 된다면 일반적인 성능 향상의 결과로 받아들이기 어려울 수 있기에, 총 두 가지의 임베딩 기법을 준비하였다.  

- nn.Embedding
    
    nn.Embedding은 토큰화한 단어들을 정수 인코딩한다. 그 후, 정수 인코딩 결과를 기반으로 lookup table을 생성한다. 모델이 손실 함수에 따라 학습하는 과정에서 lookup table의 가중치들도 같이 학습된다.  
    
- GloVe
    
    GloVe는 기존 카운트 기반의 방법론과 예측 기반의 방법론을 절충한 기법이다. 글로브에서 임베딩 된 중심 단어와 주변 단어 벡터의 내적 값이 전체 코퍼스에서의 동시 등장 확률로 이어지도록 한다. 
    

성능 향상 측정을 위해 ELMo가 해당 임베딩들과 결합하였을 때 결과를 ELMo를 사용하지 않았을 때 결과와 비교하고자 한다. 그렇기에 실험에서 다룰 경우는 아래와 같이 총 4가지이다.

- nn.Embedding
- GloVe
- nn.Embedding + ELMo
- GloVe + ELMo

또한 ELMo를 통한 각 임베딩 기법들의 성능 향상만 측정하는 것이 아닌, nn.Embedding + ELMo, GloVe + ELMo를 비교하면서 임베딩 기법 별로 ELMo 성능에 영향이 있는지 확인해보기로 하였다.

실험을 위한 task는 다중 텍스트 분류로 선정하였다. 텍스트를 여러 분류로 구분하기 위해선 텍스트 내 사용된 단어의 문맥적 의미가 중요하게 적용할 것이다. 데이터는 AG News Classification Dataset을 사용했으며, Description 열은 데이터 처리가 일괄적으로 되지 않아 Title 열을 사용하였다.  정리하자면, 입력받은 Title 데이터를 World, Sports, Business, Sci/Tech 중 하나의 주제로 구분하는 것이 실험의 task이다.

구현에 있어서는 pytorch를 사용하였다. 먼저 데이터셋은 AG News Classification의 Title 데이터와 word encoder, elmo_mode를 초기화 파라미터로 받는다. word encoder는 텍스트 데이터를 입력받아 토큰화와 패딩, 인코딩을 진행하는 클래스이다.  elmo_mode에 따라 word encoder에서 처리한 단어 토큰들을 데이터셋에 반환값에 포함한다.

모델은 실험에서 다루는 경우들 모두 LSTM layer는 공통되기 때문에 Embedding Layer, LSTM Layer를 나누어 각각 구현하였다. Embedding Layer에는 elmo_mode라는 boolean argument를 추가하여 그 값에 따라 ELMo와 기존의 임베딩이 결합된다. 더불어 Embedding Layer와 LSTM layer를 연결하는 Classifier 클래스를 만들어 임베딩 결과가  LSTM layer로 이어지고, 카테고리를 분류한다.

학습은 Trainer 클래스를 통해 데이터셋, 모델, 옵티마이저, 손실 함수, learning rate 등을 입력받아 이루어진다. 손실 함수로는 다중 분류에 적합한 Cross Entropy 함수를 사용하였고, 옵티마이저에는 Adam을 사용하였다. 또한, ReduceLROnPlateau를 ****scheduler로 설정하여 성능 향상이 없을 때 learning rate를 감소하게 했다.

## 실험 결과
![image](https://github.com/user-attachments/assets/98973c86-65de-431c-8c06-f0a794603890)
전체적인 결과를 살펴보자면, Base 모델을 사용했을 때보다 ELMo와 함께 결합했을 때가 Accuracy, F1-Score 두 평가지표 모두에서 높은 성능을 보였다. 다만 nn.Embedding의 경우 Accuracy가 0.73%p 증가하였고, GloVe의 경우 1.1%p 증가한 것으로 보아 큰 성능 변화는 없는 것으로 보인다. F1-Score에서도 마찬가지로 증가 폭이 크지 않은 것으로 보아 ELMo 결합에 따른 성능 향상은 존재하나, 상당한 정도는 아니라고 생각된다.


결과를 보면 nn.Embedding이 GloVe보다 전반적인 성능이 좋게 나왔는데, 이는 실험에 사용한 뉴스 데이터의 특성 때문인 것으로 보인다. 뉴스 데이터에는 고유명사, 내지는 특수한 단어들이 주로 사용되는데, 해당 단어들을 정수 인코딩하여 단어 사전에 추가하는 nn.Embedding과 달리, GloVe의 경우 사전에 만들어진 단어 사전을 사용한다. 이로 인해 GloVe는 해당 단어들을 <UNK> 토큰으로 처리하여 예측 정확도에 차질이 생기는 것이다. 실제로 GloVe를 사용한 임베딩 레이어 학습 시 freeze를 False로 하여 가중치 업데이트를 진행했을 때, Accuracy 86.47%, F1-Score 0.86으로 nn.Embedding 모델과 유사하게 나온 것을 확인할 수 있다. 

## 결론
실험 결과는 기존의 논문과는 다르게 큰 성능 향상을 보이고 있지는 않다. 이는 실험 자체의 한계에서 기인한 것으로 보이는데, 이는 다음과 같다.

- 사전 학습 모델의 한계
    
    ELMo 역시 사전에 학습된 모델을 사용하는 것으로, GloVe와 같이 고유명사, 특수한 단어를 처리하기에 적합하지 않을 수도 있다. 학습을 진행하기 전, ELMo에 Title 문장들을 학습한다면 더 높은 성능 향상이 이루어져 있을 지도 모른다.
    
- 부족한 데이터셋
    
    ELMo는 문장이 입력되었을 때, 문맥적 의미를 파악할 수 있도록 도와주는 모델이다. 그런 점에서 뉴스 데이터 중 Title 데이터를 선택한 것은 ELMo의 성능을 최대한으로 발휘하기 힘든 선택이었을 수 있다. Title 데이터의 경우 완전한 문장으로 이루어지지 않았고, 중간중간 전처리되지 못한 데이터들도 있을 수 있기에 좋은 성능을 내기에 적합하지 않았을 수 있다.
    
- 하이퍼파라미터, 모델 설계의 문제
    
    정확한 비교를 위해 네 가지 경우의 모델들의 하이퍼파라미터를 모두 동일한 조건에서 진행하였다. embedding dimension의 경우 GloVe 모델 로드를 위해 GloVe 모델에 따른 차원 값으로 설정이 되어있기 때문에 다른 모델들에게는 최적화된 결과가 아닐지도 모른다. 또한, Hidden layer의 설계, 기존 Embedding layer와 elmo representaion의 결합 알고리즘 등을 바꾸는 것을 통해 성능 향상을 도모해볼 수도 있겠다.
    

ELMo는 사전 학습된 모델을 사용하여 문맥을 고려한 임베딩을 시도했다는 점에서 후에 나올 Seq2Seq, Attention, Transformer 등에 영향을 끼쳤다. 비록, 해당 실험에서는 높은 성능 향상을 이루지 못했지만 ELMo는 자연어 처리 분야에서 문맥적 정보를 활용하는 새로운 패러다임을 제시했다는 점에서 의의가 있다.
