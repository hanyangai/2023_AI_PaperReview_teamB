# [Playing Atari with Deep Reinforcement Learning]( https://arxiv.org/pdf/1312.5602.pdf)
<br>

## 0. Abstract
-	Reinforcement Learning(이하 RL)을 통해 high-dimensional sensory input으로부터 직접 control policies를 학습한 첫 딥러닝 모델
-	모델 구조 : CNN + Q-learning 변형 알고리즘
-	Input : raw pixels, Output : 미래의 보상을 예측하는 value function
-	Atari 2600 Games 중 7개의 게임에 동일한 알고리즘 적용
-	결과 : 6개의 게임에서 기존 접근법보다 높은 점수, 3개 게임에서 전문 플레이어보다 높은 점수 기록
<br>

## 1. Introduction
#### 문제 제기 : 기존 RL의 난제, 딥러닝을 통한 해결 가능성 모색
- high-dimensional sensory input(vison, speech 등)으로부터 직접 agents를 control하는 것을 학습하는 것은 RL의 오랜 과제였음
- 기존 RL 연구는 hand-crafted features에 의존적으로, feature representation이 얼마나 잘 되었는지가 중요했음
- 딥러닝은 raw sensory data로부터 high-level features를 추출함으로써 비전 및 음성인식 분야에서 큰 발전(CNN, multilayer perceptrons, restricted Boltzmann machines, RNN 등)을 이룸
<br>

#### RL에 딥러닝 기법을 적용할 경우의 문제점
1) 딥러닝에는 input과 target 사이에 직접적인 연관성이 존재함  
  RL은 scalar reward signal로부터 학습하는데, signal은 대개 듬성듬성(sparse)하고 노이즈가 많으며 (행동과 보상 사이에)딜레이가 있음  
  &rarr; 직접적인 연관성을 찾기 어려움
3) 딥러닝의 데이터는 상호독립적  
  RL은 연관성이 높은 sequences로 나타남
4) 딥러닝은 데이터가 고정된 분포를 갖는다고 가정  
  RL은 알고리즘이 새로운 행동을 학습함에 따라 데이터 분포가 달라짐
<br>

#### 극복 방법 : CNN
- Q-learning 알고리즘으로 네트워크를 학습
- Stochastic gradient Descent(SGD) 알고리즘으로 가중치를 업데이트
- Experience Replay : 데이터가 상호 비독립적이며, 데이터 분포가 가변적이라는 문제를 해결하기 위해 고안된 기법. 과거의 transition을 랜덤하게 샘플링하고, 학습 분포를 smooth하게 함
<br>

#### 실험 환경 : Atari 2600 Games
<p align="center">
<img src=img/Figure1.png/>
</p>
<br>


## 2. Background
#### Agent와 환경 ε의 상호작용
-	agent는 매 time-step마다 취할 수 있는 행동들(A = {1, …, K}) 중에서 한가지(a<sub>t</sub>)를 선택
-	선택에 따라 환경의 내부 상태(vector of raw pixel values)가 수정되고 보상(r<sub>t</sub>)을 받음
-	게임의 점수는 현재의 행동뿐만 아니라 이전에 거쳤던 일련의 행동에 의해 결정되고, 행동에 대한 피드백은 수천 회의 time-step이 진행된 후에 받게 됨
-	agent는 현재의 장면(x<sub>t</sub>)만을 관찰하기 때문에 전체적인 상황을 이해하기 위해 행동들의 sequences(s<sub>t</sub> = x<sub>1</sub>, a<sub>1</sub>, x<sub>2</sub>, ..., a<sub>t-1</sub>, x<sub>t</sub>) 통해 학습 진행
-	매 sequence는 유한 Markov Decision Process(MDP)로 표현되므로 s<sub>t</sub>를 통해 MDP에 standard RL method를 적용할 수 있음. 이는 t 시점의 상태를 표현하기 위해 전체 시퀀스를 사용함을 의미
-	agent는 discounted future reward(R<sub>t</sub>) 극대화를 목표로 action을 선택함

