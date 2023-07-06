# [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)
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
- 딥러닝은 raw sensory data로부터 high-level features를 추출함으로써 비전 및 음성인식 분야에서 큰 발전(CNN, multilayer perceptrons, RBM, RNN 등)을 이룸
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
- [Q-learning](https://ko.wikipedia.org/wiki/Q%EB%9F%AC%EB%8B%9D)으로 네트워크를 학습
- Stochastic gradient Descent(SGD) 알고리즘으로 가중치를 업데이트
- [Experience Replay](https://paperswithcode.com/method/experience-replay) : 데이터가 상호 비독립적이며, 데이터 분포가 가변적이라는 문제를 해결하기 위해 고안된 기법. 과거의 transition을 랜덤하게 샘플링하고, 학습 분포를 smooth하게 함
<br>

#### 실험 환경 : Atari 2600 Games
<p align="center">
<img src=img/Figure1.png/ width="850">
</p>
<br>


## 2. Background
#### Agent와 환경 ε의 상호작용
-	agent는 매 time-step마다 취할 수 있는 행동들(A = {1, …, K}) 중에서 한가지(a<sub>t</sub>)를 선택
-	선택에 따라 환경의 내부 상태(vector of raw pixel values)가 수정되고 보상(r<sub>t</sub>)을 받음
-	게임의 점수는 현재의 행동뿐만 아니라 이전에 거쳤던 일련의 행동에 의해 결정되고, 행동에 대한 피드백은 수천 회의 time-step이 진행된 후에 받게 됨
-	agent는 현재의 장면(x<sub>t</sub>)만을 관찰하기 때문에 전체적인 상황을 이해하기 위해 행동들의 sequences(s<sub>t</sub> = x<sub>1</sub>, a<sub>1</sub>, x<sub>2</sub>, ..., a<sub>t-1</sub>, x<sub>t</sub>) 통해 학습 진행
-	매 sequence는 유한 [Markov Decision Process(MDP)](https://ko.wikipedia.org/wiki/%EB%A7%88%EB%A5%B4%EC%BD%94%ED%94%84%EA%B2%B0%EC%A0%95%EA%B3%BC%EC%A0%95)로 표현되므로 s<sub>t</sub>를 통해 MDP에 standard RL method를 적용할 수 있음. 이는 t 시점의 상태를 표현하기 위해 전체 시퀀스를 사용함을 의미
-	agent는 discounted future reward(R<sub>t</sub>) 극대화를 목표로 action을 선택함
<br>

#### 수식
- Future discounted return at time t (T : 게임 종료 시점)
<p align="center">
<img src=img/Future_discounted_return.png/>
</p>
<br>

- Optimal action value function (π : policy mapping sequences to actions)
<p align="center">
<img src=img/Optimal_action_value_function.png/ width="500">
</p>
<br>

- Value iteration algorithms (i : iteration)
<p align="center">
<img src=img/Value_iteration_algorithms.png/ width="500">
</p>
<br>  

- Loss functions (ρ(s, a) : behaviour distribution)
<p align="center">
<img src=img/Loss_functions.png/ width="500">
</p>
<br>

- Differentiating the loss function with respect to the weights
<p align="center">
<img src=img/Differentiating_the_loss_function.png/ width="850">
</p>
<br>

## 3. Related Work
#### TD-gammon
- RL에서 가장 성능이 좋았던 알고리즘 (backgammon 게임을 사람 수준으로 play할 수 있음)
- Q-learning과 유사한 model-free RL 알고리즘
- 1개의 hidden layer를 가진 multi-layer perceptron을 사용하여 value function 근사화
- 체스나 바둑에서는 성능이 좋지 않음
- model-free RL 알고리즘을 non-linear function approximators 또는 off-policy learning과 결합할 경우 Q-network가 발산하기도 함  
  &rarr; 수렴을 위해 후속 RL 연구들은 linear function approximators 위주로 이루어짐
<br>

#### RL과 딥러닝의 결합 시도
- Deep neural networks을 환경 ε를 예측하는 데 사용
- value function 혹은 policy를 구하기 위해 RBM 사용
- Q-learning의 발산 문제 : gradient temporal difference 방법으로 극복
- 그러나 RBM, Q-learning 등의 방법들은 non-linear control에는 확장되지 못함
<br>

#### Neural Fitted Q-learning(NFQ)
- 본 연구(DQN)와 가장 유사한 접근 방식을 지닌 선행 연구
- NFQ는 RPROP 알고리즘을 사용하여 Q-network의 파라미터를 업데이트
- 모든 데이터셋을 한 번에 학습하는 batch update를 이용하기 때문에 연산 부담이 매우 큼  
   &rarr; DQN은 대량의 데이터도 효율적으로 연산할 수 있는 Stochastic Gradient Updates 채택
- NFQ는 오토인코더를 사용하여 visual input의 low dimensional representation을 학습하는 방식  
   &rarr; DQN은 visual input을 바로 사용하는 end-to-end 방식
<br>
 
## 4. Deep Reinforcement Learning
#### Experience replay
-	매 time-step에서 agent의 경험(e<sub>t</sub>)을 데이터셋에 저장  
  &rarr; 수많은 에피소드가 replay memory에 쌓임  
 	&rarr; Q-learning updates(minibatch updates) 적용하여 경험(e)을 샘플링하는 기법
-	Experience replay 적용 이후 agent는 [ϵ-greedy 알고리즘](https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/)에 따라 행동을 선택하고 수행
-	Neural network에 가변 길이 input을 사용하기 어렵기 때문에 본 연구는 고정 길이 history(ϕ(s<sub>t</sub>))를 입력으로 사용
<br>

#### Pseudo code
<p align="center">
<img src=img/Algorithm1.png/>
</p>
<br>

#### 기존 Q-learning과 비교하여 deep Q-learning이 가지는 장점
1) 기존의 방법은 각 step의 Experience를 한 번만 사용  
   &rarr; 잠재적으로 많은 가중치 업데이트에 재사용하므로 훨씬 데이터 효율적임
2) 연속적인 sample들로 학습하는 것은 데이터들 간의 high correlations 때문에 비효율적임  
   &rarr; sample의 random 추출을 통해 업데이트의 효율성을 높임
3) 기존의 on-policy를 학습하면 현재의 parameter들이 다음 데이터 샘플을 결정하므로 의도치 않은 feedback loops가 발생하고, parameter들이 local minimum으로 수렴하거나 발산할 수 있음  
   &rarr; experience replay를 사용하면 parameter의 발산이나 진동을 피하고 학습을 매끄럽게 진행할 수 있음. experience replay로 학습하려면 off-policy를 학습해야 함
<br>

#### Preprocessing and Model Architecture
- Atari games는 128 color palette의 210×160 픽셀 이미지로 구성됨  &rarr; 연산 부담을 줄이기 위해 전처리 필요
1)	RGB 이미지를 gray-scale 이미지로 변환
2)	이미지 사이즈를 110x84 픽셀로 다운샘플링
3)	84x84 픽셀의 정사각형 이미지로 crop
<br>

## 5. Experiments
#### 테스트 환경 : Atari 2600 Games
- 7개 (Beam Rider, Breakout, Enduro, Pong, Q*bert, Sequest, Space Invaders) 게임에 대해 동일한 조건(네트워크 구조, 학습 알고리즘, 하이퍼파라미터 등)으로 테스트
- 보상 구조 : 1(positive reward), -1(negative reward), 0(unchanged)
- 최적화 알고리즘 : RMSProp (minibatch size : 32)
- Behavior policy : ϵ-greedy 알고리즘
- Frame skipping technique : agent는 모든 frame이 아닌 k번째 frame을 보고 action을 선택하므로 이 기법을 적용하면 k배 만큼 더 게임을 play할 수 있음 (연산량이 줄었기 때문). 본 실험에서는 k=4로 설정 (4 배수 프레임에서만 action을 선택). 단 Space Invader 게임에서만 k=3으로 설정
<br>

#### Training and Stability
<p align="center">
<img src=img/Figure2.png/ width="850">
</p>
<br>

-	(왼쪽의 두 그래프) Breakout, Seaquest에서 학습을 하는 동안 total reward가 어떻게 변화하는지 보여줌. 두 그래프는 점진적 개선을 보이지 않으며, 상당히 noisy함
-	(오른쪽의 두 그래프) Policy에 대한 action-value를 예측하는 Q-function은 상당히 안정적인데, 이 함수는 agent가 policy를 따랐을 때 얼마의 discounted reward를 얻을 수 있는지를 나타냄. average predicted Q가 agent를 통해 얻은 average total reward보다 훨씬 smooth하게 증가함
-	Q값이 반드시 수렴한다는 이론적인 검증은 없지만, 본 연구의 방식이 RL과 SGD를 사용하여 neural network를 안정적으로 학습시킴을 알 수 있음
<br>

#### Visualizing the Value Function
<p align="center">
<img src=img/Figure3.png/ width="850">
</p>
<br>

-	Seaquest에서 학습된 value function을 시각화한 것. 본 연구의 방식으로 일련의 복잡한 이벤트들에 대해 어떻게 진화해 나가야 할지 학습할 수 있음을 보임
-	A : 화면의 왼쪽에 적이 등장했을 때 predicted value가 jump함
-	B : 발사된 미사일이 적을 맞추기 직전일 때 predicted value가 최대로 상승함
-	C : 화면에서 적이 사라졌을 때 predicted value가 초기값 수준으로 감소함
<br>

#### Main Evaluation
<p align="center">
<img src=img/Table1.png/>
</p>
<br>

-	ϵ-greedy 알고리즘(ϵ = 0.05) 적용 결과 비교
<br>

## 5.	Conclusion
- 본 연구는 RL을 위한 새로운 딥러닝 모델을 소개하고, input으로 raw pixels만을 사용하여 Atari 2600의 control policies를 학습할 수 있음을 보임
- stochastic minibatch updates와 experience replay memory를 결합한, 변형된 online Q-learning을 제안함
- 본문에서 제시된 방법으로 architecture나 hyperparameters 조정 없이 7개 중 6개 게임에서 SOTA 달성함을 보임
<br>






