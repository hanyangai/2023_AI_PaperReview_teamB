# [VAE(Variational auto encoder)](https://arxiv.org/pdf/1312.6114)


Key point
**input값을 latent space인 저차원으로 압축 후 다시 고차원으로 돌려 복원한다.**
<br>
<p align="center">
<img src=img/VAE.png />
</p>

- VAE는 위 그림과 같이 encoder, decoder, 그리고 latent space로 구성된 구조를 가진다. VAE 학습 단계에서 이 encoder와 decoder의 parameter를 학습하는 것을 목적으로 한다. 
#### Encoder
Input을 latent space로 변환한다. Input x가 주어졌을 때 latent vector z의 분포, 즉 q(z|x)를 approximate하는 것을 목적으로 한다. 
예를 들어, q(z|x)를 잘 나타내는 분포로 정규분포(예시)를 선택한다면, q(z|x)를 approximate할 때 이 정규 분포를 잘 나타내는 **평균(mu)** 과  **표준편차(sigma)** 파라미터를 찾는 것


#### Decoder
encoder와 반대로 latent space를 input으로 변환한다. latent vector z가 주어졌을 때 x의 분포 p(z|x)를 approximate하는 것을 목적으로 한다. z의 vector가 주어짐에 따라 데이터 x를 생성하므로, decoder가 generative model의 역할을 한다. 


#### Latent Space
Latent space는 숨겨진 vector들을 뜻한다. input이 들어오면 output을 똑같이 만들 수 있는 latent space를 만들 수 있지만, 이를 방지하기 위해 noise를 sampling하여 latent space를 만든다. 

Ex) 표준 정규분포(평군 0, 표준편차가 1인 정규분포)로부터 하나의 Noise epsilon을 샘플링하여 얻고, encoder로 얻는 분산을 곱하고 평균을 더해서 latent vector 를 얻는다. 이를 reparametrization trick이라고 한다.

#### VAE loss를 최소화 해야하는 파라미터
p theta(x)를 maximize하는 theta를 찾는 것을 목적으로 한다.
VAE는 likelihood maximize하는 방향으로 학습한다.
이는 cross-entropy와 같은 방식으로 표현할 수 있다.



## References
VAE(Variational auto encoder): https://process-mining.tistory.com/161