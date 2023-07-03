# [Latent Diffusion](https://arxiv.org/pdf/2112.10752)

Key point
**기존 Diffusion model에 비해 계산량을 현저히 줄인 모델**
- denosing 과정에서 auto encoder사용
- 픽셀공간이 아닌 latent space에서 denoising을 진행하면서 computing cost 감소
- cross-attention을 사용함으로써 텍스트, 오디오 등과 같은 다른 도메인을 함께 사용 가능

<br>

## Architecture

<p align="center">
<img src=img/modelfigure.png />
</p>
<br>


#### Method
- 픽셀 공간에서 이미 학습된 diffusion model의 분석으로 시작. 

1. Latent space(Low dimensional)
Likelihood-based generative model로서, High Dimensional pixel보다 Low Dimension **Latent Space**에서의 연산이 훨씬 유리하다.

2. Underlying Unet
Inductive bias를 활용할 수 있다. **Time-conditional Unet**

Latent Diffusion모델은 추론능력 상 이전 모델에 비해 **sementic한 포인트들**에 좀 더 집중이 가능하다. 

<br>

#### VAE(Variational auto encoder)
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

<br>
<br>

<br>

#### Cross-Attention Mechanism
<p align="center">
<img src=img/Cross-attention.png />
</p>

> Latent diffusion 모델은 zT 노이즈가 오염된 이미지이고, 쿼리값으로 들어가, 텍스트와 embedding을 key값으로 가져간다.
두 개의 가중치를 Dot product(결과는 스칼라) 계산하고, softmax로 계산한다. 다시 텍스트 이미지에 Dot product연산하는 cross attention mechanism을 사용한다.

<br>

#### Loss

##### 1. Perceptual loss
<p align="center">
<img src=img/Perceptual-Loss.png />
</p>
<br>

- Style과 Context를 보존하고자 사용하는 loss이다.<br><br>
- y와 context target yc를 통과시켜 loss를 구하고, Feature Reconstruction Loss 계산<br>
    - Context는 VGG-16을 통과시켜 Feature를 구한다.<br><br>
- y와 style target(sementic한 의미)ys를 통과시켜서 Style Reconstruction Loss를 계산한다.<br>
    - Gram matrix는 Context의 가로축, 세로축 채널과 공통적으로 발견되는지 그 연관성 정도를 나타낸다.<br>


<br>
<br>

##### 2. A Patch-based adversarial objective
<p align="center">
<img src=img/Patch-GAN.jpeg />
</p>

- 전체 이미지를 한번에 T/F를 통해서 score를 측정한 것이 아닌,<br>
Patch 단위로 T/F를 판별하는 방식이다. 
- 이 방식을 사용하게 되면 지역적인 사실성을 살릴 수 있고 L1이나 L2 loss 처럼 pixel 단위 loss를 사용했을 때 나타날 수 있는 blurriness 현상을 완화시킬 수 있다.
- sliding window가 지나가며 연산을 수행하므로 파라미터 개수가 훨씬 작아진다. 
- low frequency에 대해서는 L1 Regularization term과 local 영역에서는 High frequency 영역(엣지)에 대해 patch 단위로 학습함으로서 두 방식의 장점을 모두 취할 수 있음.  
<br>


<br>

#### 실습

##### [Github코드](https://github.com/CompVis/latent-diffusion)

<br>

##### A virus monster is playing guitar
<p align="center">
<img src=img/a-virus-monster-is-playing-guitar,-oil-on-canvas.png/>
</p>
<br><br>


##### River and mountain
<p align="center">
<img src=img/river-and-mountain.png/>
</p>
<br><br>

##### Many students are studying together and watching a notebook
<p align="center">
<img src=img/Many-students-are-studying-together-and-watching-a-notebook.png/>
</p>
<br><br>



## References
- Latent Diffusion: https://velog.io/@yeonheedong/RUS-High-Resolution-Image-Synthesis-with-Latent-Diffusion-Models

- VAE(Variational auto encoder): https://process-mining.tistory.com/161

- Perceptual loss: https://memesoo99.tistory.com/58

- Adversarial Patch: https://pangguinland.tistory.com/169

- Patch GAN: https://brstar96.github.io/devlog/mldlstudy/2019-05-13-what-is-patchgan-D/