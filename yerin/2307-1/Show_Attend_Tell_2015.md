# [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
<br>

## 1. 개요
- 2014년 CVPR 논문 [Show and Tell](https://arxiv.org/pdf/1411.4555.pdf)에 attention을 적용하여 image captioning 성능을 향상시킴
- 모델 구조 : encoder-decoder
- 디코더에 attention 추가
<p align="center">
<img src=img/2_Figure1.PNG/>
</p>
<br>

## 2. Network Architecture 
#### 1) Encoder : CNN
- pretrained model인 VGGnet 사용
- 기존 논문에서 fully connected layer 사용한 것과 달리 본 논문에서는 feature vectors(annotation vectors) 사용

  &rarr; 디코더가 특정 위치의 정보만을 가져올 수 있게 함
<br>

#### 2) Decoder : LSTM
<p align="center">
<img src=img/LSTM.PNG// width="500">
</p>
<br>

- 기존 논문에서 이미지는 h<sub>0</sub>(최초의 hidden state vector)를 구할 때에만 직접적으로 사용되고, 이후에는 c<sub>t</sub>(cell state)에 문장과 결합된 정보로 남아 간접적으로 영향을 미치게 됨 
- 본 논문에서는 LSTM에 attention을 적용하여 z<sub>t</sub>(image feature vectore)가 매번 input으로 사용됨
<p align="center">
<img src=img/2_Figure4.PNG// width="700">
</p>
<br>

## 3. Attention Mechanism
<p align="center">
<img src=img/2_Figure2.PNG/>
</p>
<br>

#### 1) Stochastic "Hard" Attention
- 위치정보를 나타내는 s<sub>t</sub>(location variable)를 사용하여 annotation vectors 중에 하나의 위치만을 바라봄
- 확률적(stochastic) 매커니즘으로 sampling을 통해 강화학습으로 학습
<br>

#### 2) Deterministic "Soft" Attention
- 가중치가 반영된 전체 feature map을 참조하므로 비교적 넓은 영역을 바라봄
- backpropagation으로 학습
<br>

## 4. 실험 및 결과
<p align="center">
<img src=img/2_Table1.PNG/>
</p>
<br>
