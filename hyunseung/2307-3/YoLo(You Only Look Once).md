# [You Only Look Once: Unified, Real-Time Object Detection)](https://arxiv.org/pdf/1506.02640.pdf)


Key point
**YOLO는 전체 이미지를 입력으로 받아 하나의 Covolutional network를 통해 bounding box의 위치와 점수가 나온다. Two stage가 아닌 one stage detector**
<br>

#### Abstract
* Object detection 문제를 회귀 문제로 구성하여 푼다.
* 다른 실시간 detector보다 약 2배의 mAP를 달성한다.
* Generalized된 특징을 학습한다.



#### Introduction
* 인간은 이미지를 한 눈에 보고 이미지에 어떤 물체가 있는지, 어디에 있는지, 어떻게 상호작용하는지를 바로 알 수 있다.
* DPM, 변형가능한 모델은 이미지를 sliding하면서 윈도우 접근방식으로 분류를 해왔다.


* YOLO는 단일 회귀문제로 재구성하여, 이미지를 한번만 보면 어떤 물체가 있고 어디에 있는지 예측이 가능한다.
* YOLO는 매우 간단하다. (그림 1 참조) 단일 convolution 네트워크는 여러 bounding box에 대한 클래스 확률을 동시에 예측한다. 
* 기존에 비해 아래와 같이 이점이 있다.
1. YOLO is fast : 초당 45fps 가능하고 (기준이 되는 이미지 사이즈 안나와있음;;)
빠른 버전의 YOLO는 150fps 이상으로 실행될 수 있다.  

2. YOLO는 예측할 때 전체 이미지에 대해 추론한다. Fast R-CNN에 비해 백그라운드 오류가 절반 미만이다.

3. YOLO는 객체의 일반화 표현을 학습한다.
기존 모델 DPM 및 R-CNN과 같은 최고의 detector보다 큰 차이를 보인다. 



#### Unified Detection

* YOLO의 통합된 detecting의 특징
1. 이미지를 S x S 그리드로 나눈다.

2. 객체 중심이 그리드 셀에 속하면 해당 그리드 셀이 객체를 감지한다.

3. 각 그리드 셀은 B Bounding box와 그 각 각의 박스들의 confidence 점수를 예측한다.
4. 보통 우리는 그 confidence를 Pr(Object) * IOU(truth)pred로 정의한다.
5. 셀에 어떠한 객체가 없다면, confidence score는 0가 되어야한다. 
6. 그렇지 않다면, confidence 점수가 예측된 상자와 실제 정답 간의 intersection over union(IoU)같아지도록 하고 싶다.
7. 각 각의 바운딩 박스는 5개의 예측(x,y,w,h, and confidence)으로 구성된다.
8. (x, y)는 box의 센터를 의미한다.
9. w = width, h = height
10. confidence prediction은 예측된 박스와 실제 정답 박스와의 IoU를 나타낸다.

11. 각 각의 grid cell은 C conditional 클래스 확률을 예측한다. 
Pr(Classi|Object). Boxes B가 여러개 있음에도 불구하고, 우리는 하나의 셋의 grid하나당 class 확률을 알고 싶어한다. 
12. Pr(Classi|Object) * Pr(Object) * IOUpred = Pr(Classi) * IOUpred는 각각의 박스에 대해서 confidence scores를 주게 된다. 

<p align="center">
<img src=img/Yolo_network.png />
</p>

2.1 Network Design
* Fast 24 콘볼루션 레이어를 사용하지 않고, 9개의 콘볼루션 레이어를 사용함.
* YOLO와 Fast YOLO의 Training and Testing 파라미터 개수는 같다.
* 아래와 같이 24개의 콘볼루션 레이어들과 2개의 FCL을 사용함.
* ImageNet의 pretrained된 콘볼루션 레이어를 사용함. (1000-class competition dataset)


2.2 Training
* 20 convolutional layers, average-pooling, fully connected layer를 사용함.
* 위 레이어에에 4개의 Convolutional layers와, 2개의 fully connected layers를 랜덤 초기화weight된 상태로 초기화함. 
* Detection은 종종 fine-grained visual information을 필요로하기 때문에
우리는 224X224 사이즈에서 448X448로 변경했음.
* Activation function은 leaky rectified linear activation을 사용함.
* optimize를 용이하기 위해 MSE를 사용함.
* 최종 레이어는 class probabilities와 bounding box coordinates를 예측한다. 
* 대다수의 그리드 셀에는 객체가 존재하지 않기 때문에 YOLO가 모든 그리드 셀에서 confidence = 0이라고 예측하도록 학습되게 할 수 있다.
* 객체가 존재하는 바운딩 박스의 confidence loss 가중치를 늘리고, 반대로 객체가 존재하지 않는 바운딩 박스의 confidence loss 가중치를 줄인다. 
* 2가지 파라미터로 조절이 가능하다. λ_coord와 λ_noobj 입니다. 논문에서는 λ_coord = 5, λ_noobj = .5로 설정함.
<br>
* 구조상 문제를 해결을 위해 아래 3가지 개선안을 적용
1) localization loss와 classfication loss 중 localization loss의 가중치를 증가시킨다. 
2) 객체가 없는 그리드 셀의 confidence loss보다 객체가 존재하는 그리드 셀의 confidence 가중치를 증가시킨다.
3) bounding box의 너비(width)와 높이(height)에 square root를 취해준 값을 loss function으로 사용한다. 
* 과적합을 막기위해 dropout과 data augmentation을 적용함.

<p align="center">
<img src=img/Yolo_loss.png />
</p>

* Object가 존재하는 그리드 셀 i의 bounding box predictor j에 대해 x와 y의 loss 계산.
* width, height의 loss를 계산. 큰 box에 대해서는 작은 분산(small deviation)을 반영하기 위해 제곱근을 취한 후, sum-squared error를 구한다.
* Object가 존재하는 그리드 셀 i의 boudnig box predictor j에 대해, confidence score의 los를 계산. (Ci = 1)
* Object가 존재하지 않는 그리드 셀 i의 bounding box predictor j에 대해, confidence score의 loss를 계산(Ci = 0)
* Object가 존재하는 그리드 셀 i에 대해, conditional class probability의 loss rPtks. (p_i(c) = 1 if class c is correct, otherwise: p_i(c)=0)

λ_coord: Coordinates(x,y,w,h)에 대한 loss와 다른 loss들과의 균형을 위한 balancing parameter. 
λ_noobj: 객체가 있는 box와 없는 box 간에 균형을 위한 balancing parameter.
(일반적으로 image내에는 객체가 있는 그리드 셀보다 없는 셀이 훨씬 많다.)

2.3. Inference(추론)
* 파스칼VOC 데이터 셋에 대해서 YOLO는 한 이미지 당 98개의 bounding box를 예측하고, 그 bounding box마다 클래스 확률(class probabilities)를 구해준다.

<br>
* YOLO의 그리드 디자인은 한 가지 단점이 있다. 하나의 객체에 여러 그리드 셀이 동시에 검출하는 경우가 있다는 점. 
* 다중 검출(multiple detections)문제라고 한다. 이런 다중 검출(multiple detections)문제는 non-maximal suppression라는 방법을 통해 개선할 수 있다.
YOLO는 non-maximal suppression을 통해 mAP 2~3%가량 향상시킴. 

2.4 Limitations of YOLO

* YOLO는 하나의 그리드 셀마다 두 개의 bounding box를 예측한다. 
* 하나의 그리드 셀마다 오직 하나의 객체만 검출이 가능하다. 이는 공간적 제약(spatial constraints)를 야기함. 이 뜻은, 하나의 그리드 셀에 두 개 이상의 객체가 붙어있으면 잘 검출 하지 못하는 문제를 말한다. 
*또한, YOLO모델은 큰 bounding box와 작은 bounding box의 loss에 대해 동일한 가중치를 둔다는 단점. 크기가 작은 bounding box는 위치가 조금만 달려져도 성능에 큰 영향을 줄 수 있다. 큰 bounding box에 비해 작은 bounding box가 위치 변화에 따른 IoU변화에 더 심하기 때문이다. 이를 부정확한 localization문제라고 부른다.

요약
- 작은 객체들이 몰려있는 경우 검출을 잘 못한다.
- 훈련 단계에서 학습하지 못한 종횡비(aspect ratio)를 테스트 단계에서 마주치면 고전한다. 
- 큰 boudning box와 작은 bounding box의 loss에 대해 동일한 가중치를 둔다.



#### Comparision to Other Detection Systems
##### Deformable parts models(DPM)
* 객체 검출 모델 중 하나인 DPM은 슬라이딩 윈도우(sliding window)방식을 사용함. DPM은 하나로 연결된 파이프라인이 아니라 서로 분리된 파이프 라인으로 구성되어있다. 
* 독립적인 파이프라인이 각각 특징 추출(feature extraction), 위치파악(region classfication), bounding box 예측 (bounding box prediction)등을 수행. 
* 이에 비해 YOLO는 이렇게 분리된 파이프라인을 하나의 콘볼루션 신경망으로 대채한 모델. 이 신경망의 특징 추출, bounding box 예측, non-maximal suppression 등을 한번에 처리. YOLO는 DPM보다 더 빠르고 정확하다. 

##### R-CNN
* R-CNN은 슬라이딩 윈도 대신 region proposal 방식을 사용하여 객체를 검출하는 모델입니다. selective search라는 방식으로 여러 bounding box를 생성하고, 콘볼루션 신경망으로 feature를 추출하고, SVM으로 bounding box에 대한 점수를 측정한다. 선형 모델(linear model)로 bounding box를 조정하고 non-maximal suppression로 중복된 검출을 제거. 각 단계 별로 독립적으로 튜닝해야하기 때문에 R-CNN은 속도가 굉장히 느리다. 실시간 객체 검출 모델로 사용하기에는 한계가 있다.

#### Experiments
* YOLO를 다른 실시간(real-time) 객체 검출 모델과 비교. 
* YOLO와 Fast R-CNN의 성능의 차이를 비교하기 위해 VOC 2007 데이터 셋에서 에러를 구함. Fast R-CNN은 이 논문이 나온 시점을 기준으로 성능이 가장 좋은 R-CNN 계열의 모델임. 

=> YOLO는 기존 모델에 비해 속도는 월등히 빠르고, 정확도도 꽤 높은 수준이다.

<br>
<p align="center">
<img src=img/Yolo_generalizability.png />
</p>

- YOLO는 훈련 단계에서 접하지 못한 새로운 이미지도 잘 검출한다.

#### Real-Time Detection In the Wild
<br>
<p align="center">
<img src=img/Yolo_realtime.png />
</p>

#### Conclusion
* YOLO는 단순하면서도 빠르고 정확하다. 또한, YOLO는 훈련 단계에서 보지 못한 새로운 이미지에 대해서도 객체를 잘 검출한다. 즉, 새로운 이미지에 대해서도 강건하여 애플리케이션에서도 충분히 활용할만한 가치가 있다.




## References
- 모델 원리 설명: https://dotiromoook.tistory.com/24
- YOLO https://bkshin.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-YOLOYou-Only-Look-Once
