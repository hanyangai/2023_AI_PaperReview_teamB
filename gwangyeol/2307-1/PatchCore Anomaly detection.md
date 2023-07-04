# **Anomaly detection**

# **[22′ CVPR] PatchCore : Towards Total Recall in Industrial Anomaly Detection**

1 . Introduction (1분)

- Anomaly detection이란?
    - Anomaly detection 방식은 어떤 데이터 안에서 측정된 값들이 정상 데이터가 가진 특징과 다른 결과가 나왔을 경우를 비정상 데이터(anomaly data)로 탐지하는 방식
    - 이상치 탐지 안에는 supevised, unsupervised, semi-supervised 방식이 존재
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/15b13f2d-46f9-48d8-af8a-90d3f135a8f6/Untitled.png)
        
- Anomaly detection vs Anomaly segmentation
    - detection 같은 경우, input image 내 이상치 포함 여부에 따라 이상치 탐지
    - segmentation 같은 경우, input image 내, pixel-level에 따라 이상치 탐지
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/720ffd72-bd2b-4498-b56b-02f8297fd072/Untitled.png)
        
- PatchCore 방식 Anomaly detection이란?
    - imageNet 기반 Pretrain model을 적극 활용하여 학습 과정 없이, 양품 데이터셋의 feature 추출 하여, anomaly image를 탐지 하는 방식의 모델
    - 양품 데이터셋 전체를 저장하는 방식이 아닌 coreset 내 subsampling 과정을 통해 효과적으로 저장하며, 적은 양의 양품 데이터만으로도 높은 성능과 inference time을 감소
    - keyword - coreset subsampling(greedy search), faiss****,**** memory bank,
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7fcf668f-181b-4e63-8ad8-6494176bef0a/Untitled.png)
        

2 . Related Works & Method (2분)

- Anomaly Detection 방법론
    - densitiy-based method : 정상 데이터의 분포를 통해 비정상 데이터를 탐지
        - pretrained network를 사용하는 방식이며, patchcore 이전에 spade, padim 방식이 존재
        (PatchCore - [SPADE](https://arxiv.org/abs/2005.02357), [PaDim](https://arxiv.org/abs/2011.08785) 방식을 이용한 task)
            - SPADE - locally aware 하지 않은 단점
            - PaDim - 동일 위치의 patch featrue끼리만 비교하므로, alignment가 맞지 않을 경우,
                           anomaly에 대한 판단 불가
    - classification-based method : proxy task를 정의하여 사전 학습 모델을 통해 one-class classification을 적용하여 비정상 데이터 탐지
    - Recontruction-based method : 정상 데이터만을 복원하도록 학습하여 비정상 데이터 inference 시,
    recontruction된 결과와 차이로 비정상 데이터 탐지
    
    ![**자료 출처 - 고려대학교 산업경영공학부 DSBA 연구실** ](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5dad4de0-741c-4625-b8d1-ae01877d8a5b/Untitled.png)
    
    **자료 출처 - 고려대학교 산업경영공학부 DSBA 연구실** 
    
- PatchCore Model Structure
    - Model task
        1. Pretrained network - feature를 추출
        2. 추출된 feature들의 locally aware patch feature로 만듦
        3. coreset sampling을 통해 memory bank에 추가 될 patch feature를 선별 후, 추가
        4. memory bank에 있는 feature들을 하나씩 꺼내, test 단계에서 anomaly socre를 통해 판정
    - Training
        - locally aware patch features
            - Mid-level feature를 통한 local patch feature 사용
            -. 이전 anomaly detection 같은 경우, high level feature를 사용하였으나,
               space data loss, pretrain bias의 빈도가 높음
            -. patchcore에서는 위치 정보 및 imagenet bias의 비중을 낮추기 위해,
               mid level feature를 사용
            -. patchcore는 intermediate feature hierarchies i,  i+1만을 사용하는 이유는
                user feature의 generality를 방지하고, spatial resolution을 보존하기 위함
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/88ea7340-e2ac-4550-b85f-c08b9120e465/Untitled.png)
                
                ![자료 출처 -https://pdfs.semanticscholar.org/3154/d217c6fca87aedc99f47bdd6ed9b2be47c0c.pdf?ref=dataroots.ghost.io](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ba74bf06-ca86-49c5-a4e5-3221fb2e14c8/Untitled.png)
                
                자료 출처 -https://pdfs.semanticscholar.org/3154/d217c6fca87aedc99f47bdd6ed9b2be47c0c.pdf?ref=dataroots.ghost.io
                
        - corset subsampling
    - PatchCore
        - memory bank
            - patchcore 방식은 k nearest neighbor 방식으로 anomaly score를 계산 하기에, memory bank 안에 feature들 많으면 많을수록 computation 하는 과정이 오래 걸린다.
            (spade 방식이 추출된 feature들을 memory 안에 모두 저장, inference time, 연산량 이슈)
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a3b14668-0e8e-4088-8eff-50c9fd5bb4f0/Untitled.png)
                
                ![왼 - memory bank feature, 오 - 군집 중, 대표값을 표시(형광색)](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3a367646-a04f-4ebd-a12c-6669a6d7f6f0/Untitled.png)
                
                왼 - memory bank feature, 오 - 군집 중, 대표값을 표시(형광색)
                
            - 최초 sample을 고르고, 이후 patch feature들 중 가장 멀리 떨어져 있는 patch feature를 선택하는 과정을 반복 하여, 군집 데이터들의 대표값들을 선정
            - uniform distribution test는 아래 그림과 같이 greedy search algorithm이 중복 없이 sampling 되어 있는 것을 볼 수 있음.
                
                ![random, greedy search 방식](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/19d361b9-7314-4f92-8c77-77e499568499/Untitled.png)
                
                random, greedy search 방식
                
    - Testing
        - detection and segmentation(localization)
            - 
        - anomaly score
            - image level - patch들 중 가장 큰 anomaly score를 image에 대한 anomaly score
            - pixel level - local patch에 대한 resolution을 source(원본) Image size로 interpolation 후 gaussian smoothing을 적용
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4a8470b0-df25-493f-851e-17a62e5d8337/Untitled.png)
            
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7fcf668f-181b-4e63-8ad8-6494176bef0a/Untitled.png)
        

3 . Method (3분) - 2번이랑 통합?

- 수도 코드에서 보여주는 알고리즘에 대한 설명
- patchcore에서 새로이 제안된 방식 (앞서 있었던 anomaly detection 논문들과의 차이? 성능 비교? 등…)

4 . Experiments (4분)

- 각 백본마다의 성능 비교
    - Backbone의 classification 성능에 따라 모든 지표가 개선되지 않음
    - 3번째 layer의 local patch features를 사용하는 것이 가장 좋음

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b84035a8-2b57-47df-b760-925881271bbb/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dcde47b8-f254-478e-a38c-2964df0621f6/Untitled.png)

- coreset 인퍼런스 타임

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5c873b20-5f49-4040-96c2-1027063de28a/Untitled.png)

- coreset sampling 방법론 성능
- 이전 sota WideResNet50을 backbone으로 사용한 PaDiM이였으나, patchcore 등장이후 바뀜
    - 적은 데이터 수에도 이전 방법론보다 높은 성능을 보임
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/81950af2-8664-445a-85d0-172d36bd2d1f/Untitled.png)
        
    - patcch core에서 coreset sampling을 진행해도 비교적 성능 하락이 적으며, 초기엔 오히려 성능이
    잠시 오르는 부분을 아래 그래프에서 확인 가능 (이전 random, learned 방식보다 우수한 성능)
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d1d05168-cf7d-4937-ba8b-48bf08445c6c/Untitled.png)
        

5 . Conclusion

- patchcore는 density base method 방식이며, 이전 sota anomaly detection 방식보다 가볍고 우수한 성능이 보장된다.
- feature extract를 다양한 방법론으로 접근 하거나, image alingment 방식을 개선할 새로운 방법론이 더 나오면 산업 분야에 더 우수한 성능? General하게 적용 가능 할 것 같습니다.

6 . Q&A - END (~)
