[22′CVPR] PatchCore : Towards Total Recall in Industrial Anomaly Detection**

> [arxiv Link](https://arxiv.org/pdf/2106.08265.pdf) - PatchCore Anomaly Detection**
> 

---

1 . Introduction

- Anomaly detection ??
    - Anomaly detection 방식은 어떤 데이터 안에서 측정된 값들이 정상 데이터가 가진 특징과 다른 결과가 나왔을 경우를 비정상 데이터(anomaly data)로 탐지하는 방식
    - 이상치 탐지 방식은 supervised, unsupervised, semi-supervised(one-class) 방식이 존재
        
        <img width="676" alt="1" src="https://github.com/hanyangai/2023_AI_PaperReview_teamB/assets/90014998/7f503f3a-3072-4630-89d5-78768fb13bc7">

        
- Anomaly detection & Anomaly segmentation(localization)
    - detection 같은 경우, input image 내 이상치 포함 여부에 따라 이상치 탐지
    - segmentation 같은 경우, input image 내, pixel-level에 따라 이상치 탐지
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/720ffd72-bd2b-4498-b56b-02f8297fd072/Untitled.png)
        
- PatchCore 방식 Anomaly detection이란?
    - imageNet 기반 Pretrain model을 적극 활용하여 학습 과정 없이, 데이터셋의 feature 추출 하여, anomaly image를 탐지 하는 방식의 모델
    - 양품 데이터셋 전체를 저장하는 방식이 아닌 coreset 내 subsampling 과정을 통해 효과적으로 저장하며, 적은 양의 양품 데이터만으로도 높은 성능과 
    inference time(linear)을 감소
    - **keyword - coreset subsampling(greedy search), faiss, memory bank…**
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7fcf668f-181b-4e63-8ad8-6494176bef0a/Untitled.png)
        

2 . Related Works & Method

- Anomaly Detection 방법론
    - densitiy-based method : 정상 데이터의 분포를 통해 비정상 데이터를 탐지
        - pretrained network를 사용하는 방식이며, patchcore 이전에 spade, padim 방식이 존재
        (PatchCore - [SPADE](https://arxiv.org/abs/2005.02357), [PaDim](https://arxiv.org/abs/2011.08785) 방식을 이용한 task)
            - SPADE - locally aware 하지 않아 주변 patch들에 대한 정보 없고 모든 feature를 사용, train feature count만큼 linear 하게 inference time이 증가하는 단점
            - PaDim - 동일 위치의 patch featrue끼리만 비교하므로, alignment가 맞지 않을 경우, anomaly에 대한 판단 불가
    - classification-based method : 지도 학습 기반이며, 사전 학습 모델을 통해 one-class classification을 적용하여 비정상 데이터 탐지
    - Recontruction-based method : 정상 data pattern train 후, input sample를 재구성하여 정상적인 data pattern과의 차이를 계산하는 방식
    
    ![**자료 출처 - 고려대학교 산업경영공학부 DSBA 연구실** ](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5dad4de0-741c-4625-b8d1-ae01877d8a5b/Untitled.png)
    
    **자료 출처 - 고려대학교 산업경영공학부 DSBA 연구실** 
    
- PatchCore Model Structure
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7fcf668f-181b-4e63-8ad8-6494176bef0a/Untitled.png)
    
    - PatchCore Model task
        1. Pretrained network - feature를 추출
        2. 추출된 feature들의 locally aware patch feature로 만듦
        3. coreset sampling을 통해 memory bank에 추가 될 patch feature를 선별 후, 추가
        4. memory bank에 있는 feature들을 하나씩 꺼내, test 단계에서 anomaly socre를 통해 판정
    - Training
        - locally aware patch features
            - Mid-level feature를 통한 local patch feature 사용
            -. 이전 anomaly detection 같은 경우, high level feature를 사용하였으나,
               space data loss(high level로 갈수록 위치 정보 손실), pretrain bias의 빈도가 높음
            -. patchcore에서는 위치 정보 및 imagenet bias의 비중을 낮추기 위해,
               mid level feature를 사용
            -. patchcore는 intermediate feature hierarchies i,  i+1만을 사용하는 이유는
               user feature의 generality를 방지하고, spatial resolution을 보존하기 위함
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/88ea7340-e2ac-4550-b85f-c08b9120e465/Untitled.png)
                
                ![자료 출처 -https://pdfs.semanticscholar.org/3154/d217c6fca87aedc99f47bdd6ed9b2be47c0c.pdf?ref=dataroots.ghost.io](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ba74bf06-ca86-49c5-a4e5-3221fb2e14c8/Untitled.png)
                
                자료 출처 -https://pdfs.semanticscholar.org/3154/d217c6fca87aedc99f47bdd6ed9b2be47c0c.pdf?ref=dataroots.ghost.io
                
        - coreset subsampling
            - SPADE 방식처럼 global averaging 한 후, feature를 memory bank에 모두 넣으면 무거움
            - Greedy Search 방법으로 subsampling을 수행
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/93116c29-9a85-4bb9-bd43-4a68f4a7bc92/Untitled.png)
                
    - PatchCore
        - memory bank (pre - coreset subsampling)
            - Patchcore 방식은 k nearest neighbor 방식으로 anomaly score를 계산 하기에, memory bank 안에 feature들 많으면 많을수록 computation 하는 과정이 오래 걸림
            (prev - spade 방식이 추출된 feature들을 memory 안에 모두 저장, inference time,
            연산량 이슈)
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a3b14668-0e8e-4088-8eff-50c9fd5bb4f0/Untitled.png)
                
                ![왼 - memory bank feature, 오 - 군집 중, 대표값을 표시(형광색)](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3a367646-a04f-4ebd-a12c-6669a6d7f6f0/Untitled.png)
                
                왼 - memory bank feature, 오 - 군집 중, 대표값을 표시(형광색)
                
            - 최초 sample을 고르고, 이후 patch feature들 중 가장 멀리 떨어져 있는 patch feature를 선택하는 과정을 반복 하여, 군집 데이터들의 대표값들을 선정
            - uniform distribution test는 아래 그림과 같이 greedy search algorithm이 중복 없이 sampling 되어 있는 것을 볼 수 있음.
                
                ![random, greedy search 방식](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/19d361b9-7314-4f92-8c77-77e499568499/Untitled.png)
                
                random, greedy search 방식
                
    - Testing
        - detection and segmentation(localization)
            - detection - image level 이므로 arg max, min
            (test image 내에서 anomaly score가 가장 높은 값을 anomaly score로 사용)
            - seg, localization - pixel level 단위이며, 공간 정보를 가지고 있어 해당 공간에 대응되는 patch들끼리 비교를 하면 anomaly score 산출 가능
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7113c56a-c5d0-4d3d-bafc-401eccc51a05/Untitled.png)
                
            - Test image에 대한 local patch features 추출하여 각 patch features를 memory bank에 query로 사용
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/83f18856-9eeb-408c-a320-98ebef5e6fbb/Untitled.png)
                
            - Nearest Neighbor 방법을 통해 query와 coreset에 대한 distance 를 기준으로 anomaly score 계산
            - Nearest Neigbhor는 faiss를 통해 빠르게 연산 가능
            - Faiss는 모든 feature를 비교하지 않고, cluster을 활용하여 효율적인 similarity 계산 가능함
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/31c1ce4b-1134-4577-ab55-8850951d6505/Untitled.png)
                
        - anomaly score
            - image level - patch들 중 가장 큰 anomaly score를 image에 대한 anomaly score
            - pixel level - local patch에 대한 resolution을 source(원본) Image size로 interpolation 후 gaussian smoothing을 적용
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4a8470b0-df25-493f-851e-17a62e5d8337/Untitled.png)
            

3 . Experiments

- anomaly detection and segmentation
    - performance auroc or pixel-wise auroc - MVTec data set
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b84035a8-2b57-47df-b760-925881271bbb/Untitled.png)
        
- coreset sampling 방법론 성능
    - 이전 sota WideResNet50을 backbone으로 사용한 PaDiM이였으나, patchcore 등장 이후 바뀜
    - 적은 데이터 수에도 이전 방법론보다 높은 성능을 보임
    - higher sample efficiency (학습 수가 적어도 sota level의 성능 확인)
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/81950af2-8664-445a-85d0-172d36bd2d1f/Untitled.png)
        
    - patcch core에서 coreset sampling을 진행해도 비교적 성능 하락이 적으며, 초기엔 오히려 성능이
    잠시 오르는 부분을 아래 그래프에서 확인 가능 (이전 random, learned 방식보다 우수한 성능)
    - memory bank size를 줄여도 성능 유지 (inference time 감소)
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dccd5150-68a7-4f16-ad1b-8bcf9e578b2e/Untitled.png)
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d1d05168-cf7d-4937-ba8b-48bf08445c6c/Untitled.png)
        

4 . Conclusion

- patchcore는 density base method 방식이며, 이전 sota anomaly detection 방식보다 가볍고 우수한 성능이 보장된다.
- feature extract를 다양한 방법론으로 접근 하거나, image alingment 방식을 개선할 새로운 방법론이 더 나오면 산업 분야에 더 우수한 성능? General하게 적용 가능 할 것 같습니다.

5 . Q&A - END (~)
