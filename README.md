# ⚽️ U-Net 기반 축구 영상 Semantic Segmentation (11 Classes)

## 📌 프로젝트 요약
축구 경기 장면 이미지에서 각 픽셀을 11개 클래스(잔디/선수/골대/관중 등)로 분류하는
Semantic Segmentation 모델을 U-Net 기반으로 구현하고, 학습 및 평가를 통해 성능과 한계를 분석한 프로젝트입니다.

세그멘테이션은 “이미지 1장 = 라벨 1개”가 아니라 “이미지 1장 = 픽셀 수만큼 라벨”을 맞춰야 하므로 난이도가 높습니다.
특히 본 데이터는 100장으로 매우 작고 프레임 유사성이 강해, 과적합과 데이터 누수 방지가 핵심 이슈였습니다.

---

## 데이터
- Football Semantic Segmentation (UEFA 슈퍼컵 2017 하이라이트 기반)
- 총 100장 프레임
- 11 classes:
  Goal Bar, Referee, Advertisement, Ground, Ball,
  Coaches & Officials, Audience,
  Goalkeeper A, Goalkeeper B, Team A, Team B

---

## 핵심 설계
- 데이터가 작고 프레임 유사성이 높아 랜덤 분할 시 누수 위험이 큼  
  → GroupKFold로 평가 구조를 구성
- 작은 데이터에서 full fine-tuning은 과적합이 빠르게 진행될 가능성이 큼  
  → 전이학습 U-Net(ResNet34 encoder) + feature extraction(encoder freeze) 중심 전략 사용

---

## 모델/학습 구성
- Baseline: 간단한 U-Net으로 파이프라인 sanity check
- Main: segmentation_models_pytorch의 Unet(resnet34 encoder, imagenet weights)
  - encoder freeze(feature extraction)
- Loss:
  - CrossEntropy + Dice(가중치 포함, alpha 조정)
- Optimizer:
  - AdamW (lr=1e-3, weight_decay=1e-3)
- Metric:
  - mIoU(mean Intersection over Union)

---

## 결과 요약
- 교차검증 최고 성능(Best Val mIoU): 0.5359
- 전체 데이터 기준 최종 평가(Final mIoU): 0.4615

---

## 결론
- 100장 소규모 세그멘테이션에서는 과적합이 강하게 발생했다.
- 이 환경에서는 partial fine-tuning보다 feature extraction이 더 안정적이고 성능도 유리했다.
- CE+Dice 조합과 학습률/규제(weight_decay) 조정이 성능에 유의미한 영향을 주었다.


