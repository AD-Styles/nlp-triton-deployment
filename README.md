# 🚀 nlp-triton-deployment
### 실습으로 익히는 NVIDIA Triton Inference Server 기반 NLP 모델 배포

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Triton](https://img.shields.io/badge/NVIDIA-Triton_Server-76B900?style=flat-square&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

---

본 프로젝트는 파인튜닝이 완료된 자연어 처리(NLP) 모델을 실제 프로덕션(Production) 환경에 배포하기 위한 파이프라인을 다룹니다. NVIDIA Triton Inference Server를 활용하여 딥러닝 모델의 서빙(Serving) 구조를 설계하고, 지연 시간(Latency) 최소화 및 처리량(Throughput) 극대화를 위한 실무적인 백엔드 엔지니어링 역량을 확보하는 데 중점을 두었습니다.

## 📌 프로젝트 개요 (Project Overview)
단순히 로컬 환경에서 모델을 실행하는 것을 넘어, 대규모 트래픽을 감당할 수 있는 안정적인 AI 서비스 아키텍처를 구축합니다. 파이토치(PyTorch) 모델을 범용적인 ONNX 포맷으로 변환하고, 동적 배치(Dynamic Batching)와 모델 동시 실행(Concurrent Model Execution)을 지원하는 Triton 서버와 통신하는 클라이언트 로직을 직접 구현하여 모델 배포의 A to Z를 경험합니다.

## 📂 프로젝트 구조 (Project Structure)
```text
📂 model_repository/                      # Triton 서버가 읽어들이는 모델 저장소 (구조화)
    └── bert_classifier/
        ├── 1/
        │   └── model.onnx                # ONNX 포맷으로 내보낸 모델 가중치
        └── config.pbtxt                  # 입출력 텐서 및 동적 배치 설정 파일
📂 src/                                     
    └── 1_triton_client_pipeline.py       # 전처리 및 Triton 서버 HTTP/gRPC 통신 클라이언트
├── .gitignore                            # 모델 파일 및 캐시 제외 설정
├── LICENSE                               # MIT License
├── README.md                             # 배포 파이프라인 요약 및 실행 가이드
└── requirements.txt                      # tritonclient 등 필수 라이브러리
