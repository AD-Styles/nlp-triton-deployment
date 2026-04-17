# 🌐 Deploying BERT: From PyTorch to Triton Server
### 실습으로 익히는 NVIDIA Triton Inference Server 기반 NLP 모델 서빙과 백엔드 추론 파이프라인 설계

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Triton](https://img.shields.io/badge/NVIDIA-Triton_Server-76B900?style=flat-square&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

---

## 📌 프로젝트 개요 (Project Overview)
본 프로젝트는 파인튜닝이 완료된 자연어 처리(NLP) 모델을 실제 프로덕션(Production) 환경에 배포하기 위한 파이프라인을 다룹니다. NVIDIA Triton Inference Server를 활용하여 딥러닝 모델의 서빙(Serving) 구조를 설계하고, 지연 시간(Latency) 최소화 및 처리량(Throughput) 극대화를 위한 실무적인 백엔드 엔지니어링 역량을 확보하는 데 중점을 두었습니다.

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
```

---

## 🛠️ 기술 스택 (Tech Stack)
| 구분 | 상세 항목 |
| :--- | :--- |
| **Language** | Python |
| **Model Serving** | NVIDIA Triton Inference Server |
| **Model Format** | ONNX (Open Neural Network Exchange) |
| **Communication** | HTTP / REST, gRPC |

---

## 🚀 주요 기능 및 워크플로우 (Key Features & Workflow)
| 단계 | 주요 기능 (Features) | 핵심 기술 (Key Tech) |
| :--- | :--- | :--- |
| **1. 모델 내보내기** | PyTorch 모델을 최적화된 서빙 포맷으로 변환 | ONNX Export, Tensor Tracing |
| **2. 저장소 구성** | Triton 아키텍처 규칙에 맞는 디렉토리 및 설정 파일 작성 | `config.pbtxt`, Versioning |
| **3. 서버 최적화** | 실시간 트래픽 효율을 위한 동적 배치 구성 | Dynamic Batching, Concurrency |
| **4. 클라이언트 통신** | 백엔드 애플리케이션에서 모델로 추론 요청 및 결과 수신 | Triton Python Client (HTTP) |

---

## 💡 회고록 (Retrospective)
&emsp;&emsp;L2 프로젝트에서 BERT를 파인튜닝하며 '모델의 성능(Accuracy)'을 높이는 데 집중했다면, 이번 L3 프로젝트는 '모델의 속도(Latency)'와 '효율성(Throughput)'이라는 완전히 새로운 차원의 엔지니어링을 경험하는 계기가 되었습니다. 아무리 성능이 뛰어난 모델이라도, 사용자의 요청을 제때 처리하지 못하거나 서버 비용을 낭비한다면 좋은 AI 프로덕트가 될 수 없다는 실무적인 딜레마를 깨달았습니다.

&emsp;&emsp;PyTorch 모델을 ONNX로 변환하고 NVIDIA Triton을 통해 서빙하는 과정을 직접 구축해 보니, 왜 글로벌 테크 기업들이 추론 전용 서버(Inference Server)를 별도로 구성하는지 명확히 이해할 수 있었습니다. 특히, 여러 요청을 모아서 한 번에 GPU로 올려보내는 '동적 배치(Dynamic Batching)' 기술은 하드웨어 자원을 극한으로 끌어쓰는 서버 최적화의 꽃이었습니다.

&emsp;&emsp;전처리 파이프라인(L1)부터 고도화된 모델 파인튜닝(L2), 그리고 최종 프로덕션 배포(L3)까지 이어지는 전체 사이클을 경험하며, 단순히 AI 모델을 훈련시키는 '연구자'를 넘어, 모델을 실제 비즈니스 가치로 연결할 수 있는 'AI 엔지니어'로서의 시야를 갖추게 되었습니다.
