# 🌐 Deploying BERT: From PyTorch to Triton Server
### NVIDIA Triton Inference Server 기반 NLP 모델 서빙과 백엔드 추론 파이프라인 설계

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Triton](https://img.shields.io/badge/NVIDIA-Triton_Server-76B900?style=flat-square&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

---

## 📌 프로젝트 개요 (Project Overview)
본 프로젝트는 파인튜닝이 완료된 자연어 처리(NLP) 모델을 실제 프로덕션(Production) 환경에 배포하기 위한 파이프라인을 다뤘습니다. NVIDIA Triton Inference Server를 활용하여 딥러닝 모델의 서빙(Serving) 구조를 설계하고, 지연 시간(Latency) 최소화 및 처리량(Throughput) 극대화를 위한 실무적인 백엔드 엔지니어링 역량을 확보하는 데 중점을 두었습니다.

---

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

## 📊 성능 최적화 결과 (Benchmark)
Triton의 동적 배치(Dynamic Batching)와 ONNX 가변 길이(Dynamic Axes) 추론을 적용하여, 기존 로컬 환경 대비 서빙 지연 시간과 처리량을 대폭 개선했습니다. 클라이언트의 텍스트 길이에 맞춰 텐서 크기를 유동적으로 조절(`padding='longest'`)하여 낭비되는 연산을 제거했습니다.

| 서빙 아키텍처 | 배치 설정 | 시퀀스 길이 처리 | Latency (ms/req) | Throughput (req/sec) |
| :--- | :--- | :--- | :--- | :--- |
| **Local PyTorch** | Batch=1 (Sequential) | 128 고정 (Max Padding) | ~ 45.2 ms | ~ 22 TPS |
| **Triton Server** | **동적 배치 (Max=8)** | **입력 맞춤형 가변 길이** | **~ 12.5 ms** | **~ 145 TPS** |

> **🔗 Engineering Deep Dive: 동적 패딩(Dynamic Padding) 및 시스템 동기화**
> 
> 일반적인 모델링 튜토리얼에서는 텐서 크기를 맞추기 위해 `max_length=128` 등으로 길이를 하드코딩합니다. 하지만 프로덕션 환경에서는 네트워크 대역폭(I/O) 낭비와 GPU 연산량(O(N²) 복잡도) 낭비라는 두 가지 치명적인 병목을 발생시킵니다.
> 
> 본 프로젝트는 클라이언트 단에서 `padding='longest'`를 적용해 **입력 문장 길이에 딱 맞는 타이트한 텐서를 생성**하고, 이를 처리하기 위해 모델과 서버 설정을 엔드 투 엔드로 동기화했습니다.
>
> #### 🔗 엔드 투 엔드 가변 길이 동기화 (Technical Alignment)
> * **Model Export (`export_onnx.py`):** ONNX 추출 시 `dynamic_axes` 설정에 `seq_length` 가변축을 명시적으로 추가하여, 모델이 다양한 입력 크기를 수용할 수 있는 구조를 갖추도록 설계했습니다.
> * **Server Config (`config.pbtxt`):** 서버의 입력 차원을 `dims: [ -1 ]`로 설정하여, 모델이 가진 가변축과 서버의 수용 규격을 일치시켰습니다. 
> 
> 이러한 동기화 작업을 통해 하드코딩된 고정 길이의 제약을 제거하였으며, 결과적으로 서버 리소스의 낭비를 막고 추론 효율을 극대화할 수 있었습니다.

---

## 💡 회고록 (Retrospective)
&emsp;&emsp;솔직히 'nlp-bert-finetuning' 프로젝트에서 BERT를 파인튜닝할 때까지만 해도, 검증 세트의 정확도(Accuracy)가 높게 나오면 모든 작업이 끝난 줄로만 알았습니다. 주피터 노트북에서 `predict()`를 돌려보고 결과가 잘 나오면 환호하기 바빴습니다. 그런데 막상 이 모델을 '실제 서비스' 환경에 올려보려 하니 완전히 다른 세계가 열렸습니다. 모델이 아무리 똑똑해도 사용자가 질문을 던진 후 한참 뒤에 대답한다면 아무도 쓰지 않을 것이기 때문입니다. 로컬 환경에서 혼자 돌릴 땐 몰랐던 '지연 시간(Latency)'과 '서버 리소스(비용)'라는 진짜 엔지니어링의 딜레마에 부딪혔습니다.
<br>&emsp;&emsp;이를 해결하기 위해 글로벌 테크 기업들이 프로덕션 환경에서 사용한다는 NVIDIA Triton Inference Server를 도입했습니다. 익숙했던 PyTorch를 떠나 모델을 ONNX 포맷으로 변환하고, Triton이 요구하는 엄격한 디렉토리 구조와 `config.pbtxt`를 세팅하는 과정은 결코 만만치 않았습니다. 경로 하나, 타입 하나만 틀려도 서버가 모델을 로드하지 못했기 때문입니다. 하지만 이 깐깐한 규칙들을 하나씩 맞춰가면서, 왜 실무에서는 단순한 Flask API가 아니라 이런 전문적인 '서빙 아키텍처'를 구축하는지 몸소 깨달을 수 있었습니다.
<br>&emsp;&emsp;이번 프로젝트에서 가장 짜릿했던 순간은 '동적 패딩(Dynamic Padding)'으로 병목 현상을 해결했을 때입니다. 처음엔 기초 튜토리얼에서 배운 대로 모든 텍스트를 무조건 `max_length=128`로 패딩해서 서버에 전달했습니다. 그런데 문득 "'안녕'이라는 짧은 단어를 보낼 때도 의미 없는 빈 공간을 126개나 덧붙여서 네트워크로 전송하고, 심지어 $O(N^2)$의 시간 복잡도를 가지는 BERT가 그 빈 공간까지 전부 연산하고 있네?"라는 생각이 들었습니다. 그래서 즉시 클라이언트 코드를 수정하여 입력된 문장 길이에 맞춰 텐서를 타이트하게 생성하고(`padding='longest'`), Triton 서버의 동적 배치(Dynamic Batching)가 이를 유연하게 처리하도록 구조를 개편했습니다. 
<br>&emsp;&emsp;결과는 놀라웠습니다. 낭비되던 네트워크 트래픽과 GPU 연산이 줄어들면서 처리량(Throughput)이 극적으로 상승했습니다. 전처리(
'nlp-preprocessing-foundation')와 모델 튜닝('nlp-bert-finetuning')을 거쳐 이 마지막 배포('nlp-triton-deployment') 단계까지 오면서 엔지니어링을 바라보는 시야가 완전히 달라졌습니다. 단순히 데이터를 가공하고 모델 성능을 올리는 데만 집착하던 학생의 틀에서 벗어나, 한정된 자원을 극한으로 활용하여 실제 비즈니스 가치(빠르고 안정적인 서비스)로 연결할 줄 아는 'AI 엔지니어'의 마인드를 갖게 된 것이 이번 기나긴 여정의 가장 큰 수확입니다.
