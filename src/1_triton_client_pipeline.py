import argparse
import time
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from transformers import BertTokenizer

class TritonNLPClient:
    """NVIDIA Triton Inference Server와 통신하기 위한 프로덕션 레벨 클라이언트"""
    
    def __init__(self, url: str, model_name: str):
        self.url = url
        self.model_name = model_name
        # BERT 토크나이저 로드 (학습 시 사용한 모델과 동일해야 함)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        try:
            # HTTP 클라이언트 초기화
            self.client = httpclient.InferenceServerClient(url=self.url)
        except Exception as e:
            raise ConnectionError(f"Triton 서버({self.url}) 클라이언트 초기화 실패: {e}")

    def check_server_health(self) -> bool:
        """서버 가동 상태 및 모델 로드 상태 확인"""
        try:
            if not self.client.is_server_ready():
                print(f"🚨 Triton Server ({self.url}) 가 아직 준비되지 않았습니다.")
                return False
            
            if not self.client.is_model_ready(self.model_name):
                print(f"🚨 모델 '{self.model_name}'이 서버에 로드되지 않았습니다.")
                return False
                
            print(f"✅ Triton Server 및 모델 '{self.model_name}' 준비 완료.")
            return True
        except InferenceServerException as e:
            print(f"🚨 서버 상태 확인 중 오류 발생 (네트워크 연결 확인 필요):\n{e}")
            return False

    def preprocess(self, text: str):
        """
        [최적화 핵심] Dynamic Sequence Length 대응 전처리
        padding='longest'를 사용하여 입력 문장 길이에 딱 맞는 텐서 크기를 생성합니다.
        """
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='longest',     # 하드코딩된 128 대신 문장 길이에 맞춤
            truncation=True,       # 혹시 모를 초장문은 잘라냄
            return_attention_mask=True
        )
        
        # Triton 입력을 위한 NumPy 배열 변환 (배치 차원 추가)
        input_ids = np.array([encoded['input_ids']], dtype=np.int32)
        attention_mask = np.array([encoded['attention_mask']], dtype=np.int32)
        
        return input_ids, attention_mask

    def infer(self, text: str):
        """Triton 서버에 추론 요청 및 Latency 측정"""
        input_ids, attention_mask = self.preprocess(text)

        # 입력 텐서 구성
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT32"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT32")
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        # 출력 텐서 지정 (L2 모델 결과인 2개 클래스)
        outputs = [httpclient.InferRequestedOutput("output")]

        try:
            start_time = time.time()
            # 서버로 추론 요청 전송
            response = self.client.infer(self.model_name, inputs, outputs=outputs)
            latency_ms = (time.time() - start_time) * 1000
            
            # 결과 처리
            result_logits = response.as_numpy("output")
            prediction = np.argmax(result_logits, axis=1)[0]
            
            return prediction, latency_ms
        except InferenceServerException as e:
            print(f"🚨 추론 과정에서 에러 발생:\n{e}")
            return None, None

if __name__ == "__main__":
    # 실행 시 인자를 통해 유연하게 설정 가능
    parser = argparse.ArgumentParser(description="Production-Ready Triton NLP Client")
    parser.add_argument('--url', type=str, default='localhost:8000', help='Triton Server URL')
    parser.add_argument('--model', type=str, default='bert_classifier', help='Model Name')
    args = parser.parse_args()

    # 클라이언트 객체 생성
    client = TritonNLPClient(url=args.url, model_name=args.model)
    
    # 서버 헬스체크 후 추론 실행
    if client.check_server_health():
        test_sentences = [
            "This model serving pipeline is extremely efficient!",
            "I am very happy with the deployment results."
        ]
        
        print(f"\n{'='*50}")
        print(f"🚀 Triton Inference 파이프라인 가동 시작")
        print(f"{'='*50}")

        for text in test_sentences:
            pred, lat = client.infer(text)
            if pred is not None:
                sentiment = "Positive" if pred == 1 else "Negative"
                print(f"🔹 Input: {text}")
                print(f"🔸 Result: {sentiment} (Latency: {lat:.2f}ms)")
                print(f"{'-'*50}")
