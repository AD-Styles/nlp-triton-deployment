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
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # [Fix 1] 서버 연결 실패 예외 처리
        try:
            self.client = httpclient.InferenceServerClient(url=self.url)
        except Exception as e:
            raise ConnectionError(f"Triton 서버({self.url}) 클라이언트 초기화 실패: {e}")

    def check_server_health(self) -> bool:
        """Triton 서버 상태 검증"""
        try:
            if not self.client.is_server_ready():
                print(f"🚨 Triton Server ({self.url}) is not ready.")
                return False
            print(f"✅ Triton Server ({self.url}) is online and ready.")
            return True
        except InferenceServerException as e:
            print(f"🚨 서버 상태 확인 중 오류 발생 (네트워크/서버 다운 확인 필요):\n{e}")
            return False

    def preprocess(self, text: str):
        """동적 길이(Dynamic Axes)를 활용한 최적화된 토큰화"""
        # [Fix 3] max_length=128 하드코딩 제거, 입력된 문장 길이에 딱 맞게 패딩 (longest)
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='longest',  # 동적 시퀀스 길이 허용 (연산량 최소화)
            return_attention_mask=True
        )
        input_ids = np.array([encoded['input_ids']], dtype=np.int32)
        attention_mask = np.array([encoded['attention_mask']], dtype=np.int32)
        return input_ids, attention_mask

    def infer(self, text: str):
        """추론 요청 및 Latency 측정"""
        input_ids, attention_mask = self.preprocess(text)

        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT32"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT32")
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        outputs = [httpclient.InferRequestedOutput("output")]

        # [Fix 2] 추론 중 발생할 수 있는 에러 처리 및 속도 측정
        try:
            start_time = time.time()
            response = self.client.infer(self.model_name, inputs, outputs=outputs)
            latency_ms = (time.time() - start_time) * 1000
            
            result_logits = response.as_numpy("output")
            prediction = np.argmax(result_logits, axis=1)[0]
            return prediction, latency_ms
        except InferenceServerException as e:
            print(f"🚨 추론 실패 (모델 이름 확인 또는 서버 로그 참조):\n{e}")
            return None, None

if __name__ == "__main__":
    # [Fix 1] 하드코딩 제거 및 CLI 환경 지원
    parser = argparse.ArgumentParser(description="Production NLP Triton Client")
    parser.add_argument('--url', type=str, default='localhost:8000', help='Triton Server HTTP URL (ex: localhost:8000)')
    parser.add_argument('--model', type=str, default='bert_classifier', help='Target model name in repository')
    args = parser.parse_args()

    client = TritonNLPClient(url=args.url, model_name=args.model)
    
    if client.check_server_health():
        sample_text = "Deploying models with dynamic batching and variable sequence lengths is incredibly efficient!"
        print(f"\n--- Triton Inference Request ---")
        print(f"Input: {sample_text}")
        
        result, latency = client.infer(sample_text)
        if result is not None:
            print(f"Predicted Class: {result} (Latency: {latency:.2f}ms)")
