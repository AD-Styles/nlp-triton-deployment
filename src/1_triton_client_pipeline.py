import numpy as np
import tritonclient.http as httpclient
from transformers import BertTokenizer

class TritonNLPClient:
    """NVIDIA Triton Inference Server와 통신하기 위한 클라이언트 클래스"""
    
    def __init__(self, url: str = "localhost:8000", model_name: str = "bert_classifier"):
        self.client = httpclient.InferenceServerClient(url=url)
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def check_server_health(self) -> bool:
        """Triton 서버의 상태 확인"""
        if not self.client.is_server_ready():
            print("🚨 Triton Server is not ready.")
            return False
        print("✅ Triton Server is online and ready.")
        return True

    def preprocess(self, text: str):
        """텍스트를 Triton 서버가 인식할 수 있는 NumPy 배열(Tensor)로 변환"""
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        # Triton은 보통 int32 또는 int64 형태의 NumPy 배열을 입력으로 받습니다.
        input_ids = np.array([encoded['input_ids']], dtype=np.int32)
        attention_mask = np.array([encoded['attention_mask']], dtype=np.int32)
        return input_ids, attention_mask

    def infer(self, text: str):
        """Triton 서버에 추론 요청 및 결과 반환"""
        input_ids, attention_mask = self.preprocess(text)

        # Triton 입력 텐서 구성
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT32"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT32")
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        # Triton 출력 텐서 지정
        outputs = [httpclient.InferRequestedOutput("output")]

        # 서버에 요청 전송
        response = self.client.infer(self.model_name, inputs, outputs=outputs)
        result_logits = response.as_numpy("output")
        
        # 가장 확률이 높은 클래스 반환
        prediction = np.argmax(result_logits, axis=1)
        return prediction[0]

if __name__ == "__main__":
    client = TritonNLPClient()
    
    if client.check_server_health():
        sample_text = "Deploying models with Triton is extremely efficient!"
        print(f"--- Triton Inference Request ---")
        print(f"Input: {sample_text}")
        result = client.infer(sample_text)
        print(f"Predicted Class: {result}")
