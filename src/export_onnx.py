import torch
from transformers import BertModel

def export_to_onnx(model_path, save_path):
    # 1. 모델 로드
    # 실제 환경에서는 학습된 가중치를 로드해야 함: 
    # model = BERTClassifier(...) 
    # model.load_state_dict(torch.load(model_path))
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # 2. 가상의 입력값 생성 (Dummy Input)
    # 초기 생성 시에는 128 길이를 기준으로 하지만, dynamic_axes 설정이 우선됨
    dummy_input_ids = torch.ones(1, 128, dtype=torch.long)
    dummy_mask = torch.ones(1, 128, dtype=torch.long)

    # 3. ONNX로 내보내기
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_mask),
        save_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        # [CRITICAL] 1번 인덱스(시퀀스 길이)를 'seq_length'라는 가변축으로 정의
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_length'}, 
            'attention_mask': {0: 'batch_size', 1: 'seq_length'}, 
            'output': {0: 'batch_size'}
        },
        opset_version=12
    )
    print(f"✅ 가변 길이 설정이 적용된 모델이 {save_path}로 변환되었습니다.")

if __name__ == "__main__":
    # 저장 경로를 본인의 레포지토리 구조에 맞게 확인하십시오.
    export_to_onnx(None, "model_repository/bert_classifier/1/model.onnx")
