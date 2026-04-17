import torch
from transformers import BertModel

def export_to_onnx(model_path, save_path):
    # 1. 모델 로드
    model = BertModel.from_pretrained('bert-base-uncased')
    # 실제 환경에서는 model.load_state_dict(torch.load(model_path))로 학습된 가중치 로드
    model.eval()

    # 2. 가상의 입력값 생성 (Dummy Input)
    dummy_input_ids = torch.ones(1, 128, dtype=torch.long)
    dummy_mask = torch.ones(1, 128, dtype=torch.long)

    # 3. ONNX로 내보내기
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_mask),
        save_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=12
    )
    print(f"✅ 모델이 {save_path}로 성공적으로 변환되었습니다.")

if __name__ == "__main__":
    export_to_onnx(None, "model_repository/bert_classifier/1/model.onnx")
