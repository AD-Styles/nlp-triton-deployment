import torch
from transformers import BertModel

def export_to_onnx(save_path):
    # 1. 모델 로드 (Pretrained 모델 사용)
    model = BertModel.from_pretrained('bert-base-uncased')
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
        # [핵심] 1번 인덱스(시퀀스 길이)를 'seq_length' 가변축으로 지정
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_length'}, 
            'attention_mask': {0: 'batch_size', 1: 'seq_length'}, 
            'output': {0: 'batch_size'}
        },
        opset_version=12
    )
    print(f"✅ 가변 시퀀스 길이가 적용된 모델이 {save_path}로 변환되었습니다.")

if __name__ == "__main__":
    export_to_onnx("model_repository/bert_classifier/1/model.onnx")
