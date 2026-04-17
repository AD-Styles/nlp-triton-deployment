import torch
import torch.nn as nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.pooler_output)

def export_to_onnx(save_path):
    model = BERTClassifier(num_classes=2)
    model.eval()

    dummy_input = torch.ones(1, 128, dtype=torch.long)
    torch.onnx.export(
        model, (dummy_input, dummy_input), save_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        # [일치 확인] README에서 언급한 seq_length 가변축 적용
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_length'}, 
            'attention_mask': {0: 'batch_size', 1: 'seq_length'}, 
            'output': {0: 'batch_size'}
        },
        opset_version=12
    )
    print(f"✅ 모델 변환 완료: {save_path}")

if __name__ == "__main__":
    export_to_onnx("model_repository/bert_classifier/1/model.onnx")
