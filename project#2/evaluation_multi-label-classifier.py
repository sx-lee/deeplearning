import os
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import json
import argparse

# JSONL 파일 읽어서 리스트로 변환
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f:
            json_list.append(json.loads(line.strip()))
    return json_list

# 리스트 데이터를 JSONL 파일로 저장
def jsonldump(j_list, fname):
    with open(fname, "w", encoding="utf-8") as f:
        for json_data in j_list:
            f.write(json.dumps(json_data, ensure_ascii=False) + "\n")

# 분석 대상 디렉터리와 파일 설정
data_dir = r"C:\Users\PC\Desktop\1.데이터\Training\02.라벨링데이터\TL_1.발화단위평가_경제활동_상품상거래"
file_prefix = "경제활동_상품상거래_"
start_idx = 1
end_idx = 100

# 결과를 저장할 리스트
all_data = []

# 분석 작업 수행
for idx in range(start_idx, end_idx + 1):
    file_name = f"{file_prefix}{idx}.json"
    file_path = os.path.join(data_dir, file_name)

    # 파일 존재 여부 확인 후 처리
    if os.path.exists(file_path):
        print(f"Processing: {file_name}")
        try:
            # JSON 데이터 읽기
            json_data = jsonlload(file_path)
            all_data.extend(json_data)  # 데이터를 병합
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    else:
        print(f"File not found: {file_name}")

# 결과를 저장할 JSONL 파일 경로
output_path = os.path.join(data_dir, "processed_data.jsonl")

# 분석 결과 저장
jsonldump(all_data, output_path)
print(f"Processed data saved to: {output_path}")

# Torch 사용 환경 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Likeability 레이블 정의
likeability_labels = ['no', 'yes']
likeabilityid2label = {idx: label for idx, label in enumerate(likeability_labels)}
likeabilitylabel2id = {label: idx for idx, label in enumerate(likeability_labels)}

# TP, FP, FN, TN 초기화
TP_result_dict = {label: 0 for label in likeability_labels}
FP_result_dict = {label: 0 for label in likeability_labels}
FN_result_dict = {label: 0 for label in likeability_labels}
TN_result_dict = {label: 0 for label in likeability_labels}

# CLI 인자 파싱
def parse_args():
    parser = argparse.ArgumentParser(description="classifier using pretrained model")

    parser.add_argument(
        "--base_model", type=str, default="klue/roberta-base"
    )
    parser.add_argument(
        "--max_len", type=int, default=512
    )
    parser.add_argument(
        "--classifier_hidden_size", type=int, default=768
    )
    parser.add_argument(
        "--classifier_dropout_prob", type=float, default=0.1, help="dropout in classifier"
    )
    args = parser.parse_args()
    return args

# 모델 정의
class SimpleClassifier(nn.Module):
    def __init__(self, args, num_label):
        super().__init__()
        self.dense = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.dropout = nn.Dropout(args.classifier_dropout_prob)
        self.output = nn.Linear(args.classifier_hidden_size, num_label)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

class CustomClassifier(nn.Module):
    def __init__(self, args, num_label, len_tokenizer):
        super(CustomClassifier, self).__init__()

        self.num_label = num_label
        self.pre_trained_model = AutoModel.from_pretrained(args.base_model)
        self.pre_trained_model.resize_token_embeddings(len_tokenizer)

        self.labels_classifier = SimpleClassifier(args, self.num_label)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.pre_trained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )

        sequence_output = outputs[0]
        logits = self.labels_classifier(sequence_output)

        loss = None

        if labels is not None:
            class_weights = torch.tensor([9.0, 1.0]).to(device)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.num_label),
                            labels.view(-1))

        return loss, logits

# Likeability 모델 로드
likeability_test_data = "경로/likeability_test_data.jsonl"  # 이 부분을 실제 파일 경로로 수정하세요.
likeability_model_path = "경로/likeability_model.pth"  # 이 부분을 모델 경로로 수정하세요.

test_data_list = jsonlload(likeability_test_data)

parser = argparse.ArgumentParser(prog="train", description="")
args = parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.base_model)

model = CustomClassifier(args, 2, len(tokenizer))
model.load_state_dict(torch.load(likeability_model_path, map_location=device))
model.to(device)
model.eval()

print(len(test_data_list))
print('대화 단위')
pred_data_list = []

# 예측 결과 저장
for data in tqdm(test_data_list):
    history = data['input']

    tokenized_data = tokenizer(history, padding='max_length', max_length=512,
                               truncation=True)

    input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
    attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)

    with torch.no_grad():
        _, logits = model(input_ids, attention_mask)
    predictions = torch.argmax(logits, dim=-1)

    pred_data_list.append({
        "id": data['id'],
        "input": data['input'],
        "output": likeabilityid2label[int(predictions[0])]
    })

    if likeabilityid2label[int(predictions[0])] == 'yes' and data['output'] == 'yes':
        TP_result_dict["likeability"] += 1
    elif likeabilityid2label[int(predictions[0])] == 'no' and data['output'] == 'no':
        TN_result_dict["likeability"] += 1
    elif likeabilityid2label[int(predictions[0])] == 'no' and data['output'] == 'yes':
        FN_result_dict["likeability"] += 1
    elif likeabilityid2label[int(predictions[0])] == 'yes' and data['output'] == 'no':
        FP_result_dict["likeability"] += 1

# 결과를 저장
jsonldump(pred_data_list, 'likeability-output.json')

# 다중 라벨 분류기 로드
label2id = {"linguistic_acceptability": 0, "consistency": 1, "interestingness": 2, "unbias": 3, "harmlessness": 4, "no_hallucination": 5, "understandability": 6, "sensibleness": 7, "specificity": 8}
id2label = {idx: label for label, idx in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(likeability_model_path)
multi_label_classifier = AutoModelForSequenceClassification.from_pretrained(likeability_model_path).to(device)

test_data_list = jsonlload("경로/다중라벨테스트파일.jsonl")  # 다중 라벨 테스트 파일 경로로 수정

print('발화 단위')
print(len(test_data_list))

pred_data_list = []
for data in tqdm(test_data_list):
    history = data['input']['history']
    bot_answer = data['input']['bot_answer']

    tokenized_data = tokenizer(bot_answer, history, padding='max_length', max_length=512,
                               truncation=True)

    input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
    attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)

    result = {}
    threshold = 0.5
    with torch.no_grad():
        outputs = multi_label_classifier(input_ids, attention_mask)
        logits = outputs.logits

    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > threshold).int()

    for i in range(len(probabilities[0])):
        if predictions[0][i] == 1:
            result[id2label[i]] = 'yes'
        else:
            result[id2label[i]] = 'no'

    pred_data_list.append({
        "id": data['id'],
        "input": data['input'],
        "output": result
    })

    # TP, TN, FP, FN 갱신
    for key, value in data['output'].items():
        if result[key] == 'yes' and value == 'yes':
            TP_result_dict[key] += 1
        elif result[key] == 'no' and value == 'no':
            TN_result_dict[key] += 1
        elif result[key] == 'no' and value == 'yes':
            FN_result_dict[key] += 1
        elif result[key] == 'yes' and value == 'no':
            FP_result_dict[key] += 1

jsonldump(pred_data_list, 'multi-label-output.json')

# 결과 출력
print("TP_result_dict: ", TP_result_dict)
print("TN_result_dict: ", TN_result_dict)
print("FN_result_dict: ", FN_result_dict)
print("FP_result_dict: ", FP_result_dict)

# 정확도 계산
accuracy_dict = {}
accuracy_sum = 0
for key in TP_result_dict.keys():
    accuracy_dict[key] = (TP_result_dict[key] + TN_result_dict[key]) / (TP_result_dict[key] + TN_result_dict[key] + FP_result_dict[key] + FN_result_dict[key])
    accuracy_sum += accuracy_dict[key]

print("accuracy_dict: ", accuracy_dict)
print("accuracy: ", accuracy_sum / len(accuracy_dict))



