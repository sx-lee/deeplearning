import json
import os

import torch
import argparse
import torch.nn as nn
from tqdm import trange, tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from sklearn.metrics import f1_score

PADDING_TOKEN = 1
S_OPEN_TOKEN = 0
S_CLOSE_TOKEN = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&',
                                  '&bank-account&', '&num&', '&online-account&']
}

labels = ['no', 'yes']
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}


def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)

    return j


# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)


# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list


# jsonlist를 jsonl 형태로 저장
def jsonldump(j_list, fname):
    f = open(fname, "w", encoding='utf-8')
    for json_data in j_list:
        f.write(json.dumps(json_data, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="unethical expression classifier using pretrained model")
    parser.add_argument(
        "--train_data", type=str, default="../data/nikluge-au-2022-train.jsonl",
        help="train file"
    )
    parser.add_argument(
        "--test_data", type=str, default="../data/nikluge-au-2022-test.jsonl",
        help="test file"
    )
    parser.add_argument(
        "--pred_data", type=str, default="./output/result.jsonl",
        help="pred file"
    )
    parser.add_argument(
        "--dev_data", type=str, default="../data/nikluge-au-2022-dev.jsonl",
        help="dev file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8
    )
    parser.add_argument(
        "--do_train", action="store_true"
    )
    parser.add_argument(
        "--do_eval", action="store_true"
    )
    parser.add_argument(
        "--do_test", action="store_true"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10
    )
    parser.add_argument(
        "--base_model", type=str, default="xlm-roberta-base"
    )
    parser.add_argument(
        "--model_path", type=str, default="./saved_models/default_path/"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/"
    )
    parser.add_argument(
        "--do_demo", action="store_true"
    )
    parser.add_argument(
        "--max_len", type=int, default=256
    )
    parser.add_argument(
        "--classifier_hidden_size", type=int, default=768
    )
    parser.add_argument(
        "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier"
    )
    args = parser.parse_args()
    return args


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
            class_weights = torch.tensor([7.0, 3.0]).to(device)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.num_label),
                            labels.view(-1))

        return loss, logits


def tokenize_and_align_labels(tokenizer, input_text, label, max_len):
    data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': [],
    }
    tokenized_data = tokenizer(input_text, padding='max_length', max_length=max_len, truncation=True)
    data_dict['input_ids'].append(tokenized_data['input_ids'])
    data_dict['attention_mask'].append(tokenized_data['attention_mask'])
    data_dict['label'].append(label)

    return data_dict


def get_dataset(raw_data, tokenizer, max_len):
    input_ids_list = []
    attention_mask_list = []
    token_labels_list = []

    for utterance in raw_data:
        tokenized_data = tokenize_and_align_labels(tokenizer,
                                                   utterance['input'],
                                                   label2id[utterance['output']],
                                                   max_len)
        input_ids_list.extend(tokenized_data['input_ids'])
        attention_mask_list.extend(tokenized_data['attention_mask'])
        token_labels_list.extend(tokenized_data['label'])

    print(token_labels_list[:10])

    return TensorDataset(torch.tensor(input_ids_list), torch.tensor(attention_mask_list),
                         torch.tensor(token_labels_list))


def evaluation(y_true, y_pred):
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    print(y_true[:5])
    print(y_pred[:5])

    print('f1_score: ', f1_score(y_true, y_pred, average=None))
    print('f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))


def train(args=None):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print('train')
    print('model would be saved at ', args.model_path)

    print('loading train data')
    train_data = jsonlload(args.train_data)
    dev_data = jsonlload(args.dev_data)

    print('tokenizing train data')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print('We have added', num_added_toks, 'tokens')

    train_dataloader = DataLoader(get_dataset(train_data, tokenizer, args.max_len), shuffle=True,
                                  batch_size=args.batch_size)
    dev_dataloader = DataLoader(get_dataset(dev_data, tokenizer, args.max_len), shuffle=True,
                                batch_size=args.batch_size)

    print('loading model')
    model = CustomClassifier(args, len(labels), len(tokenizer))
    model.to(device)

    # print(model)

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.eps
    )
    epochs = args.num_train_epochs
    max_grad_norm = 1.0
    total_steps = epochs * len(train_dataloader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    epoch_step = 0

    for _ in trange(epochs, desc="Epoch"):
        model.train()
        epoch_step += 1
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            model.zero_grad()
            optimizer.zero_grad()

            loss, _ = model(b_input_ids, b_input_mask, b_labels)

            loss.backward()

            total_loss += loss.item()

            # print('batch_loss: ', loss.item())

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print("Epoch: ", epoch_step)
        print("Average train loss: {}".format(avg_train_loss))

        if args.do_eval:
            model.eval()

            pred_list = []
            label_list = []

            for batch in dev_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    loss, logits = model(b_input_ids, b_input_mask, b_labels)

                predictions = torch.argmax(logits, dim=-1)
                pred_list.extend(predictions)
                label_list.extend(b_labels)

            evaluation(label_list, pred_list)

        model_saved_path = args.model_path + 'saved_model_epoch_' + str(epoch_step) + '.pt'
        torch.save(model.state_dict(), model_saved_path)

    print("training is done")


def test(args):

    test_data_list = jsonlload(args.test_data)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    model = CustomClassifier(args, 2, len(tokenizer))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    TP_result_dict = {
        "likeability": 0
    }

    FP_result_dict = {
        "likeability": 0
    }

    FN_result_dict = {
        "likeability": 0
    }

    TN_result_dict = {
        "likeability": 0
    }

    print(len(test_data_list))

    pred_data_list = []

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
            "output": id2label[int(predictions[0])]
        })

        if id2label[int(predictions[0])] == 'yes' and data['output'] == 'yes':
            TP_result_dict["likeability"] += 1
        elif id2label[int(predictions[0])] == 'no' and data['output'] == 'no':
            TN_result_dict["likeability"] += 1
        elif id2label[int(predictions[0])] == 'no' and data['output'] == 'yes':
            FN_result_dict["likeability"] += 1
        elif id2label[int(predictions[0])] == 'yes' and data['output'] == 'no':
            FP_result_dict["likeability"] += 1

    # TP_result_dict=  {'linguistic_acceptability': 43089, 'consistency': 43616, 'interestingness': 46263, 'unbias': 43824, 'harmlessness': 43118, 'no_hallucination': 39887, 'understandability': 44285, 'sensibleness': 43345, 'specificity': 46947}
    # TN_result_dict=  {'linguistic_acceptability': 5660, 'consistency': 4239, 'interestingness': 1859, 'unbias': 5172, 'harmlessness': 6421, 'no_hallucination': 5864, 'understandability': 2721, 'sensibleness': 4196, 'specificity': 1842}
    # FN_result_dict=  {'linguistic_acceptability': 355, 'consistency': 841, 'interestingness': 94, 'unbias': 780, 'harmlessness': 344, 'no_hallucination': 1614, 'understandability': 1146, 'sensibleness': 769, 'specificity': 86}
    # FP_result_dict=  {'linguistic_acceptability': 967, 'consistency': 1375, 'interestingness': 1855, 'unbias': 295, 'harmlessness': 188, 'no_hallucination': 2706, 'understandability': 1919, 'sensibleness': 1761, 'specificity': 1196}

    jsonldump(pred_data_list, 'likeability-output.json')

    # print("data_len: ", len(test_data_list))
    print("TP_result_dict: ", TP_result_dict)
    print("TN_result_dict: ", TN_result_dict)
    print("FN_result_dict: ", FN_result_dict)
    print("FP_result_dict: ", FP_result_dict)

    accuracy_dict = {}

    precision_for_true = {}
    precision_for_no = {}

    recall_for_no = {}

    accuracy_sum = 0

    for key in TP_result_dict.keys():
        accuracy_dict[key] = (TP_result_dict[key] + TN_result_dict[key]) / (
                    TP_result_dict[key] + TN_result_dict[key] + FP_result_dict[key] + FN_result_dict[key])
        accuracy_sum += accuracy_dict[key]

        precision_for_true[key] = (TP_result_dict[key]) / (
                TP_result_dict[key] + FP_result_dict[key])
        precision_for_no[key] = (TN_result_dict[key]) / (
                TN_result_dict[key] + FN_result_dict[key])
        recall_for_no[key] = (TN_result_dict[key]) / (
                TN_result_dict[key] + FP_result_dict[key])

    print("accuracy_dict: ", accuracy_dict)
    # print("precision_for_true: ", precision_for_true)
    # print("precision_for_no: ", precision_for_no)
    # print("recall_for_no: ", recall_for_no)
    print("accuracy: ", accuracy_sum / len(accuracy_dict))


def demo(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    test_data = jsonlload(args.test_data)

    model = CustomClassifier(args, 2, len(tokenizer))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    for data in tqdm(test_data):
        tokenized_data = tokenizer(data['input'], padding='max_length', max_length=args.max_len, truncation=True)

        input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
        attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)

        with torch.no_grad():
            _, logits = model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=-1)
        data['output'] = int(predictions[0])

    jsonldump(test_data, args.output_dir + 'result.jsonl')


if __name__ == '__main__':

    args = parse_args()

    if args.do_train:
        train(args)
    elif args.do_demo:
        demo(args)
    elif args.do_test:
        test(args)
