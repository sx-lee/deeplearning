import argparse
import json
import logging
import os
import sys

import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from datasets import Dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Argument parsing for training configurations
parser = argparse.ArgumentParser(prog="train", description="Train Table to Text with BART")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--download", type=str, required=True, help="output directory path to save artifacts")
g.add_argument("--model-path", type=str, default="klue/roberta-base", help="model file path")
g.add_argument("--tokenizer", type=str, default="klue/roberta-base", help="huggingface tokenizer path")
g.add_argument("--max-seq-len", type=int, default=512, help="max sequence length")
g.add_argument("--batch-size", type=int, default=64, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=64, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help="the number of gradient accumulation steps")
g.add_argument("--epochs", type=int, default=10, help="the number of training epochs")
g.add_argument("--learning-rate", type=float, default=3e-6, help="max learning rate")
g.add_argument("--weight-decay", type=float, default=0.000, help="weight decay")
g.add_argument("--seed", type=int, default=42, help="random seed")


def main(args):
    logger = logging.getLogger("train")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguments ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore

    logger.info(f'[+] Load Tokenizer"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f'[+] Load Dataset')
    train_ds = Dataset.from_json("../data/processed_data.jsonl")  # 합쳐진 데이터 파일을 로드
    valid_ds = Dataset.from_json("../data/dev_voted.jsonl")

    labels = list(train_ds["output"][0].keys())
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)

    def preprocess_data(examples):
        text1 = examples["input"]["history"]
        text2 = examples["input"]["bot_answer"]
        encoding = tokenizer(text2, text1, padding="max_length", truncation=True, max_length=args.max_seq_len)

        encoding["labels"] = [0.0] * len(labels)
        for key, idx in label2id.items():
            if examples["output"][key] == 'yes':
                encoding["labels"][idx] = 1.0

        return encoding

    encoded_tds = train_ds.map(preprocess_data, remove_columns=train_ds.column_names)
    encoded_vds = valid_ds.map(preprocess_data, remove_columns=valid_ds.column_names)

    logger.info(f'[+] Load Model from "{args.model_path}"')
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    model.resize_token_embeddings(len(tokenizer))

    targs = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    def multi_label_metrics(predictions, labels, threshold=0.5):
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        metrics = {'f1': f1_micro_average, 'roc_auc': roc_auc}
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, labels=p.label_ids)
        return result

    trainer = Trainer(
        model,
        targs,
        train_dataset=encoded_tds,
        eval_dataset=encoded_vds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == "__main__":
    exit(main(parser.parse_args()))
