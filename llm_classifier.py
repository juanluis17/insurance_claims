import pandas as pd
import os
import pickle
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt
import collections
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from transformers import pipeline
from datasets import Dataset, load_from_disk

# GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Loading claims
label2id = {"car": 0, "home": 1, "life": 2, "health": 3, "sports": 4}
id2label = {value: key for key, value in label2id.items()}

revision = 3

if os.path.exists(f"dataset.hf_{revision}"):
    train_valid = load_from_disk(f"dataset.hf_{revision}")
else:
    data = pd.read_csv("./data/insurance_dataset.csv", header=0)
    data = data.dropna().reset_index(drop=True)
    texts = data['text'].tolist()
    labels = [label2id[x] for x in data['label'].tolist()]
    c = collections.Counter(labels)
    print(c)
    dataset = Dataset.from_dict({"text": texts, "labels": labels})
    train_valid = dataset.train_test_split(test_size=0.2)
    train_valid.save_to_disk(f"dataset.hf_{revision}")

# Parameters of the models
epoch = 1000
batch_size = 3

# Directories
save_checkpoints_dir = f"./checkpoints_{revision}/"
if not os.path.exists(save_checkpoints_dir):
    os.makedirs(save_checkpoints_dir)

save_results_dir = f"./results_{revision}"
if not os.path.exists(save_results_dir):
    os.makedirs(save_results_dir)

# LLM Classifiers
estimators = [
    ["gpt2", 'gpt2'],
    ["bart_large", 'facebook/bart-large'],
    ["bert_base", 'bert-base-uncased'],
    ["distilbert", 'distilbert-base-uncased'],
    ["roberta_base", 'roberta-base'],
]

print("Starting...")
for classifier_name, classifier_dir in estimators:
    print(f'\tTesting {classifier_name}')
    model_dir = os.path.join(save_checkpoints_dir, classifier_name)
    model_dir_results = os.path.join(save_results_dir, f'{classifier_name}_preds.pkl')
    if os.path.exists(model_dir_results):
        pred_labels, scores = pickle.load(open(model_dir_results, 'rb'))
    else:
        if os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, num_labels=len(label2id), id2label=id2label, label2id=label2id
            )
        else:
            if any(k in classifier_dir for k in ("gpt", "opt", "bloom")):
                padding_side = "left"
            else:
                padding_side = "right"
            tokenizer = AutoTokenizer.from_pretrained(classifier_dir, padding_side=padding_side,
                                                      trust_remote_code=True)
            if getattr(tokenizer, "pad_token_id") is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.pad_token = tokenizer.eos_token

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


            def preprocess_function(examples):
                return tokenizer(examples["text"], truncation=True)


            train_valid_tokenized = train_valid.map(preprocess_function, batched=True)
            train_valid_tokenized = train_valid_tokenized.remove_columns("text")

            model = AutoModelForSequenceClassification.from_pretrained(
                classifier_dir, num_labels=len(label2id), id2label=id2label, label2id=label2id
            )
            if 'gpt' in model_dir:
                model.config.pad_token_id = model.config.eos_token_id
            if 'bart' in model_dir:
                batch_size = 2
            training_args = TrainingArguments(
                output_dir=model_dir,
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epoch,
                weight_decay=0.01,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                save_total_limit=1,
                logging_steps=10,
            )
            callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_valid_tokenized['train'],
                eval_dataset=train_valid_tokenized['test'],
                tokenizer=tokenizer,
                callbacks=callbacks,
                data_collator=data_collator
            )
            trainer.train()
            model.save_pretrained(model_dir)
        classifier = pipeline("text-classification", model=model, tokenizer=classifier_dir)
        test_data = pd.read_csv("./data/test_insurance_dataset.csv", header=0)
        test_data = test_data.dropna().reset_index(drop=True)
        texts = test_data['text'].tolist()
        labels = [label2id[x] for x in test_data['label'].tolist()]

        predictions = []
        scores = []
        for text in texts:
            prediction = classifier(text)
            predictions.append(label2id[prediction[0]["label"]])
            scores.append(prediction[0]["score"])

        res = classification_report(y_true=labels, y_pred=predictions, output_dict=True,
                                    target_names=["car", "home", "life", "health", "sports"])
        acc_ = accuracy_score(y_true=labels, y_pred=labels)
        roc_au_score_macro = roc_auc_score(labels, scores, average='macro')
        roc_au_score_micro = roc_auc_score(labels, scores, average='micro')
        print(res)
        print(acc_)
