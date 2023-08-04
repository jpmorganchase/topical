import argparse
import ast
import logging
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    label_ranking_average_precision_score,
)
from sklearn.model_selection import ShuffleSplit, train_test_split
from torch import nn
from transformers import TrainingArguments

from attention_classifier import Attention, Classifier, ClassifierTrainer, Encoder
from dataset import Dataset, fit_cca_components, fit_label_binarizer, fit_pca_components

# from skopt import gp_minimize
# from skopt.space.space import Categorical
logging.getLogger().setLevel(logging.INFO)


def compose_embeddings(
    dataset, docstring: bool = True, code: bool = True, dep: bool = True
) -> List:
    if docstring and code and dep:
        embeddings = [
            np.concatenate((e1, e2, e3), axis=1)
            for e1, e2, e3 in zip(
                dataset["dep_emb"].tolist(),
                dataset["code_emb"].tolist(),
                dataset["docstring_emb"].tolist(),
            )
        ]
    elif docstring and code:
        embeddings = [
            np.concatenate((e1, e2), axis=1)
            for e1, e2 in zip(
                dataset["code_emb"].tolist(), dataset["docstring_emb"].tolist()
            )
        ]
    elif docstring and dep:
        embeddings = [
            np.concatenate((e1, e2), axis=1)
            for e1, e2 in zip(
                dataset["dep_emb"].tolist(), dataset["docstring_emb"].tolist()
            )
        ]
    elif code and dep:
        embeddings = [
            np.concatenate((e1, e2), axis=1)
            for e1, e2 in zip(dataset["dep_emb"].tolist(), dataset["code_emb"].tolist())
        ]
    elif code:
        embeddings = dataset["code_emb"].tolist()
    elif dep:
        embeddings = dataset["dep_emb"].tolist()
    elif docstring:
        embeddings = dataset["docstring_emb"].tolist()
    else:
        raise ValueError
    return embeddings


def calculate_f1_score(predicted, gt):
    scores = []
    for threshold in np.linspace(0, 1, num=100):
        predicted_label = [
            [1 if o > threshold else 0 for o in label] for label in predicted
        ]
        scores.append(
            f1_score(
                np.array([l.numpy() for l in gt]),
                np.array(predicted_label),
                average="weighted",
            )
        )
    index = np.argmax(scores)
    return np.linspace(0, 1, num=100)[index]


def main():
    os.environ["WANDB_DISABLED"] = "true"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="GRU", help="type of recurrent net [LSTM, GRU]"
    )
    parser.add_argument(
        "--emsize",
        type=int,
        default=192,
        help="size of word embeddings [Uses pretrained on 50, 100, 200, 300]",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=48,
        help="number of hidden units for the RNN encoder",
    )
    parser.add_argument(
        "--nlayers", type=int, default=2, help="number of layers of the RNN encoder"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=5, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=30, help="upper epoch limit")
    parser.add_argument(
        "--batch_size", type=int, default=2, metavar="N", help="batch size"
    )
    parser.add_argument("--drop", type=float, default=0, help="dropout")
    parser.add_argument("--bi", default=True, help="[USE] bidirectional encoder")
    parser.add_argument("--cuda", action="store_false", help="[DONT] use CUDA")
    parser.add_argument(
        "--fine", action="store_true", help="use fine grained labels in SST"
    )
    parser.add_argument(
        "--repo_size",
        type=int,
        help="number of scripts per repository to be used as sequence length",
        default=15,
    )
    parser.add_argument("--k", type=int, default=4, help="Number of tries")
    parser.add_argument(
        "--dataset", type=str, default="full_resources.pkl", help="Path to dataset"
    )
    parser.add_argument("--n_cca_comp", type=int, default=128)
    parser.add_argument("--n_pca_comp", type=int, default=128)

    args = parser.parse_args()
    logging.info("[Model hyperparams]: {}".format(str(args)))

    dataset = pd.read_pickle(args.dataset)
    reports = []
    lraps = []
    rs = ShuffleSplit(n_splits=args.k)
    j = 0
    for train_index, test_index in rs.split(dataset):
        logging.info(f"run {j}")
        train = dataset.iloc[train_index]
        test = dataset.iloc[test_index]
        train_labels = [ast.literal_eval(label) for label in train["labels"].tolist()]
        test_labels = [ast.literal_eval(label) for label in test["labels"].tolist()]
        le = fit_label_binarizer(train_labels)
        train_embeddings = compose_embeddings(train)
        test_embeddings = compose_embeddings(test)
        logging.info(f"Fitting PCA with {args.n_pca_comp} each")
        print(len(train), len(dataset))
        pca = fit_pca_components(train_embeddings, args.n_pca_comp)
        train_ds = Dataset(train_labels, train_embeddings, le, pca=pca)
        test_ds = Dataset(test_labels, test_embeddings, le, pca=pca)

        nlabels = len(train_ds.le.classes_)
        logging.info(f"Number of labels: {nlabels}")
        encoder = Encoder(
            args.n_pca_comp,
            args.hidden,
            nlayers=args.nlayers,
            dropout=args.drop,
            bidirectional=args.bi,
            rnn_type=args.model,
        )
        attention_dim = args.hidden if not args.bi else 2 * args.hidden
        attention = Attention(attention_dim, attention_dim, attention_dim)
        model = Classifier(encoder, attention, attention_dim, num_classes=nlabels)
        logging.info(f"Model {j} initiated")
        j += 1

        # MODEL TRAINING
        training_args = TrainingArguments(
            overwrite_output_dir=True,
            output_dir="./results_classifier_attention_layer_pca",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./results_classifier_attention_layer_384",
            max_grad_norm=args.clip,
            learning_rate=args.lr,
            report_to=None,
            save_strategy="epoch",
        )
        trainer = ClassifierTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
        )
        try:
            logging.info(f"Beginning training phase on: {len(train)} training samples")
        except:
            pass
        trainer.train()
        predicted_labels = []
        gt_labels = []
        logging.info("Beginning testing")

        # MODEL EVALUATION : THRESHOLD FIX
        with torch.no_grad():
            for i, data in enumerate(test_ds):
                data["embeddings"] = data["embeddings"].unsqueeze(0)
                data["attention_mask"] = data["attention_mask"].unsqueeze(0)
                gt_labels.append(data["labels"])
                outputs = model(data)
                predicted_labels.append(np.array(torch.nn.Sigmoid()(outputs[0][0])))

        test_output, val_output, test_gt, val_gt = train_test_split(
            predicted_labels, gt_labels, train_size=0.75
        )
        selected_threshold = calculate_f1_score(predicted=val_output, gt=val_gt)
        logging.info(f"Selected threshold: {selected_threshold}")
        test_predicted = [
            [1 if o > selected_threshold else 0 for o in label] for label in test_output
        ]
        report = classification_report(
            np.array([l.numpy() for l in test_gt]),
            np.array(test_predicted),
            output_dict=True,
        )
        lrap = label_ranking_average_precision_score(
            np.array([l.numpy() for l in gt_labels]), np.array(predicted_labels)
        )
        report["lrap"] = lrap
        report["optimized_threshold"] = selected_threshold
        reports.append(report)
        print(report)
    with open("final_metrics_1.pkl", "wb") as f:
        pickle.dump(reports, f)


if __name__ == "__main__":
    main()
