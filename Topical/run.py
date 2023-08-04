import argparse
import ast
import logging
import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import classification_report, label_ranking_average_precision_score
from sklearn.model_selection import ShuffleSplit, train_test_split
from transformers import TrainingArguments

from attention_classifier import Attention, Classifier, ClassifierTrainer, Encoder
from dataset import Dataset, fit_label_binarizer, fit_pca_components
from tagging_task import calculate_f1_score

load_dotenv()
DATASET_PATH = os.getenv("PREPROCESSED_DATASET_PATH")
DATASET_CSV = os.getenv("CSV_DATASET")

logging.getLogger().setLevel(logging.INFO)


def compose_embeddings(
    dataset: pd.DataFrame, docstring: bool = True, code: bool = True, dep: bool = True
) -> List:
    """
    Method to concatenate the hybrid embedding after PCA projection  of each component.

    :param dataset: DataFrame containing the projected PCA embedding for each repository
    :param docstring: Boolean to indicate whether to add docstring embeddings in the hybrid embedding
    :param code: Boolean to indicate whether to add code embeddings in the hybrid embedding
    :param dep: Boolean to indicate whether to add dependency embeddings in the hybrid embedding
    :return: concatenated embeddings for each script in each repository
    """
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
            np.concatenate((e1,e2), axis=1)
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
        # print(embeddings[0].shape)
    elif code:
        embeddings = dataset["code_emb"].tolist()
    elif dep:
        embeddings = dataset["dep_emb"].tolist()
    elif docstring:
        embeddings = dataset["docstring_emb"].tolist()
    else:
        raise ValueError
    return embeddings


def topics_dataset(topics: str, dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, row in dataset.iterrows():
        if isinstance(row["labels"], str):
            row["labels"] = ast.literal_eval(row["labels"])
        else:
            pass
        intersection = set(topics).intersection(set(row["labels"]))
        if len(intersection) > 0:
            row["labels"] = list(intersection)
            rows.append(row)
        filtered_dataset = pd.DataFrame(rows)
    return filtered_dataset


def main():
    os.environ["WANDB_DISABLED"] = "true"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="GRU", help="type of recurrent net [LSTM, GRU]"
    )
    parser.add_argument(
        "--emsize",
        type=int,
        default=128,
        help="size of word embeddings [Uses pretrained on 50, 100, 200, 300]",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=96,
        help="number of hidden units for the RNN encoder",
    )
    parser.add_argument(
        "--nlayers", type=int, default=2, help="number of layers of the RNN encoder"
    )
    parser.add_argument("--lr", type=float, default=0.002, help="initial learning rate")
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
    parser.add_argument("--k", type=int, default=10, help="Number of tries")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.normpath(
            os.path.join(os.getcwd(), DATASET_PATH, "resources.pkl")
        ),
        help="Path to dataset embedding",
    )
    parser.add_argument("--n_pca_comp", type=int, default=192)
    parser.add_argument("--topics", type=str, nargs="+", default=["Computervision"])
    parser.add_argument("--dep_embedding",action="store_false")
    parser.add_argument("--script_embedding",action="store_false")
    parser.add_argument("--doc_embedding",action="store_false")
    parser.add_argument("--use_pca",action="store_true") # False if you don't pass the argument

    args = parser.parse_args()
    print(args.topics)
    use_dep_embedding = args.dep_embedding
    use_script_embedding = args.script_embedding
    use_doc_embedding = args.doc_embedding
    print("Dep",use_dep_embedding)
    print("Code",use_script_embedding)
    print("Docstring", use_doc_embedding)
    which_embedding_to_use = [use_dep_embedding,use_doc_embedding,use_script_embedding]
    num_embeddings = int(use_dep_embedding) + int(use_script_embedding) + int(use_doc_embedding)
    logging.info("[Model hyperparams]: {}".format(str(args)))

    dataset = pd.read_pickle(args.dataset)
    dataset = topics_dataset(args.topics, dataset)
    reports = []
    rs = ShuffleSplit(n_splits=args.k, test_size=0.3)
    j = 0

    for train_index, test_index in rs.split(dataset):

        embedding_types = ['dep_emb','docstring_emb','code_emb']
        final_embedding_types = [types for i,types in enumerate(embedding_types) if which_embedding_to_use[i]]
        logging.info(f"run {j}")
        train = dataset.iloc[train_index]
        test = dataset.iloc[test_index]
        train_labels = train["labels"].tolist()
        test_labels = test["labels"].tolist()
        le = fit_label_binarizer(train_labels)

        if args.use_pca:
            embedding_dim = args.n_pca_comp
            logging.info(f"Fitting PCA with {args.n_pca_comp//num_embeddings} each")
            pca_comps = [
                fit_pca_components(train_emb, args.n_pca_comp // num_embeddings)
                for train_emb in [dataset[types].tolist() for types in final_embedding_types]
            ]
            for index, pca_comp in enumerate(pca_comps):
                print(np.array(train[final_embedding_types[index]])[0].shape)
                train[final_embedding_types[index]] = np.array([pca_comp.transform(emb) for emb in train[final_embedding_types[index]]])
                train[final_embedding_types[index]] = np.array(train[final_embedding_types[index]])
                print(np.array(train[final_embedding_types[index]])[0].shape,'hi')
                test[final_embedding_types[index]] = np.array([pca_comp.transform(emb) for emb in test[final_embedding_types[index]]])
                test[final_embedding_types[index]] = np.array(test[final_embedding_types[index]])
        else:
            for index, pca_comp in enumerate(final_embedding_types):
                train[final_embedding_types[index]] = np.array(train[final_embedding_types[index]])
                test[final_embedding_types[index]] = np.array(test[final_embedding_types[index]])
            print(np.array(train[final_embedding_types[index]])[0].shape)
            embedding_dim = num_embeddings*np.array(train[final_embedding_types[index]])[0].shape[-1]
        
        train_embeddings = compose_embeddings(train,use_doc_embedding, use_script_embedding, use_dep_embedding)
        test_embeddings = compose_embeddings(test,use_doc_embedding, use_script_embedding, use_dep_embedding)
        train_ds = Dataset(train_labels, train_embeddings, le)
        test_ds = Dataset(test_labels, test_embeddings, le)
        nlabels = len(train_ds.le.classes_)
        logging.info(f"Number of labels: {nlabels}")
        encoder = Encoder(
            embedding_dim=embedding_dim,
            hidden_dim=args.hidden,
            nlayers=args.nlayers,
            dropout=args.drop,
            bidirectional=args.bi,
            rnn_type=args.model,
            use_pca = args.use_pca,
            num_pca_components = args.n_pca_comp
        )
        attention_dim = args.hidden if not args.bi else 2 * args.hidden
        attention = Attention(attention_dim, attention_dim, attention_dim)
        model = Classifier(encoder, attention, attention_dim, num_classes=nlabels)
        logging.info(f"Model {j} initiated")
        j += 1
        training_args = TrainingArguments(
            overwrite_output_dir=True,
            output_dir="./results_classifier",
            num_train_epochs=20,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./results_classifier",
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
            predicted_labels,
            gt_labels,
            train_size=0.75,
            random_state=random.randint(0, 100),
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
    recall = (
        np.mean([report["weighted avg"]["recall"] for report in reports]),
        np.std([report["weighted avg"]["recall"] for report in reports]),
    )
    precision = (
        np.mean([report["weighted avg"]["precision"] for report in reports]),
        np.std([report["weighted avg"]["precision"] for report in reports]),
    )
    f1_score_ = (
        np.mean([report["weighted avg"]["f1-score"] for report in reports]),
        np.std([report["weighted avg"]["f1-score"] for report in reports]),
    )

    # LRAP SCORES

    lrap_mean = np.mean([report["lrap"] for report in reports])
    lrap_std = np.std([report["lrap"] for report in reports])

    # Support
    support = np.mean([report["weighted avg"]["support"] for report in reports])
    # Threshold

    thresold = (
        np.mean([report["optimized_threshold"] for report in reports]),
        np.std([report["optimized_threshold"] for report in reports]),
    )
    report = "\n".join(
        [
            f"LRAP: {np.round(lrap_mean, 3)} ({np.round(lrap_std, 3)})",
            f"recall: {np.round(recall[0], 3)} ({np.round(recall[1], 3)})",
            f"precision: {np.round(precision[0], 3)} ({np.round(precision[1], 3)})",
            f"f1-score: {np.round(f1_score_[0], 3)} ({np.round(f1_score_[1], 3)})",
            f"threshold: {np.round(thresold[0], 3)} ({np.round(thresold[1], 3)})",
            f"support: {support})",
            "------------------------",
        ]
    )
    output_path = os.path.dirname(args.dataset)
    print("results saved to: ", output_path)
    if not os.path.exists(os.path.join(output_path, "results_datasize.txt")):
        with open(os.path.join(output_path, "results_datasize.txt"), "w") as f:
            f.write(report)
    else:
        with open(os.path.join(output_path, "results_datasize.txt"), "a") as f:
            f.write("\n")
            f.write(report)


if __name__ == "__main__":
    main()
