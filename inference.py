import argparse
from datasets import load_dataset, load_from_disk
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Path or name of the model.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SeaLLMs/FreshQA-multilingual",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./transferability_results/7B/Qwen_base_7B.json",
        help="Output path for JSON results.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["identical", "mean shifting", "linear projection"],
        help="List of methods to apply.",
    )
    parser.add_argument(
        "--use_template",
        type=bool,
        default=True,
        help="Whether to use template in encoding.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="encoding batch size"
    )
    return parser.parse_args()


def get_num_layers(model):
    """Determine the number of layers in the model."""
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    elif hasattr(model.config, "n_layer"):
        return model.config.n_layer
    else:
        raise ValueError(
            "Cannot determine the number of layers from the model configuration."
        )


def encode(texts, tokenizer, model, batch_size=50, num_layer=81, template=True):
    last_token_representations = {n: [] for n in range(num_layer)}
    for start_index in tqdm(range(0, len(texts), batch_size)):
        batch = texts[start_index : start_index + batch_size]
        if template:
            messages_list = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                for prompt in batch
            ]
            inputs = [
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in messages_list
            ]
            model_inputs = tokenizer(
                inputs, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)
        else:
            model_inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)
        model.eval()
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        for index, layer_hidden_states in enumerate(hidden_states):
            if template:
                last_token_states = layer_hidden_states[:, -6, :].cpu()
            else:
                last_token_states = layer_hidden_states[:, -1, :].cpu()
            last_token_representations[index].append(last_token_states)

    last_token_representations = {
        n: torch.vstack(last_token_representations[n]) for n in range(num_layer)
    }
    return last_token_representations


def mean_shifting(in_train_X, ood_train_X, ood_test_X):
    in_mean = torch.mean(in_train_X, axis=0)
    out_mean = torch.mean(ood_train_X, axis=0)
    mean_diff = in_mean - out_mean
    shifted_test_X = ood_test_X + mean_diff
    return shifted_test_X


def linear_projection(in_train_X, ood_train_X, ood_test_X):
    W, residuals, rank, s = np.linalg.lstsq(ood_train_X, in_train_X, rcond=None)
    shifted_test_X = ood_test_X @ W
    return shifted_test_X


def linear_projection_with_centered(in_train_X, ood_train_X, ood_test_X):
    in_mean = torch.mean(in_train_X, axis=0)
    out_mean = torch.mean(ood_train_X, axis=0)
    in_train_X -= in_mean
    ood_train_X -= out_mean
    W, residuals, rank, s = np.linalg.lstsq(ood_train_X, in_train_X, rcond=None)
    ood_test_X -= out_mean
    shifted_test_X = ood_test_X @ W
    # shifted_test_X += out_mean
    shifted_test_X += in_mean
    return shifted_test_X


def subspace_transformation(
    in_train_X, ood_train_X, ood_test_X, method="linear projection"
):
    if method == "linear projection":
        shifted_test_X = linear_projection(in_train_X, ood_train_X, ood_test_X)
    elif method == "linear projection with center":
        shifted_test_X = linear_projection_with_centered(
            in_train_X, ood_train_X, ood_test_X
        )
    elif method == "mean shifting":
        shifted_test_X = mean_shifting(in_train_X, ood_train_X, ood_test_X)
    elif method == "identical":
        shifted_test_X = ood_test_X
    else:
        print("not implemented")
    return shifted_test_X


def main():
    args = parse_args()

    dataset = load_dataset(args.dataset_name)["test"]
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    languages = [
        "English",
        "Chinese",
        "Vietnamese",
        "Thai",
        "Khmer",
        "Indonesian",
        "Malay",
        "Lao",
    ]
    question_dict = {language: dataset[language] for language in languages}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )
    if "Llama" in args.model_name:
        tokenizer.pad_token = tokenizer.bos_token
        model.config.pad_token_id = model.config.bos_token_id

    num_layer = get_num_layers(model) + 1

    representation_dict = {
        language: encode(
            question_dict[language],
            tokenizer,
            model,
            batch_size=args.batch_size,
            num_layer=num_layer,
            template=args.use_template,
        )
        for language in languages
    }

    transferability_dict = {}

    n_splits = 5

    for method in args.methods:
        method_dict = {}
        for n in tqdm(range(num_layer), desc=f"processing method {method}"):
            layer_result = {}
            for language in languages:
                X = representation_dict[language][n].to(torch.float32)
                y = dataset["false_premise"]

                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

                scores = np.zeros([n_splits, len(languages)])
                for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    train_X_dict = {}
                    test_X_dict = {}
                    for lang in languages:
                        train_X_dict[lang] = representation_dict[lang][n][train_index]
                        test_X_dict[lang] = representation_dict[lang][n][val_index]

                    lgr = LogisticRegression(max_iter=1000, random_state=42)
                    lgr.fit(X_train, y_train)

                    for score_index, lang in enumerate(languages):
                        vanilla_train_X = train_X_dict[lang].to(torch.float32)
                        vanilla_test_X = test_X_dict[lang].to(torch.float32)

                        shifted_test_X = subspace_transformation(
                            X_train, vanilla_train_X, vanilla_test_X, method=method
                        )
                        y_pred = lgr.predict(
                            shifted_test_X
                        )

                        score = accuracy_score(y_val, y_pred)
                        scores[fold - 1, score_index] = score

                mean_score = np.mean(scores, axis=0)
                layer_result[language] = mean_score
            method_dict[n] = layer_result
        transferability_dict[method] = method_dict

    def convert_ndarrays_to_lists(d):
        if isinstance(d, dict):
            return {k: convert_ndarrays_to_lists(v) for k, v in d.items()}
        elif isinstance(d, (list, tuple, np.ndarray)):
            return [convert_ndarrays_to_lists(element) for element in d]
        return d

    transferability_dict_serializable = convert_ndarrays_to_lists(transferability_dict)

    with open(args.output_path, "w") as json_file:
        json.dump(transferability_dict_serializable, json_file, indent=4)


if __name__ == "__main__":
    main()
