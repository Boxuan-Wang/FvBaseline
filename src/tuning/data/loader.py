# from src.tuning.data.preprocess import *
"""
Training data loading.

NI JSON path: Natural-Instructions-style JSON via ni_dataset.py; each example needs
Task, Definition, Positive Examples, Negative Examples, Instance (input + output list)
for DataCollatorForNI.

CL preprocessed path: HuggingFace save_to_disk folders under
<cl_preprocessed_root>/<cl_task_subdir>/<train_mixed|val>/ (e.g. data-0-of-1.arrow).
Rows are normalized to the same NI shape via _ensure_ni_row.
"""
import os
from hashlib import md5
from typing import Dict, List

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def gen_ni_cache_path(cache_dir, data_args):
    hash_str = (data_args.data_dir or "") + (data_args.task_name or "") + \
               str(data_args.max_num_instances_per_task) + str(data_args.max_num_instances_per_eval_task)
    print(hash_str)
    hash_obj = md5(hash_str.encode("utf-8"))
    hash_id = hash_obj.hexdigest()
    cache_path = os.path.join(cache_dir, str(hash_id))

    return cache_path


def _load_cl_split(root: str, task_subdir: str, split_name: str) -> Dataset:
    path = os.path.join(root, task_subdir, split_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"CL preprocessed split not found: {path} "
            "(expected HuggingFace datasets save_to_disk folder with e.g. data-0-of-1.arrow)."
        )
    return load_from_disk(path)


def get_cl_preprocessed_dataset(data_args, training_args) -> DatasetDict:
    root = data_args.cl_preprocessed_root
    task_subdir = data_args.cl_task_subdir
    train_name = data_args.cl_train_split
    val_name = data_args.cl_val_split

    train_ds = _load_cl_split(root, task_subdir, train_name)
    val_ds = _load_cl_split(root, task_subdir, val_name)

    if data_args.max_num_instances_per_task is not None and data_args.max_num_instances_per_task >= 0:
        n = min(len(train_ds), data_args.max_num_instances_per_task)
        train_ds = train_ds.select(range(n))
    if data_args.max_num_instances_per_eval_task is not None and data_args.max_num_instances_per_eval_task >= 0:
        n = min(len(val_ds), data_args.max_num_instances_per_eval_task)
        val_ds = val_ds.select(range(n))

    return DatasetDict({"train": train_ds, "validation": val_ds})


def get_ni_dataset(model_args, data_args, training_args):

    if data_args.cl_preprocessed_root:
        print(
            "CL preprocessed:",
            data_args.cl_preprocessed_root,
            data_args.cl_task_subdir,
            data_args.cl_train_split,
            data_args.cl_val_split,
        )
        return get_cl_preprocessed_dataset(data_args, training_args)

    data_cache_dir = gen_ni_cache_path(model_args.cache_data_dir, data_args)

    print(data_args.data_dir, data_args.task_config_dir)

    raw_datasets = load_dataset(
        "./src/tuning/data/ni_dataset.py",
        data_dir=data_args.data_dir,
        task_name=data_args.task_name,
        cache_dir=data_cache_dir,  # for debug, change dataset size, otherwise open it
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        trust_remote_code=True
    )

    raw_datasets.cleanup_cache_files()

    return raw_datasets



def convert_format_trace(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    # convert dataset from sharegpt format to alpaca format
    outputs = {"prompt": examples["prompt"],
               "query": ["" for _ in examples["answer"]],
               "response": examples["answer"]}
    return outputs


def split_dataset(
    raw_datasets,
    data_args,
    training_args
):
    train_dataset = None
    eval_dataset = None
    predict_dataset = None
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            datasize = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(datasize))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    return {"train_dataset": train_dataset if training_args.do_train else None,
            "eval_dataset": eval_dataset if training_args.do_eval else None}, predict_dataset if training_args.do_predict else None
