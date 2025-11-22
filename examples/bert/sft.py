"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/bert/sft.py
    
- 8 GPUs (DDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml \
        examples/bert/sft.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (DDP):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "ddp" \
        --script_path "examples/bert/sft.py"

- 2 Nodes, 16 GPUs (DDP):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "ddp" \
        --script_path "examples/bert/sft.py"
"""

import os
from dataclasses import dataclass, field
from functools import partial

import datasets
import transformers
import accelerate

import dllm

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "answerdotai/ModernBERT-large"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca"
    max_length: int = 512
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )
    balance_datasets: bool = field(
        default=False,
        metadata={"help": "Balance datasets by truncating to shortest dataset size"},
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/ModernBERT-large/alpaca"
    group_by_length: bool = True
    learning_rate: float = 1e-4
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    eval_steps: float = 0.1
    save_steps: float = 0.1


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        # Parse dataset specs (pipe-separated)
        specs = [s.strip() for s in data_args.dataset_args.split("|") if s.strip()]

        if data_args.balance_datasets and len(specs) > 1:
            # Load each dataset separately to balance them
            all_datasets = []
            for spec in specs:
                ds = dllm.data.load_sft_dataset(
                    spec, load_preprocessed_data=data_args.load_preprocessed_data
                )
                all_datasets.append(ds)

            # Find minimum train size across all datasets
            min_train_size = min(len(ds["train"]) for ds in all_datasets)
            logger.info(
                f"Balancing {len(specs)} datasets to min train size: {min_train_size}"
            )

            # Truncate each dataset to min size and merge
            balanced_parts = []
            for ds in all_datasets:
                balanced_ds = datasets.DatasetDict(
                    {
                        split: (
                            ds[split].select(range(min(len(ds[split]), min_train_size)))
                            if split == "train"
                            else ds[split]
                        )
                        for split in ds.keys()
                    }
                )
                balanced_parts.append(balanced_ds)

            # Concatenate balanced datasets
            merged_splits = {}
            all_splits = set()
            for ds in balanced_parts:
                all_splits.update(ds.keys())
            for split in all_splits:
                split_datasets = [ds[split] for ds in balanced_parts if split in ds]
                merged_splits[split] = datasets.concatenate_datasets(split_datasets)
            dataset = datasets.DatasetDict(merged_splits)
        else:
            # Standard loading (single dataset or no balancing)
            dataset = dllm.data.load_sft_dataset(
                data_args.dataset_args,
                load_preprocessed_data=data_args.load_preprocessed_data,
            )

        if not data_args.load_preprocessed_data:
            map_fn = partial(
                dllm.utils.default_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )
            # Filter out invalid rows (e.g., conversations with insufficient turns)
            dataset = dataset.filter(
                dllm.utils.filter_invalid_sft_rows,
                num_proc=data_args.num_proc,
                desc="Filtering invalid SFT rows",
            )
        # truncate / filter long sequences if needed
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start training...")
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=dllm.utils.NoAttentionMaskCollator(
            tokenizer,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,  # finetune on padding <eos_token>
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
