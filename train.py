from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainerCallback # miyabi用の変更
from trl import SFTTrainer, SFTConfig
import wandb
import glob
import math
from nltk import Tree

class StopAfterOneEpochCallback(TrainerCallback): # 1エポックで停止するコールバック
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_training_stop = True

def main():
    max_seq_length = 4096
    lora_rank = 64

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/qwen3-14B",
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = 16,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        lora_dropout = 0.0,
    )

    dataset = load_dataset("json", data_files={"train": "./datasets/train.jsonl", "val": "./datasets/val.jsonl"})

    dataset = dataset.filter(lambda x: x['next_action'] == 'finish') # finishのデータのみを使用

    def preprocess(example):
        tree = Tree.fromstring(example['current_state'])
        prompt = " ".join(tree.leaves()) + "\n"

        tagged = tree.pos()
        tags = [tag for _, tag in tagged]

        target = " ".join(tags)

        text = f"{prompt}{target}{tokenizer.eos_token}"

        tokenized_inputs = tokenizer(
            text,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )
        
        labels = tokenized_inputs["input_ids"].copy()
        prompt_token = tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_token["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        tokenized_inputs["labels"] = labels

        return tokenized_inputs


    train_dataset = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
    val_dataset = dataset["val"].map(preprocess, remove_columns=dataset["val"].column_names)

    batch_size = 16
    server_name = "miyabi"
    directory_number = "1"
    wandb.init(
        project="supertagging_with_LLM",
        group=server_name,
        id=f"train_{server_name}-{directory_number}",
        name=f"train_{server_name}-{directory_number}",
        resume="allow",
    )

    checkpoint_dirs = glob.glob(f"./checkpoints_dir/checkpoints_train_{server_name}-{directory_number}/checkpoint-*")

    config = SFTConfig(
        do_train=True,
        do_eval=True,
        output_dir=f"./checkpoints_dir/checkpoints_train_{server_name}-{directory_number}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size//2,
        gradient_accumulation_steps=1,
        num_train_epochs=10,
        learning_rate=1e-5,
        optim="paged_adamw_32bit", # paged_adamwは8bitか32bit
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="wandb",
        save_strategy="epoch",
        warmup_steps=math.ceil(len(train_dataset) / batch_size * 1),
        eval_strategy="epoch",
        # metric_for_best_model="eval_loss", miyabi用の変更
        # load_best_model_at_end=True, # miyabi用の変更
        save_total_limit=10,
        logging_strategy="epoch",
        # greater_is_better=False, # miyabi用の変更
        eval_accumulation_steps=1,
        group_by_length=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, # 引数名はeval_datasetと決まっている
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        callbacks=[StopAfterOneEpochCallback()], # miyabi用の変更
    )

    trainer.train(resume_from_checkpoint=bool(checkpoint_dirs))

    # trainer.save_model(f"./best-model_dir/train_{server_name}-{directory_number}") # miyabi用の変更

    wandb.finish()

if __name__ == "__main__":
    main()