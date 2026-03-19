from unsloth import FastLanguageModel
from nltk import Tree
import wandb

def predict_supertag(input_text, model, tokenizer):
    prompt = input_text + "\n"

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
    )

    input_length = inputs.input_ids.shape[1]

    generated_tokens = outputs[0, input_length:]
    action_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return action_text

def main():
    server_name = "miyabi"
    directory_number = "1"
    file_number = 3

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = f"checkpoints_dir/checkpoints_train_{server_name}-{directory_number}/checkpoint-{124495 * file_number}",
        max_seq_length = 4096,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    with open("./datasets/test-ans.txt", "r") as f:
        gold_lines = f.readlines()
        gold_lines = [line.strip() for line in gold_lines]

    with open("./datasets/test-sentence.txt", "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    
    wandb.init(
        project="supertagging_with_LLM",
        group=server_name,
        name=f"test_{server_name}-{directory_number}",
    )

    pred_tags = []
    for line in lines:
        pred_tag = predict_supertag(line, model, tokenizer)
        pred_tags.append(pred_tag.split()) # [[A, B, C], [D, E, F], ...]
    
    coverage_count = 0
    total_count = 0
    same_count = 0
    total_count = 0
    for pred_tag, gold in zip(pred_tags, gold_lines):
        gold_tree = Tree.fromstring(gold)
        gold_tag = [tag for _, tag in gold_tree.pos()]

        if len(pred_tag) == len(gold_tag):
            same_count += sum(1 for x, y in zip(pred_tag, gold_tag) if x == y)
            total_count += len(gold_tag)
            coverage_count += 1
        
        total_count += 1
    
    print(f"Coverage: {coverage_count / total_count:.4f}")
    print(f"Accuracy: {same_count / total_count:.4f}")

if __name__ == "__main__":
    main()