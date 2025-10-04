import paddle
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from paddlenlp.trainer import Trainer, TrainingArguments
from datasets import load_dataset

def preprocess(example, tokenizer):
    x = example.get("instruction", "")
    if example.get("input", ""):
        x += "\n" + example["input"]
    input_ids = tokenizer(x, max_length=512, padding="max_length", truncation=True)["input_ids"]
    labels = tokenizer(example["output"], max_length=512, padding="max_length", truncation=True)["input_ids"]
    return {"input_ids": input_ids, "labels": labels}

def main():
    model = AutoModelForCausalLM.from_pretrained("llama-7b-zh")
    tokenizer = AutoTokenizer.from_pretrained("llama-7b-zh")
    dataset = load_dataset("swanlab/alpaca-gpt4-data-zh", split="train")
    dataset = dataset.map(lambda e: preprocess(e, tokenizer))
    args = TrainingArguments(
        output_dir="./llama_output",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=50,
        learning_rate=5e-5,
        fp16=True,
        do_train=True
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()

if __name__ == "__main__":
    main()
