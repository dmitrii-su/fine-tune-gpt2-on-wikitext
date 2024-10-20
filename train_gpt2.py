from transformers import (AutoModelForCausalLM, AutoTokenizer, default_data_collator, Trainer, TrainingArguments)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import evaluate
import argparse
import torch
from itertools import chain


class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)



def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = model.config.eos_token_id


    ###if lora

    if args.train_type == 'lora':

        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.lm_head = CastOutputToFloat(model.lm_head)

        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=32,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'
        )

        model = get_peft_model(model, config)

        print_trainable_parameters(model)

    ##endif lora



    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", args.dataset)
    column_names_train = list(dataset['train'].features)
    column_names_val = list(dataset['test'].features)

    train = dataset['train']
    val = dataset['test']


    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True)

    def group_texts(examples):
        block_size = tokenizer.model_max_length
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    def compute_metrics(eval_pred):
        bleu = evaluate.load("bleu")
        predictions, labels = eval_pred
        predictions = predictions[0]
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return bleu.compute(predictions=decoded_preds, references=decoded_labels)

    train = train.map(preprocess, remove_columns=column_names_train, batched=True)
    train = train.map(group_texts, batched=True)
    val = val.map(preprocess, remove_columns=column_names_val, batched=True)
    val = val.map(group_texts, batched=True)

    tgt_dir = 'lora' if args.train_type == 'lora' else 'full'

    bsize = args.bsize_lora if args.train_type == 'lora' else args.bsize_full

    training_args = TrainingArguments(
        output_dir=args.out_folder+tgt_dir,
        logging_dir=args.log_folder+tgt_dir,
        learning_rate=args.lrate,
        per_device_train_batch_size=bsize,
        per_device_eval_batch_size=bsize,
        num_train_epochs=args.nepoch,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        report_to = "tensorboard",
        fp16 = args.use_fp16 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()





if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='openai-community/gpt2')
    parser.add_argument("--dataset", type=str, default='wikitext-103-raw-v1')

    parser.add_argument("--train_type", type=str, default='lora', choices=['lora', 'full'])
    parser.add_argument("--lora_r", type=int, default=16)

    parser.add_argument("--bsize_lora", type=int, default=32)
    parser.add_argument("--bsize_full", type=int, default=4)
    parser.add_argument("--use_fp16", type=bool, default=False)
    

    parser.add_argument("--nepoch", type=int, default=2)
    parser.add_argument("--lrate", type=float, default=0.0005)

    parser.add_argument("--out_folder", type=str, default='./output_folder/')
    parser.add_argument("--log_folder", type=str, default='./log_folder/')
    args = parser.parse_args()
    main(args)
