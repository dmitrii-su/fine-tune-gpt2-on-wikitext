# fine-tune-gpt2-on-wikitext
This is test project attenmped at finetune gpt-2 (smallest) on wikitext dataset(wikitext-103-raw-v1)
This toy repo includes two approaches: full finetune and LoRa finetune using transformers and peft library respectively
## How to use:
0. Clone repo
1. Install dependencies: 'pip install -r requirements.txt'
2. Run by 'python3 train_gpt2.py'  - by default uses LoRa, change to  -train_type = full to train entire model
3. Use 'python3 train_gpt2.py -h' for more options
4. Use tensorboard to view logs      
