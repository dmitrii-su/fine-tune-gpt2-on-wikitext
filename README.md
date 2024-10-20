# fine-tune-gpt2-on-wikitext
This is the test project aimed at finetuning gpt-2 (smallest) on wikitext dataset(wikitext-103-raw-v1)
This toy repo includes two approaches: full finetune and LoRa finetune using transformers and peft library respectively. 
So one can compare performance of two approaches. 
## How to use:
0. Clone repo
1. Install dependencies: `pip install -r requirements.txt`
2. Run by `python3 train_gpt2.py`  - by default uses LoRa approach, change to  -train_type = full to train entire model
3. Use `python3 train_gpt2.py -h` for more options and see defauls
4. Use tensorboard to view logs
## Results:
1. LoRa uses less memory, since it has only appx. 0.5%  trainable params of full number of models params.  hence bigger batch size is possible, bigger batch size -> faster tuning (almost two times in this case)
2. 
      
