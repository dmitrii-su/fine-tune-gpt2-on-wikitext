# fine-tune-gpt2-on-wikitext
This is the test project aimed at finetuning gpt-2 (smallest) on wikitext dataset(wikitext-103-raw-v1)
This toy repo includes two approaches: full finetune and LoRa finetune using transformers and peft library respectively. 
So one can compare performance of two approaches. 
## How to use:
0. Using terminal clone this repo, change folder `cd fine-tune-gpt2-on-wikitext`
1. Install dependencies: `pip install -r requirements.txt`
2. Run by `python3 train_gpt2.py`  - by default uses LoRa approach, change to  `python3 train_gpt2.py -train_type = full` to train entire model
3. Use `python3 train_gpt2.py -h` for more options and see defauls
4. Use `tensorboard --logdir log_dir` to view logs
## My results:
1. Efficiency: LoRa uses less memory, since it has only appx. 0.5%  of trainable params of full number model params.  Hence bigger batch size is possible, bigger batch size -> faster tuning (almost two times in this case).
2. Comparable perfomance: Both LoRa and full finetune showed similar resutls on test set using bleu metric. 
      
