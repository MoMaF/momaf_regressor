import momaf_dataset
import transformers
import bert_regressor
import sys
import re
import torch

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-from",default=None,help="Path to a model")
    parser.add_argument("--field",default="content-noyearnopers",help="content-orig, content-noyear, content-noyear-nopers")
    parser.add_argument("--sep",default=False, action="store_true",help="populate with SEP")
    args = parser.parse_args()

    dataset=momaf_dataset.load_dataset("momaf_nonames.jsonl") #this is a list of three datasets: train,dev,test

    ## Tokenizer loaded from AutoTokenizer
    
    ## Creating the model from the desired transformer model
    if args.load_from:
        model=bert_regressor.BertRegressor.from_pretrained(args.load_from)
        tokenizer=transformers.AutoTokenizer.from_pretrained(args.load_from)

    def encode_dataset(d):
        txt=d[args.field] #WATCH OUT THIS GLOBAL VARIABLE
        if args.sep:
            txt=re.sub(r"([.?])\s+([A-ZÄÅÖ])",r"\1 [SEP] \2",txt)
        return tokenizer(txt,truncation=True)

    def make_year_target(d):
        return {"target":(d["year"]-1970)/10.0}

    for k in dataset:
        dataset[k]=dataset[k].map(encode_dataset)
        dataset[k]=dataset[k].map(make_year_target)

    with torch.no_grad():
        for e in dataset["test"]:
            o=model(torch.tensor(e["input_ids"],device=model.device).unsqueeze(0),torch.tensor(e["attention_mask"],device=model.device).unsqueeze(0))
            p=o.prediction[0][0].item()*10+1970
            print(e["url"],e["year"],p,p-e["year"],sep="\t")
            
        
    # train_args = transformers.TrainingArguments('out.ckpt',evaluation_strategy='steps',eval_steps=30, logging_strategy='steps',save_strategy='steps',save_steps=30,save_total_limit=3,
    #                                             learning_rate=args.lr,per_device_train_batch_size=args.bsize,gradient_accumulation_steps=args.grad_acc,max_steps=args.steps, logging_steps=5, label_names=["target"],load_best_model_at_end=True,warmup_steps=150)

    # if args.pretrain_frozen:
    #     for param in model.bert.parameters():
    #         param.requires_grad=False

    # trainer = transformers.Trainer(model,train_args,train_dataset=dataset['train'],eval_dataset=dataset['validation'],tokenizer=tokenizer,callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)])
    # trainer.train()
    # if args.save_to:
    #     trainer.save_model(args.save_to)
