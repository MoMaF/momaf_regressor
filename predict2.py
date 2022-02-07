import momaf_dataset
import transformers
import bert_regressor
import sys
import re

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsize",default=30,type=int,help="Batch size %(default)d")
    parser.add_argument("--load-from",default=None,help="Path to a model")
    parser.add_argument("--field",default="content-noyearnopers",help="content-orig, content-noyear, content-noyear-nopers")
    parser.add_argument("--sep",default=False, action="store_true",help="populate with SEP")
    args = parser.parse_args()

    dataset=momaf_dataset.load_dataset("momaf_nonames.jsonl") #this is a list of three datasets: train,dev,test
    ## Tokenizer loaded from AutoTokenizer

    ## Creating the model from the desired transformer model
    if args.load_from:
        model=bert_regressor.BertRegressor.from_pretrained(args.load_from)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.load_from)

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

    #train_args = transformers.TrainingArguments('out.ckpt',evaluation_strategy='steps',eval_steps=30, logging_strategy='steps',save_strategy='steps',save_steps=30,save_total_limit=3,
    #                                            learning_rate=args.lr,per_device_train_batch_size=args.bsize,gradient_accumulation_steps=args.grad_acc,max_steps=args.steps, logging_steps=5, label_names=["target"],load_best_model_at_end=True,warmup_steps=150)

    trainer = transformers.Trainer(model,tokenizer=tokenizer)
    x=trainer.predict(dataset["test"])
    print(x.predictions)
