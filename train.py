import momaf_dataset
import transformers
import bert_regressor

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert",default="TurkuNLP/bert-base-finnish-cased-v1",help="BERT basis. Default %(default)s")
    args = parser.parse_args()

    dataset=momaf_dataset.load_dataset("momaf.jsonl") #this is a list of three datasets: train,dev,test
    print(dataset)

    config = transformers.AutoConfig.from_pretrained(args.bert)
    ## Tokenizer loaded from AutoTokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert)
    ## Creating the model from the desired transformer model
    model = bert_regressor.BertRegressor.from_pretrained(args.bert, config=config)

    def encode_dataset(d):
        return tokenizer(d['content-noyear'],truncation=True)

    def make_year_target(d):
        return {"target":d["year"]}

    for k in dataset:
        dataset[k]=dataset[k].map(encode_dataset)
        dataset[k]=dataset[k].map(make_year_target)

    train_args = transformers.TrainingArguments('out.ckpt',load_best_model_at_end=True,evaluation_strategy='epoch',logging_strategy='epoch',save_strategy='epoch',
                                                learning_rate=1e-4,per_device_train_batch_size=15,num_train_epochs=10,label_names=["target"])

    trainer = transformers.Trainer(model,train_args,train_dataset=dataset['train'],eval_dataset=dataset['validation'],tokenizer=tokenizer)
    trainer.train()
