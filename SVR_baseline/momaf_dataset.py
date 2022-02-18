import datasets


def load_dataset(fname):
    dataset=datasets.load_dataset('json',data_files={"train":fname,"validation":fname,"test":fname},split=["train[:80%]","validation[80%:90%]","test[90%:]"])
    return {"train":dataset[0],"validation":dataset[1],"test":dataset[2]}

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    dataset=load_dataset("momaf.jsonl")
    print(dataset)
