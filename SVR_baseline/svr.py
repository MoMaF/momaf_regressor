import sklearn.feature_extraction
import sklearn.svm
import momaf_dataset
import sys
import re
import random
import numpy as np

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--field",default="content-noyearnopers",help="content-orig, content-noyear, content-noyear-nopers")
    args = parser.parse_args()

    dataset=momaf_dataset.load_dataset("momaf_nonames.jsonl") #this is a list of three datasets: train,dev,test
    print(dataset)
    a=set()
    for t in ("train","validation","test"):
        s=0
        print("LEN",t,len(dataset[t]))
        for x in dataset[t]:
           s+=x["year"]
           assert x["url"] not in a
           a.add(x["url"])
        print(t,"=",s/len(dataset[t]))
    
    train_texts=[x[args.field] for x in dataset["train"]]
    train_years=[float(x["year"]) for x in dataset["train"]]

    dev_texts=[x[args.field] for x in dataset["validation"]]
    dev_years=[float(x["year"]) for x in dataset["validation"]]

    test_texts=[x[args.field] for x in dataset["test"]]
    test_years=[float(x["year"]) for x in dataset["test"]]


    vectorizer=sklearn.feature_extraction.text.TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),max_features=300000)
    train_v=vectorizer.fit_transform(train_texts)
    dev_v=vectorizer.transform(dev_texts)
    test_v=vectorizer.transform(test_texts)


    grid=[]
    for C in (1,10,100,1000):
        svr=sklearn.svm.SVR(kernel="linear",C=C)
        svr.fit(train_v,train_years)

        predicted=svr.predict(dev_v)
        diff=predicted-np.asarray(dev_years)
        err=np.mean(np.abs(diff))
        print("C=",C,"Err=",err)
        grid.append((err,C,svr))

    grid.sort()
    best_err,best_C,best_svr=grid[0]
    predicted=best_svr.predict(test_v)
    diff=predicted-np.asarray(test_years)
    err=np.mean(np.abs(diff))
    print("TEST err",np.mean(np.abs(diff)),"first 20",np.mean(np.abs(diff[:20])),"first 20 bias",np.mean(diff[:20]))
    
