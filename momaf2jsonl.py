import json
import sys
import traceback
import random
import re
random.seed(0)

def preproc(txt):
    txt=txt.replace("<I>","")
    txt=txt.replace("</I>","")
    txt=txt.replace("<i>","")
    txt=txt.replace("</i>","")
    txt=txt.replace("<br />"," ")
    return txt

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    all_momaf=json.load(sys.stdin)["results"]["bindings"]

    movies=[]
    for mov in all_momaf:
        try:
            url=mov["filmiri"]["value"]
            assert "http" in url
            id=mov["id"]["value"]
            year=int(mov["year"]["value"])
            content=preproc(mov["contentdescription"]["value"])
            mov_dict={"id":id,"url":url,"year":year,"content-orig":content}
            mov_dict["content-noyear"]=re.sub("[0-9]","0",content)
            movies.append(mov_dict)
        except:
            pass
    random.shuffle(movies)
    for m in movies:
        print(json.dumps(m,ensure_ascii=False,sort_keys=True))
    #print(list(all_momaf[0]))
    
    
