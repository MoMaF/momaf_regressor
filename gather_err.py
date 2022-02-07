import sys
import json
import os

for fname in sys.argv[1:]:
    with open(fname,"rt") as f:
        for line in f:
            if line.startswith("{'eval_loss"):
                data=eval(line.strip())
                print(os.path.basename(fname),data["eval_loss"],data["epoch"],sep="\t")
