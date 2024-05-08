import re
import random
from main import *
import torch
f=open("train_sft.csv","r")
text=f.read()
pattern = r'<s>Human:(.*?)</s>'
a=re.findall(pattern,text,re.DOTALL)
pattern = r'<s>Assistant:(.*?)</s>'
b=re.findall(pattern,text,re.DOTALL)
num=0
while True:
    try:
        i=random.randint(0,len(a))
        ask=a[i]
        answer=b[i]
        train(ask,answer)
        generation(ask)
        num+=1
        if num%200==0:
            torch.save(obj=model,f="model.pth")
        else:
            continue
    except:
        continue

