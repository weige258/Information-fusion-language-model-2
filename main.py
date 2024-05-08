import torch
from Model import *

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

def encode(text):
    tensor=torch.tensor([],dtype=torch.long)
    for letter in text:
        try:
            tensor=torch.cat((tensor,torch.tensor([ord(letter)])))
        except:
            continue
    return tensor

def probability(letter_tensor):
    tensor=torch.zeros(dict_size)
    try:
        tensor[letter_tensor]=1
    except:
        pass
    return tensor

try:
    model=torch.load(f="model.pth",map_location=device).to(device)
    print("载入模型")
except:
    model=MainModel().to(device)
    print("新建模型")
loss_func=torch.nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=3e-5)

def train(ask,answer):
    input=encode(ask).to(device)
    target=input[-1].to(device)
    for next in encode(answer):
        output=model(input,target.unsqueeze(0))
        label=probability(next).to(device)
        loss=loss_func(output,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        target=next.to(device)

def generation(text):
    output_text=""
    prompt = encode(text).to(device)
    target=prompt[-1].to(device)
    for i in range(max_length):
        try:
            output=model(prompt,target.unsqueeze(0))
            index=int(torch.argmax(output))
            letter=chr(index).encode("utf-8").decode("utf-8")
            output_text+=letter
            target=torch.tensor(index).to(device)
        except:
            continue
    print(output_text)
    return output_text
