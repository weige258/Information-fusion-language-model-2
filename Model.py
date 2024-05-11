import torch

def layer_norm(tensor):
    device=tensor.device
    tensor=torch.nn.LayerNorm([tensor.size()[0],tensor.size()[1]],device=device)(tensor)
    return tensor

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class TransformerBlock(torch.nn.Module):
    def __init__(self,emb_size,num_heads,drop_out=0.1):
        super().__init__()
        self.rms_norm=RMSNorm(emb_size)
        self.attention=torch.nn.MultiheadAttention(emb_size,num_heads,drop_out)
        self.linear1 = torch.nn.Linear(emb_size,4*emb_size)
        self.linear2 = torch.nn.Linear(emb_size, 4 * emb_size)
        self.silu=torch.nn.SiLU()
        self.drop_out=torch.nn.Dropout(drop_out)
        self.linear3 = torch.nn.Linear(4*emb_size,emb_size)
    def forward(self,tensor):
        copy_tensor=tensor
        tensor=self.rms_norm(tensor)
        tensor,_=self.attention(tensor,tensor,tensor)
        tensor+=copy_tensor
        copy_tensor=tensor
        tensor = self.rms_norm(tensor)
        tensor1=self.linear1(tensor)
        tensor1=self.drop_out(tensor1)
        tensor2=self.linear2(tensor)
        tensor2=self.silu(tensor2)
        tensor2=self.drop_out(tensor2)
        tensor=tensor1*tensor2
        tensor=self.linear3(tensor)
        tensor=self.drop_out(tensor)
        tensor+=copy_tensor
        return tensor

emb_size=256
heads=32
dict_size=60000
max_length=100
temperature=0.3

class MainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=torch.nn.Embedding(num_embeddings=1114112,embedding_dim=emb_size)
        self.font_block1=TransformerBlock(emb_size,heads)
        self.font_block2 = TransformerBlock(emb_size, heads)
        self.font_block3 = TransformerBlock(emb_size, heads)
        self.font_block4 = TransformerBlock(emb_size, heads)
        self.back_block1 = TransformerBlock(emb_size, heads)
        self.back_block2 = TransformerBlock(emb_size, heads)
        self.back_block3 = TransformerBlock(emb_size, heads)
        self.back_block4 = TransformerBlock(emb_size, heads)
        self.back_block5 = TransformerBlock(emb_size, heads)
        self.back_block6 = TransformerBlock(emb_size, heads)
        self.back_block7 = TransformerBlock(emb_size, heads)
        self.back_block8 = TransformerBlock(emb_size, heads)
        self.output_layer=torch.nn.Linear(emb_size,dict_size)
    def forward(self,prompt,autoregressive):
        prompt=self.emb(prompt)
        prompt=layer_norm(prompt)
        autoregressive=self.emb(autoregressive)
        autoregressive=self.font_block1(autoregressive)
        autoregressive = self.font_block2(autoregressive)
        autoregressive = self.font_block3(autoregressive)
        autoregressive = self.font_block4(autoregressive)
        autoregressive=(autoregressive*prompt).sum(dim=0).unsqueeze(0)
        autoregressive =self.back_block1(autoregressive)
        autoregressive = self.back_block2(autoregressive)
        autoregressive = self.back_block3(autoregressive)
        autoregressive = self.back_block4(autoregressive)
        autoregressive = self.back_block5(autoregressive)
        autoregressive = self.back_block6(autoregressive)
        autoregressive = self.back_block7(autoregressive)
        autoregressive = self.back_block8(autoregressive)
        autoregressive=torch.flatten(autoregressive)
        autoregressive=self.output_layer(autoregressive)
        return autoregressive
