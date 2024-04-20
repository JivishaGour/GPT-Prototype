import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

device = 'cuda' if torch.cuda.is_available() else 'cpu' # all the calculations happen on GPU and get a lot faster

# reading data
with open('Dataset.txt', 'r', encoding='utf-8') as f:
    data = f.read()
#print length of data
#print("Length of dataset in characters: ", len(data))

#printing all the unique characters that occur in the dataset
chars = sorted(list(set(data)))
vocab_size = len(chars)
#print(''.join(chars))
#print(" All the unique characters that occur in the data are: " , vocab_size)

# create the decoder and the encoder
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #encoder take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder takes a list of integers, output a string 

# now encoding the entire dataset and storing it into a torch.Tensor
import torch # we use pyTorch:
data = torch.tensor(encode(data), dtype=torch.long)
data = data.to(device)
#print(data.shape, data.dtype)


# splitting the data into train and test sets
n = int(0.8*len(data)) # first 80% will be train, rest test
train_data = data[:n]
test_data = data[n:]

train_data = train_data.to(device)
test_data = test_data.to(device)
# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
eval_iters = 200
eval_interval = 500
max_iters = 5000
lr = 5e-5
dmodel = 684
n_head = 6
n_layer = 6
dropout = 0.2


# data loading:
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data #this gives the data array
    ix = torch.randint(len(data) - block_size, (batch_size,)) # these are going to be random 4 positions to get a chunk out of them the data, b/c batch size = 4, so they will be between 0 and len(data) - block size
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
xb, yb = get_batch('train')
xb, yb = xb.to(device), yb.to(device)




#print('inputs:')
#print(xb.shape)
#print(xb)
#print('targets:')
#print(yb.shape)
#print(yb)

#print('----')

 # for b in range(batch_size): # batch dimension
 #   for t in range(block_size): # time dimension
 #       context = xb[b, :t+1]
 #       target = yb[b,t]
 #      print(f"when input is {context.tolist()} the target: {target}")





#self-attention!---------------------------------------------------------------------------------------------------------------------------------------------------------
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf')) # upper triangular elements become infinity
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
#out = wei @ x
wei = wei.to(device)
out.shape


class Head(nn.Module):
    """ one head of self-attention """

    def _init_(self, head_size):
        super()._init_()
        self.key = nn.Linear(dmodel, head_size, bias=False)
        self.query = nn.Linear(dmodel, head_size, bias=False)
        self.value = nn.Linear(dmodel, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
k = k.to(device)
q = q.to(device)
wei = wei.to(device)


k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
wei = q @ k.transpose(-2, -1) * head_size**-0.5 # denominator, where we divide with sq root (dk)


# Multihead attention----------------------------------------------------------------------------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def _init_(self, num_heads, head_size):
        super()._init_()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, dmodel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
# Feed forward neural network--------------------------------------------------------------------------------------------------------------------------------------------
class FeedFoward(nn.Module): # nn.modeule is the base class for all pytorch neural network modules 
    def _init_(self, dmodel):
        super()._init_()
        self.net = nn.Sequential(   #self.net: This is where the feedforward neural network is defined using an nn.Sequential container, which is a way to chain multiple layers sequentially.
            nn.Linear(dmodel, 4 * dmodel),  # input features = dmodel , output features = 4*dmodel
            nn.ReLU(),
            nn.Linear(4 * dmodel, dmodel),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Block------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def _init_(self, dmodel, n_head):
        # dmodel: embedding dimension, n_head: the number of heads we'd like
        super()._init_()
        head_size = dmodel // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(dmodel)
        self.ln1 = nn.LayerNorm(dmodel)
        self.ln2 = nn.LayerNorm(dmodel)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
# Layer normalization----------------------------------------------------------------------------------------------------------------------------------------------------
    class LayerNorm1d:
        
        def _init_(self, dim, eps=1e-5, momentum=0.1):
            self.eps = eps
            self.gamma = torch.ones(dim)
            self.beta = torch.zeros(dim)

        def _call_(self, x):
            # calculate the forward pass
            xmean = x.mean(1, keepdim=True) # batch mean
            xvar = x.var(1, keepdim=True) # batch variance
            xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
            self.out = self.gamma * xhat + self.beta
            return self.out

        def parameters(self):
            return [self.gamma, self.beta]

    torch.manual_seed(1337)
    module = LayerNorm1d(100)
    x = torch.randn(32, 100) # batch size 32 of 100-dimensional vectors
    x = module(x)
    x.shape

# model------------------------------------------------------------------------------------------------------------------------------------------------------------------

class GPTmodel(nn.Module):

    def _init_(self):
        super()._init_()
        
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, dmodel)
        self.position_embedding_table  = nn.Embedding(block_size, dmodel)
        self.blocks = nn.Sequential(*[Block(dmodel, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(dmodel) # final layer norm
        self.lm_head = nn.Linear(dmodel, vocab_size) # language modelling head


    def forward(self, idx, targets=None):
        B, T = idx.shape
        idx = idx.to(device)
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) batch time channel(vocab_size = 65)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T, C)
        x = tok_emb + pos_emb # (B, T, c)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T, vocab_size)
        # logits/predictions = score for next character in the sequence/ we are predicting what comes next based on individual identity of a single token
        tok_emb = tok_emb.to(device)
        pos_emb = pos_emb.to(device)
        if targets is None:
            loss = None # if target is none then there is no loss to evaluate
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 2D array
            targets = targets.view(B*T)
            targets = targets.to(device)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx tot he last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1).to(device) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1).to(device) # (B, T+1)
            
        return idx
     # function to average up the loss over multiple batches:

    

model = GPTmodel().to(device)
model = model.to(device)
m = model.to(device) # we have to move the model pareameter to device because they are going to run on the GPU of the device
logits, loss = m(xb, yb)
print(logits.shape)
#print(loss)
logits = logits.to(device)
loss = loss.to(device)

#print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))
# loss is 5.1094

@torch.no_grad()
def estimate_loss():
    out = {}
    model = GPTmodel().to(device)
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y) # here we got our loss
            losses[k] = loss.item()
            out[split] = losses.mean() # here we got the average of our losses
    model.train()
    model = model.to(device)
    return out

# Optimizers are used for setting learning rate and weights in machine learning
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = 3e-4) #Adamw is an optimizer just like stochastic gradient descent
# lr = learning rate

# printing steps
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and test sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")


        # sample a batch of data 
        xb, yb = get_batch('train')

        #evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none= True)
        loss.backward()
        optimizer.step()



# generate from the model
context = torch.zeros((1,1), dtype = torch.long, device=device)
context = context.to(device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
