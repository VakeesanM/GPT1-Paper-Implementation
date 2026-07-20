import torch
import torch.nn as nn
from components.transformer import Transformer
from transformers import AutoTokenizer

CONTEXT_LEN = 128 # How many tokens the Model can look at once
EMBED_DIM = 300 # The Dimension of Vector Embeddings. GPT1 Actually used 768, but for performance reasons, I have decided to reduce it.
ATTN_HEAD = 12 # The Amount of Attention Heads at Mask Multi Head and Normal Multi Head Attention in each decoder.
DECODER_BLOCKS = 12 # The amount of Decoder Blocks in the Model.
NUM_EXPERTS = 4 # Number of Experts in each Decoder Block
device = "cuda" if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("gpt2")
softmax = nn.Softmax()

model = Transformer(embed_size=EMBED_DIM, heads=ATTN_HEAD, blocks=DECODER_BLOCKS, num_experts=NUM_EXPERTS, context_len=CONTEXT_LEN)
weights = torch.load("model_weights.pt", weights_only=True, map_location=torch.device(device))
model.load_state_dict(weights)

model.eval()

def generater_helper(use_cache: bool, tokens:str, temp=0.7, top_k=40):
    if len(tokens) > 128:
        tokens = tokens[-CONTEXT_LEN:]
        tensor = torch.tensor(tokens).unsqueeze(0)
        with torch.inference_mode():
            logits = (model(tensor.to(device), use_cache)[:, -1])/temp
            values, indexs = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, indexs, values)
            probs = softmax(logits)
            output = torch.multinomial(probs, num_samples=1).item()
            return output
            
def yield_untokenize(new_token):
    yield tokenizer.decode(new_token)

   

def generate(text:str, temp=0.7, max_len=20, top_k=40):
    yield text
    tokens = tokenizer(text)['input_ids']
    new_token = generater_helper(tokens,temp, top_k, use_cahce=False)
    yield_untokenize(new_token)
    for i in range(max_len-1):
        new_token = generater_helper(tokens,temp, top_k, use_cahce=True)
        yield_untokenize(new_token)

