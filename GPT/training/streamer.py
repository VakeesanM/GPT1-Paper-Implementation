import torch
import torch.nn as nn

class Streamer():
    def __init__(self, model, tokenizer):
        super(Streamer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.softmax = nn.Softmax()
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
    
    def generate(self, text: str, temp=0.7, max_len=50, top_k=40):
        yield text
        self.model.reset_cache()
        tokens = self.tokenizer(text)['input_ids']
        new_token = self.helper_func(tokens, False, temp, top_k)

        yield self.tokenizer.decode([new_token])

        for _ in range(max_len):
            new_token = self.helper_func([new_token], True, temp, top_k)
            yield self.tokenizer.decode([new_token])

    def helper_func(self, tokens, use_cache: bool, temp=0.7, top_k=40):
        if len(tokens) > 128:
            tokens = tokens[-128:]
        tensor = torch.tensor(tokens).unsqueeze(0)
        
        with torch.inference_mode():
            logits = (self.model(tensor.to(self.device), use_cache)[:, -1])/temp
            values, indexs = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, indexs, values)
            probs = self.softmax(logits)
            output = torch.multinomial(probs, num_samples=1).item()
            return output



