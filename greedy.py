import torch
from sampler import Sampler

class GreedySampler(Sampler):
    def __call__(self, prompt, max_new_tokens=10):
        predictions = []
        result = prompt
        # generate until max_len
        for i in range(max_new_tokens):
            input_ids = self.encode(result)
            next_token_probs = self.get_next_token_prob(input_ids=input_ids)
            
            # choose the token with the highest probability
            id = torch.argmax(next_token_probs, dim=-1).item()
            # convert to token and add new token to text
            result += self.decode(id)
            predictions.append(next_token_probs[id].item())

        return result
