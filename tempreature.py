import math
import torch
from torch.nn.functional import softmax

from sampler import Sampler

class RandomTempSampler(Sampler):
    def __call__(self, prompt, max_new_tokens=10 , temp : float = 0.5):
    
        predictions = []
        result = prompt
        # generate until max_len
        for i in range(max_new_tokens):
        
            input_ids = self.encode(result)

            next_token_probs = self.get_next_token_prob(input_ids=input_ids)

            # apply temp before softmax (sharper logits)
            next_token_probs /= temp
            # convert logits to scores
            scores = softmax(next_token_probs, dim=-1)
            # sample from scores
            id = torch.multinomial(scores, num_samples=1).item()
            # convert to token and add new token to text
            result += self.decode(id)
            # keep track of scores for next token
            predictions.append(scores[id].item())

        
        return result
    
    def sample_plot(self,prompt,temp:float = 0.5):
        
        input_ids = self.encode(prompt)

        next_token_probs = self.get_next_token_prob(input_ids=input_ids)
        next_token_probs /= temp
        scores = softmax(next_token_probs, dim=0)

        self.plot_scores(scores=scores,title=f"Tempreature : {temp}",samples=10)