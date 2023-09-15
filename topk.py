import math
import torch
from torch.nn.functional import softmax

from sampler import Sampler

class TOPKsampler(Sampler):


    def __call__(self, prompt, max_new_tokens=10 ,top_k = 1 ,temp : float = 0.5):
        
        predictions = []
        result = prompt
        # generate until max_len
        for i in range(max_new_tokens):
            # convert words to tokens
            input_ids = self.encode(result)
            next_token_probs = self.get_next_token_prob(input_ids=input_ids)

            next_token_probs = next_token_probs / temp

            indices_to_remove = next_token_probs < torch.topk(next_token_probs, top_k)[0][..., -1, None]
            new_logits = torch.clone(next_token_probs)
            new_logits[indices_to_remove] = float('-inf')


            # convert logits to scores
            scores = softmax(new_logits, dim=-1)  # Use modified logits
            # sample from scores
            id = torch.multinomial(scores, num_samples=1).item()
            # convert to token and add new token to text
            result += self.decode(id)
            # keep track of scores for next token
            predictions.append(scores[id].item())

        return result

    def sample_plot(self,prompt ,top_k = 5 ,temp : float = 0.5):

        input_ids = self.encode(prompt)
        next_token_probs = self.get_next_token_prob(input_ids=input_ids)

        next_token_probs = next_token_probs / temp

        
        # Remove all tokens with a probability less than the last token of the top-k.
        indices_to_remove = next_token_probs < torch.topk(next_token_probs, top_k)[0][..., -1, None]
        new_logits = torch.clone(next_token_probs)
        new_logits[indices_to_remove] = float('-inf')

        # convert logits to scores
        scores = softmax(new_logits, dim=-1)  # Use modified logits

        self.plot_scores(scores,title=f"Tempreature : {temp}  Top k : {top_k}" , samples = top_k + int(math.sqrt(top_k))) # to visualize if all only top_k have probability distribution


