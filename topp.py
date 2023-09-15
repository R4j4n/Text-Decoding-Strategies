
import math
import torch
from torch.nn.functional import softmax

from sampler import Sampler

class NucleusSampler(Sampler):

    def __call__(self, prompt, max_new_tokens=10 , p : float = 0.7):
        
        predictions = []
        result = prompt
        # generate until max_len
        for i in range(max_new_tokens):
            # convert words to tokens
            input_ids = self.encode(result)

            next_token_probs = self.get_next_token_prob(input_ids=input_ids)


            sorted_logits, sorted_indices = torch.sort(next_token_probs, descending=True)
            cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > p
            
            """
            When we determine which tokens to remove based on this mask, it's important to note that as soon as the cumulative probability crosses the threshold `p`, 
            all the subsequent tokens will also have cumulative probabilities greater than `p` (because the probabilities are sorted in descending order). 
            The logic here is to also exclude the very first token that caused the cumulative sum to cross the threshold, and this is achieved by shifting the mask to the right.
            By doing this shift and ensuring the first token that exceeds the threshold is included in the removal list, 
            we're adhering to the true spirit of top-p sampling: we're including in the final consideration only those tokens whose cumulative sum is less than or equal to `p`.
            """
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]

            new_logits = torch.clone(next_token_probs)
            new_logits[indices_to_remove] = float('-inf')

            # convert logits to scores
            scores = softmax(new_logits, dim=-1)  # Use modified logits
            # sample from scores
            id = torch.multinomial(scores, num_samples=1).item()\
            
            # convert to token and add new token to text
            result += self.decode(id)
            
            # keep track of scores for next token
            predictions.append(scores[id].item())

        return result
    


    def sample_plot(self,prompt, p: float):

        input_ids = self.encode(prompt)

        next_token_probs = self.get_next_token_prob(input_ids=input_ids)

        sorted_logits, sorted_indices = torch.sort(next_token_probs, descending=True)
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                
        new_logits = torch.clone(next_token_probs)
        new_logits[indices_to_remove] = float('-inf')

        # convert logits to scores
        scores = softmax(new_logits, dim=-1)  

        self.plot_scores(scores,title=f"P : {p}", samples=10) 