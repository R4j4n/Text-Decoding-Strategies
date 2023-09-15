
import math
import torch
from torch.nn.functional import softmax

from sampler import Sampler


class Beam:
    def __init__(self, device, size, input_ids, score, output=None):
        self.device = device
        self.size = size # num_beam 
        self.input_ids = input_ids.to(self.device)
        self.score = score
        self.output = output.to(self.device) if output is not None else None
        
    # get input_ids 
    def get_current_state(self):
        return self.input_ids
    

    # get probability of the sentence         
    def get_score(self):
        return self.score
    
    # create a new instance of Beam class after the top k selection
    def extend(self, token_id, score):
        new_input_ids = torch.cat([self.input_ids, token_id.unsqueeze(0)], dim=-1)
        new_score = self.score * score
        new_output = torch.cat([self.output, token_id.unsqueeze(0)], dim=-1) if self.output is not None else new_input_ids
        return Beam(self.device, self.size, new_input_ids, new_score, new_output)


class BeamSampler(Sampler):

    def beam_decode(self, ids):
        return self.tokenizer.decode(ids.squeeze().tolist())

    # Get the top k id with the greatest probability
    @staticmethod
    def get_topk(prob, k=1):
        scores, token_ids = torch.topk(prob, k=k, dim=-1)
        return scores, token_ids

    def __call__(self, prompt, max_new_tokens=10, num_beam=1):
        input_ids = self.encode(prompt)

        # initialize the beam
        # Ensure this initializes only `num_beam` beams
        beams = [Beam(self.device, num_beam, input_ids, 1) for _ in range(num_beam)]

        # loop until the maximum length is reached
        for i in range(max_new_tokens):
            all_next_token_prob = []
            for beam in beams:
                next_token_prob = self.get_next_token_prob(input_ids=beam.get_current_state())
                all_next_token_prob.append(next_token_prob)
                
            # With this
            all_topk_scores = []
            all_topk_token_ids = []
            for prob in all_next_token_prob:
                scores, token_ids = self.get_topk(prob, k=num_beam)
                all_topk_scores.append(scores)
                all_topk_token_ids.append(token_ids)

            all_topk_scores = torch.stack(all_topk_scores)
            all_topk_token_ids = torch.stack(all_topk_token_ids)

            new_beams = []
            # Then, when accessing them:
            for j, beam in enumerate(beams):
                for k in range(num_beam):
                    score = all_topk_scores[j][k].item()
                    token_id = all_topk_token_ids[j][k].unsqueeze(0)
                    new_beam = beam.extend(token_id, score)
                    new_beams.append(new_beam)

            beams = sorted(new_beams, key=lambda b: b.get_score(), reverse=True)[:num_beam]
        generated_text = self.beam_decode(beams[0].output[:, len(input_ids[0]):])

        return prompt + generated_text


