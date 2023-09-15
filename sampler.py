import torch
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForCausalLM
# torch.manual_seed(0)

class Sampler:

    def __init__(self , model_name : str ='gpt2-medium') -> None:

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu").to(self.device)

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors='pt').to(self.device)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def get_next_token_prob(self, input_ids: torch.Tensor):
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits
        logits = logits[0, -1, :]
        return logits

    def plot_scores(self, scores, title, samples):
        top_indices = torch.argsort(scores, descending=True)[:samples]
        tokens = [self.decode(idx) for idx in top_indices]

        if self.device == "cpu":
            top_probs = scores[top_indices].numpy()
        else:
            top_probs = scores[top_indices].cpu().numpy()

        
        colors = ['#E95B68', '#C4C956', '#58BB7B', '#CAC1C5', '#87601F', '#F7311B', 
                  '#C53D39', '#38658F', '#242ABC', '#9DA52F', '#329018', '#D415C5', 
                  '#6DCE59', '#ADF212', '#9CF042']
        colors = colors[0:len(top_indices)]

        fig = go.Figure(data=[
            go.Bar(x=tokens, y=top_probs, marker_color=colors, textposition='auto')
        ])
        fig.update_layout(title=title)
        fig.show()