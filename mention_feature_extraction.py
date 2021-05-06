from typing import *
import torch
import torch.nn.functional as F


class MentionFeatureExtractor(torch.nn.Module):

    def __init__(self,
                 dim: int = 768,
                 dropout_rate: float = 0.5 ,
                 mention_pooling: str = "attention",  # max / mean / attention
                 device: str = "cuda:0"
                 ):
        super(MentionFeatureExtractor, self).__init__()
        self.device = device
        self.mention_pooling = mention_pooling
        self.dim = dim
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.projection = torch.nn.Linear(self.dim, self.dim)
        torch.nn.init.eye_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.weight)

        if self.mention_pooling == "attention":
            self.query = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            torch.nn.init.normal_(self.query, mean=0.0, std=0.02)


    def forward(self,
                sentence: torch.Tensor,  # R[Batch, Word, Emb]  
                span: torch.Tensor,  # R[Batch, Word, Emb]
                span_lengths: torch.Tensor,  # Z[Batch, Word]   
                ) -> torch.Tensor:  # R[Batch, Feature]

        batch_size = sentence.size(0)
        sentence_max_len = sentence.size(1)
        emb_size = sentence.size(2)
        span_max_len = span.size(1)
        device = sentence.device
        neg_inf = torch.tensor(-10000, dtype=torch.float32, device=device)
        zero = torch.tensor(0, dtype=torch.float32, device=device)

        span = self.projection(self.dropout(span))
        sentence = self.projection(self.dropout(sentence))

        def attention_pool():
            span_attn_scores = torch.einsum('e,bwe->bw', self.query, span)
            masked_span_attn_scores = torch.where(span_lengths.type(torch.ByteTensor).to(self.device), span_attn_scores, neg_inf)
            normalized_span_attn_scores = F.softmax(masked_span_attn_scores, dim=1)
            span_pooled = torch.einsum('bwe,bw->be', span, normalized_span_attn_scores)
            return span_pooled

        span_pooled = {
            "max": lambda: torch.max(torch.where(span_lengths.unsqueeze(dim=2).expand_as(span), span, neg_inf), dim=1)[0],
            "mean": lambda: torch.sum(
                torch.where(span_lengths.unsqueeze(dim=2).expand_as(span).type(torch.ByteTensor).to(self.device), span, zero), dim=1
            ) / span_lengths.unsqueeze(dim=1).expand(batch_size, emb_size),
            "attention": lambda: attention_pool()
        }[self.mention_pooling]()  # R[Batch, Emb]

        features = span_pooled



        return features  # R[Batch, Emb]