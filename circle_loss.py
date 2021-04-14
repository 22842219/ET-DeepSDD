from typing import Tuple

import torch
from torch import nn, Tensor


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss



class Model(torch.nn.Module):

  def __init__(self,
      # type_embeddings: torch.nn.Embedding,
      input_dim: int,
      type_dim: int,
      # bottleneck_dim: int,
      dropout_rate: float
      ):
      super(Model, self).__init__()
      # self.type_embeddings = type_embeddings
      self.ffnn = torch.nn.Sequential(
      torch.nn.Dropout(dropout_rate),
      torch.nn.Linear(input_dim, input_dim // 2),
      torch.nn.Tanh(),
      torch.nn.Linear(input_dim // 2, input_dim // 2),
      torch.nn.Tanh(),
      torch.nn.Linear(input_dim // 2, type_dim),
      torch.nn.Tanh()
      )
      # self.linear = torch.nn.Linear(
      #     in_features=type_embeddings.embedding_dim,
      #     out_features=type_embeddings.num_embeddings,
      #     bias=True
      # )
      # self.linear.weight = type_embeddings.weight  # Put the embeddings into the last layer
      # self.bottleneck_dim = bottleneck_dim

      # if self.bottleneck_dim > 0:
      #     self.bottleneck_weight = torch.nn.Parameter(torch.tensor(0.1))
      #     self.bottleneck = torch.nn.Sequential(
      #         torch.nn.Linear(type_embeddings.embedding_dim, bottleneck_dim),
      #         torch.nn.Linear(bottleneck_dim, type_embeddings.num_embeddings)
      #     )

  def forward(self,
      features: torch.Tensor
      ) -> torch.Tensor:

      mapped_mentions = self.ffnn(features)  # R[Batch, Emb]
      # scores = self.linear(mapped_mentions)  # R[Batch, Type]

      # if self.bottleneck_dim > 0:
      #     bottleneck_scores = self.bottleneck(mapped_mentions)  # R[Batch, Type]
      #     scores = scores + self.bottleneck_weight * bottleneck_scores

      return nn.functional.normalize(mapped_mentions)


class Classifier(nn.Module):
  def __init__(self,
      type_embeddings: torch.nn.Embedding):
      super(Classifier, self).__init__()
      self.type_embeddings = type_embeddings
      self.linear = torch.nn.Linear(
      in_features=type_embeddings.embedding_dim,
      out_features=type_embeddings.num_embeddings,
      bias=True
      )
  self.linear.weight = type_embeddings.weight  # Put the embeddings into the last layer

  def forward(self, inp: Tensor) -> Tensor:
      mapped_mentions = self.ffnn(features)
      scores = self.linear(mapped_mentions)  # R[Batch, Type]

      return scores