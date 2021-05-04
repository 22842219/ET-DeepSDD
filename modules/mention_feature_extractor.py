import os
import torch
import torch.nn as nn

class MentionFeatureExtractor(torch.nn.Module):

    def __init__(self,
                 hierarchy: Hierarchy,
                 dim: int,
                 # dropout_rate: float,
                 with_context: bool = False,
                 context_window: int,
				 mention_window: int,
                 attention_type: str = "none"  # dynamic / scalar / none                 
                 ):
     	super(MentionFeatureExtractor, self).__init__()

     	self.dim = dim
     	self.dropout = nn.Dropout(p=0.5)
     	self.dropout_l = nn.Dropout(p=0.5)
		self.dropout_r = nn.Dropout(p=0.5)
		self.dropout_m = nn.Dropout(p=0.5)

		self.with_context = with_context
		self.context_window = context_window
		self.mention_window = mention_window

		self.attention_type = attention_type

		if self.attention_type == "dynamic":
			self.attention_layer = nn.Linear(embedding_dim, 3)
		elif self.attention_type == "scalar":
			self.component_weights = nn.Parameter(torch.ones(3).float())

	def forwar(self,
		       batch_xl: torch.Tensor,
		       batch_xr: torch.Tensor,
		       batch_xa: torch.Tensor,
		       batch_xm: torch.Tensor
			   ):
		if self.attention_type == "dynamic":		
			# If using 'dynamic attention', multiply the concatenated weights of each component by each attention weight.
			# The attention weights should correspond to the weights from batch_xm, the mention context.
			# The idea is that the network will learn to determine the effectiveness of the left, right, or mention context depending
			# on the mention context (i.e. "Apple" requires the left and right context to predict accurately, whereas "Obama" only requires
			# the mention context.		

			attn_weights = torch.softmax(self.attention_layer(batch_xm), dim=1)
			joined = torch.cat((batch_xl, batch_xr, batch_xm), dim=1).view(batch_xm.size()[0], 3, batch_xm.size()[1])
			joined = torch.einsum("ijk,ij->ijk", (joined, attn_weights))
			joined = joined.view(batch_xm.size()[0], batch_xm.size()[1] * 3)
		elif self.attention_type == "scalar":
			component_weights = torch.softmax(self.component_weights, dim=0)
			joined = torch.cat((batch_xl * component_weights[0], batch_xr * component_weights[1],  batch_xm * component_weights[2]), 1)
		elif self.attention_type  == "none":
			joined = torch.cat((batch_xl, batch_xr,  batch_xm), 1)
