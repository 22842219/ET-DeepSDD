import os, re, math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import numpy as np
from array import array


import pysdd
from pysdd.sdd import Vtree, SddManager, WmcManager
from graphviz import Source

from load_config import load_config, device
from pathlib import Path
here = Path(__file__).parent


from compute_mpe import CircuitMPE

torch.manual_seed(123)
torch.backends.cudnn.deterministic=True
from typing import *
import torch
import torch.nn.functional as F
from bert_encoder import  get_contextualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Pooler(torch.nn.Module):

    def __init__(self,
                 dim: int = 768,
                 dropout_rate: float = 0.5 ,
                 pooling: str = "mean",  # max / mean / attention
                 device: str = "cuda:0"
                 ):
        super(Pooler, self).__init__()
        self.device = device
        self.pooling = pooling
        self.dim = dim
        # self.dropout = torch.nn.Dropout(dropout_rate)
        # self.projection = torch.nn.Linear(self.dim, self.dim)


        if self.pooling == "attention":
            self.query = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32)) 
            torch.nn.init.normal_(self.query, mean=0.0, std=0.02)
   
    def forward(self,
                span: torch.Tensor,  # R[Batch, Word, Emb]
                span_mask: torch.Tensor, # R[Batch, word_ids]
                ) -> torch.Tensor:  # R[Batch, Feature]

        batch_size = span.size(0)
        sentence_len = span.size(1)
        emb_size = span.size(2)
        span_len = span.size(1)
        neg_inf = torch.tensor(-10000, dtype=torch.float32, device=device)
        zero = torch.tensor(0, dtype=torch.float32, device=device)

        # span = self.projection(self.dropout(span))

        def attention_pool():
            span_attn_scores = torch.einsum('e,bwe->bw', self.query, span)
            masked_span_attn_scores = torch.where(span_mask.type(torch.ByteTensor).to(device), span_attn_scores, neg_inf)
            normalized_span_attn_scores = F.softmax(masked_span_attn_scores, dim=1)
            span_pooled = torch.einsum('bwe,bw->be', span, normalized_span_attn_scores)
            return span_pooled

        span_pooled = {
            "max": lambda: torch.max(torch.where(span_mask.unsqueeze(dim=2).expand_as(span).type(torch.ByteTensor).to(device), span, neg_inf), dim=1)[0],
            "mean": lambda:  torch.sum(
                torch.where(span_mask.unsqueeze(dim=2).expand_as(span).type(torch.ByteTensor).to(device), span, zero), dim=1) / torch.sum(span_mask,dim =1).unsqueeze(dim=1).expand(batch_size, emb_size),
            "attention": lambda: attention_pool()
        }[self.pooling]()  # R[Batch, Emb]
       
        return span_pooled  # R[Batch, Emb]


class MentionLevelModel(nn.Module):
	def __init__(self, 
				dataset, 
				embedding_dim, 
				hidden_dim, 
				label_size, 
				model_options, 
				total_wordpieces, 
				category_counts, 
				hierarchy_matrix, 
				context_window, 
				mention_window, 
				attention_type, 
				use_context_encoders):
		super(MentionLevelModel, self).__init__()

		self.dataset = dataset
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.label_size = label_size
		self.use_hierarchy 	 = model_options['use_hierarchy']
		self.use_bilstm 	 = model_options['use_bilstm']
		self.hierarchy_matrix = hierarchy_matrix
		self.context_window = context_window
		self.mention_window = mention_window

		self.dropout = nn.Dropout(p=0.5)
		self.bc = get_contextualizer("bert-base-cased", device='cuda:0') # R[Batch, Words, Emb]
		self.embed_pooled = Pooler(dim =self.embedding_dim, dropout_rate=0.5) # R[Batch, Emb]

		if self.use_bilstm:
			self.lstm = nn.LSTM(hidden_dim,hidden_dim,1,bidirectional=True)
			self.layer_1 = nn.Linear(hidden_dim*6, hidden_dim)
		else:
			self.layer_1 = nn.Linear(hidden_dim, hidden_dim)		
		
		self.use_context_encoders = use_context_encoders
		self.projection = nn.Linear(embedding_dim, hidden_dim)	

		self.attention_type = attention_type
		if self.attention_type == "dynamic":
			print("Using dynamic attention")
			self.attention_layer = nn.Linear(embedding_dim, 3)
		elif self.attention_type == "scalar":
			self.component_weights = nn.Parameter(torch.ones(3).float())

		self.hidden2tag = nn.Linear(hidden_dim, label_size)
	
	def forward(self, batch_xl, batch_xr, batch_xa, batch_xm):

		# Convert the batch_x from wordpiece ids into bert embedding vectors		
		bert_embs_l, span_mask_l = self.bc.encode(batch_xl, frozen=True)  	# R[Batch, Word, Emb], R[Batch, Word_Mask]		
		bert_embs_r, span_mask_r  = self.bc.encode(batch_xr, frozen=True)		
		bert_embs_m, span_mask_m  = self.bc.encode(batch_xm, frozen=True)
		bert_embs_a, span_mask_a  = self.bc.encode(batch_xa, frozen=True)

		#Pooling 
		# batch_xl = self.embed_pooled(bert_embs_l, span_mask_l)  # R[Batch, Emb]
		# batch_xr = self.embed_pooled(bert_embs_r, span_mask_r)  # R[Batch, Emb]
		# batch_xm = self.embed_pooled(bert_embs_m, span_mask_m)  # R[Batch, Emb]	
		batch_xa = self.embed_pooled(bert_embs_a, span_mask_a)  # R[Batch, Emb]

		if self.use_bilstm:		
			batch_xl = batch_xl.unsqueeze(0)
			batch_xr = batch_xr.unsqueeze(0)
			batch_xm = batch_xm.unsqueeze(0)
			batch_xl, _ = self.lstm(batch_xl)
			batch_xr, _ = self.lstm(batch_xr)
			batch_xm, _ = self.lstm(batch_xm)
			batch_xl = batch_xl.squeeze(0)
			batch_xr = batch_xr.squeeze(0)
			batch_xm = batch_xm.squeeze(0)

		if self.use_context_encoders:	
			batch_xl = self.dropout(torch.relu(self.projection(batch_xl)))
			batch_xr = self.dropout(torch.relu(self.projection(batch_xr)))
			batch_xm = self.dropout(torch.relu(self.projection(batch_xm)))

		if self.attention_type == "dynamic":		
			attn_weights = torch.softmax(self.attention_layer(batch_xm), dim=1)
			joined = torch.cat((batch_xl, batch_xr, batch_xm), dim=1).view(batch_xm.size()[0], 3, batch_xm.size()[1])
			joined = torch.einsum("ijk,ij->ijk", (joined, attn_weights))
			joined = joined.view(batch_xm.size()[0], batch_xm.size()[1] * 3)
		elif self.attention_type == "scalar":
			component_weights = torch.softmax(self.component_weights, dim=0)
			joined = torch.cat((batch_xl * component_weights[0], batch_xr * component_weights[1],  batch_xm * component_weights[2]), 1)
		elif self.attention_type == "none":
			joined = torch.cat((batch_xl, batch_xr,  batch_xm), 1)
					
		batch_x_out = self.dropout(torch.relu(self.layer_1(batch_xa)))	
		y_hat = self.hidden2tag(batch_x_out)
	
		return y_hat

	def calculate_loss(self, y_hat, batch_y):
		cross_entropy = nn.BCEWithLogitsLoss()
		loss = cross_entropy(y_hat, batch_y)

		if self.use_hierarchy:
			# Create CircuitMPE for predictions
			cmpe = CircuitMPE(bytes(here/"sdd_input"/self.dataset/"et.vtree"), bytes(here/"sdd_input"/self.dataset/"et.sdd"))
			norm_y_hat = torch.sigmoid(y_hat)
			semantic_loss = cmpe.compute_wmc(norm_y_hat)
			loss = loss - 0.05 * semantic_loss

		return loss


	# Predict the labels of a batch of wordpieces using a threshold of 0.5.
	def predict_labels(self, preds):	
		hits  = (preds > 0.0).float()
		autohits = 0
		
		nonzeros = set(torch.index_select(hits.nonzero(), dim=1, index=torch.tensor([0]).to(device)).unique().tolist())
		#print hits.nonzero()
		#print "---"
		#print len(nonzeros), len(hits)
		# If any prediction rows are entirely zero, select the class with the highest probability instead.
		if len(nonzeros) != len(hits):		
			am = preds.max(1)[1]
			for i, col_id in enumerate(am):
				if i not in nonzeros:
					hits[i, col_id] = 1.0
					autohits += 1

		#print "Model predicted %d labels using argmax." % autohits
		return hits

	# Evaluate a given batch_x, predicting the labels.
	def evaluate(self, batch_xl, batch_xr, batch_xa, batch_xm):
		preds = self.forward(batch_xl, batch_xr, batch_xa, batch_xm)
		return self.predict_labels(preds)


	

