import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from load_config import device
import pandas as pd
import pysdd
from pysdd.sdd import Vtree, SddManager, WmcManager

from pysdd.iterator import SddIterator
from array import array
from graphviz import Source
import math
from load_config import load_config, device
import numpy as np
import math
from pathlib import Path
here = Path(__file__).parent

import re



torch.manual_seed(123)
torch.backends.cudnn.deterministic=True


class MarginLoss(nn.Module):
	def __init__(self, m_pos, m_neg, lambda_):
		super(MarginLoss, self).__init__()
		self.m_pos = m_pos
		self.m_neg = m_neg
		self.lambda_ = lambda_

	def forward(self, y_hat, batch_y, size_average=True):
		t = torch.ones(y_hat.size())
		if batch_y.is_cuda:
			t = t.cuda()
		# t = t.scatter_(1, batch_y.data.view(-1, 1), 1)
		# batch_y = Variable(t)
		losses = batch_y.float() * F.relu(self.m_pos*t - y_hat).pow(2) + \
				self.lambda_ * (1. - batch_y.float()) * F.relu(y_hat - self.m_neg*t).pow(2)
		return losses.mean() if size_average else losses.sum()






class MentionLevelModel(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, dataset, model_options, total_wordpieces, category_counts, hierarchy_matrix, context_window, mention_window, attention_type, use_context_encoders):

		super(MentionLevelModel, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.label_size = label_size

		self.dataset = dataset

		self.use_hierarchy = model_options['use_hierarchy']
		self.use_bilstm = model_options['use_bilstm']


		if self.use_bilstm:
			self.lstm = nn.LSTM(embedding_dim,hidden_dim,1,bidirectional=True)
			self.layer_1 = nn.Linear(hidden_dim*6, hidden_dim)
		else:
			self.layer_1 = nn.Linear(hidden_dim + hidden_dim + hidden_dim, hidden_dim)

		self.hidden2tag = nn.Linear(hidden_dim, label_size)
	

		self.dropout = nn.Dropout(p=0.5)


		self.dropout_l = nn.Dropout(p=0.5)
		self.dropout_r = nn.Dropout(p=0.5)
		self.dropout_m = nn.Dropout(p=0.5)

		self.hierarchy_matrix = hierarchy_matrix
		self.context_window = context_window
		self.mention_window = mention_window


		self.left_enc = nn.Linear(embedding_dim, hidden_dim)
		self.right_enc = nn.Linear(embedding_dim, hidden_dim)
		self.mention_enc = nn.Linear(embedding_dim, hidden_dim)
				

		self.attention_type = attention_type
		self.use_context_encoders = use_context_encoders
		
		if self.attention_type == "dynamic":
			print("Using dynamic attention")
			self.attention_layer = nn.Linear(embedding_dim, 3)
		elif self.attention_type == "scalar":
			self.component_weights = nn.Parameter(torch.ones(3).float())

	
	def forward(self, batch_xl, batch_xr, batch_xa, batch_xm):		

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

			batch_xl = self.dropout_l(torch.relu(self.left_enc(batch_xl)))
			batch_xr = self.dropout_l(torch.relu(self.right_enc(batch_xr)))
			batch_xm = self.dropout_l(torch.relu(self.mention_enc(batch_xm)))


		if self.attention_type == "dynamic":		
			# If using 'dynamic attention', multiply the concatenated weights of each component by each attention weight.
			# The attention weights should correspond to the weights from batch_xm, the mention context.
			# The idea is that the network will learn to determine the effectiveness of the left, right, or mention context depending
			# on the mention context (i.e. "Apple" requires the left and right context to predict accurately, whereas "Obama" only requires
			# the mention context.
			

			attn_weights = torch.softmax(self.attention_layer(batch_xm), dim=1)
			#print attn_weights[0]
			joined = torch.cat((batch_xl, batch_xr, batch_xm), dim=1).view(batch_xm.size()[0], 3, batch_xm.size()[1])
			joined = torch.einsum("ijk,ij->ijk", (joined, attn_weights))
			joined = joined.view(batch_xm.size()[0], batch_xm.size()[1] * 3)
		elif self.attention_type == "scalar":
			component_weights = torch.softmax(self.component_weights, dim=0)
			joined = torch.cat((batch_xl * component_weights[0], batch_xr * component_weights[1],  batch_xm * component_weights[2]), 1)
		elif self.attention_type  == "none":
			joined = torch.cat((batch_xl, batch_xr,  batch_xm), 1)
		
		batch_x_out = self.dropout(torch.relu(self.layer_1(joined)))
		y_hat = self.hidden2tag(batch_x_out)


		return y_hat



	def compute_wmc(self, every_) :	

		vtree = Vtree.from_file(b"sdd_output/et_logical_formula.vtree")
		sdd_mgr = SddManager.from_vtree(vtree)
		et_logical_formula = sdd_mgr.read_sdd_file(b"sdd_output/et_logical_formula.sdd")


		weights = torch.cat((every_, every_))
		for i, p in enumerate(every_):	
			weights[i] = 1.0 - every_[len(every_)-1-i]
		weights = array('d', weights)

		# Consitioned every literal to get marginal weighted model couting
		conditioned_wmc = []		
		wmc = []			
		lits = [None] + [sdd_mgr.literal(i) for i in range(1, sdd_mgr.var_count() + 1)]	
		for i in range(0, sdd_mgr.var_count()):	
			wmc_mgr = et_logical_formula.wmc(log_mode = False) 
			wmc_mgr.set_literal_weights_from_array(weights)						
			wmc.append(wmc_mgr.propagate())
			wmc_mgr.set_literal_weight(lits[i+1], 1)
			wmc_mgr.set_literal_weight(-lits[i+1], 0)
			every_conditioned_wmc = wmc_mgr.propagate()	#number of models where lits[i+1] being true
			conditioned_wmc.append(every_conditioned_wmc)

		return    conditioned_wmc, wmc

	
	def calculate_loss(self, y_hat, batch_y):
		cross_entropy = nn.BCEWithLogitsLoss()
		loss = cross_entropy(y_hat, batch_y)

		if self.use_hierarchy:		
			distribution_similarity = torch.empty(y_hat.size()).to(device)	
			for index, every_ in enumerate(y_hat):
				normalized_logits = torch.sigmoid(every_)
				conditioned_wmc, wmc = self.compute_wmc(normalized_logits)
				wmc = torch.FloatTensor(wmc).to(device)
				conditioned_wmc = torch.FloatTensor(conditioned_wmc).to(device)
				conditioned_pr = torch.div(conditioned_wmc, wmc)
				log_cpr = torch.log(conditioned_pr)
				distribution_similarity[index] = torch.mul(normalized_logits, log_cpr)
				torch.clamp(distribution_similarity[index], min=0.0, max=1.0)
				# with open(here / "outputs" / "y_hat_and_updated", "a") as out:
				# 	print(normalized_logits, file=out)
				# 	print(labeled_examples[index], file=out)
			semantic_loss =  torch.mean(distribution_similarity)
			print("semantic_loss:", semantic_loss)
							
			loss = loss - semantic_loss
			print(loss)
	

		return loss


	# Predict the labels of a batch of wordpieces using a threshold of 0.5.
	def predict_labels(self, preds):    	
		hits  = (preds > 0.5).float()
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
		print(preds)
		return self.predict_labels(preds)

