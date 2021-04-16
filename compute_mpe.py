import pysdd
from pysdd.sdd import Vtree, SddManager, WmcManager
from pysdd.iterator import SddIterator
from array import array
from graphviz import Source
import torch
from load_config import device

class CircuitMPE:	

	def __init__ (self, vtree_file, sdd_file):

		self.vtree = Vtree.from_file(vtree_file)
		self.sdd_mgr = SddManager.from_vtree(self.vtree)
		self.et_logical_formula = self.sdd_mgr.read_sdd_file(sdd_file)
		self.length = self.sdd_mgr.var_count()
		self.wmc_mgr = self.et_logical_formula.wmc(log_mode = False)	
		self.lits = [None] + [self.sdd_mgr.literal(i) for i in range(1, self.sdd_mgr.var_count() + 1)]		
		 			


	def compute_wmc(self,norm_y_hat):
		distribution_similarity = torch.empty(norm_y_hat.size()).to(device)	
		for index, every in enumerate(norm_y_hat):			
			weights = torch.cat((every, every))
			for i, p in enumerate(every):	
				weights[i] = 1.0 - every[len(every)-1-i]
			weights = array('d',weights)			
			wmc = self.wmc_mgr.propagate()
			# wmc = torch.FloatTensor(wmc).to(device)

			# Consitioned every literal to get marginal weighted model couting
			conditioned_wmc = []							
			for i in range(0, self.length):	
				self.wmc_mgr.set_literal_weight(self.lits[i+1], every[i])
				# self.wmc_mgr.set_literal_weight(-self.lits[i+1], 1- every[i])
				every_conditioned_wmc = self.wmc_mgr.propagate()	#number of models where lits[i+1] being true
				conditioned_wmc.append(every_conditioned_wmc)			
			conditioned_wmc = torch.FloatTensor(conditioned_wmc).to(device)
			conditioned_pr = torch.div(conditioned_wmc, wmc)
			log_cpr = torch.log(conditioned_pr)
			distribution_similarity[index] = torch.mul(norm_y_hat[index], log_cpr)
			torch.clamp(distribution_similarity[index], min=0.0, max=1.0)
		semantic_loss =  torch.mean(distribution_similarity)

		return    semantic_loss



