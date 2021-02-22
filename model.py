import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from load_config import device
import pandas as pd
import pysdd
from pysdd.sdd import Vtree, SddManager, WmcManager
from array import array
# from graphviz import Source
import math
from load_config import load_config, device
import numpy as np

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
	def __init__(self, embedding_dim, hidden_dim, bottleneck_dim, vocab_size, label_size, dataset, model_options, total_wordpieces, category_counts, hierarchy_matrix, context_window, mention_window, use_bilstm, use_marginal_ranking_loss):
		super(MentionLevelModel, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.label_size = label_size
		self.bottleneck_dim = bottleneck_dim
		self.dataset = dataset
		self.use_bilstm = use_bilstm
		self.use_marginal_ranking_loss = use_marginal_ranking_loss

		if self.use_bilstm:
			self.lstm = nn.LSTM(embedding_dim,hidden_dim,1,bidirectional=True)
			self.layer_1 = nn.Linear(hidden_dim*6, hidden_dim)
		else:
			self.layer_1 = nn.Linear(hidden_dim + hidden_dim + hidden_dim, hidden_dim)

		self.hidden2tag = nn.Linear(hidden_dim, label_size)		
		if self.bottleneck_dim > 0:
			self.bottleneck_weight = torch.nn.Parameter(torch.tensor(0.1))
			self.bottleneck = torch.nn.Sequential(
			torch.nn.Linear(hidden_dim, bottleneck_dim),
			torch.nn.Linear(bottleneck_dim, label_size)
			)

		self.use_hierarchy = model_options['use_hierarchy']

		self.dropout = nn.Dropout(p=0.5)
		self.dropout_l = nn.Dropout(p=0.5)
		self.dropout_r = nn.Dropout(p=0.5)
		self.dropout_m = nn.Dropout(p=0.5)

		self.hierarchy_matrix = hierarchy_matrix
		self.context_window = context_window
		self.mention_window = mention_window


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
	
		joined = torch.cat((batch_xl, batch_xr, batch_xm), 1)	
		batch_x_out = self.dropout(torch.relu(self.layer_1(joined))) # R[Batch, Emb]		
		y_hat = self.hidden2tag(batch_x_out) # R[Batch, Type]
		if self.bottleneck_dim > 0:
			bottleneck_scores = self.bottleneck(batch_x_out)  # R[Batch, Type]
			y_hat = y_hat + self.bottleneck_weight * bottleneck_scores
		if self.use_hierarchy:
			for index, every_ in enumerate(y_hat):
				normalized_logits = torch.sigmoid(every_)
				wmc, likelihood = self.compute_wmc(normalized_logits)		
				wmc = torch.FloatTensor(wmc).to(device)
				likelihood = torch.FloatTensor(likelihood).to(device)
				joint_pro = torch.mul(normalized_logits,likelihood)
				y_hat[index] = torch.div(joint_pro, wmc)
		return  y_hat

	def AtMostOne(self,literals, mgr):
		alpha = mgr.true()
		for lit in literals:
			alpha += ~lit			
		return alpha

	def implication(self,literals, mgr):
		alpha = mgr.true()
		beta0 = literals[0]
		for lit in literals[1:]:
			beta = ~lit | beta0  			
			alpha = alpha | beta
		return alpha

	def exactly_one(self, lits, mgr):
		alpha = mgr.false()
		for lit in lits:
			beta = lit
			for lit2 in lits:
				if lit2!=lit:
					beta = beta & ~lit2
			alpha = alpha | beta
		return alpha

	def compute_wmc(self, every_) :

		vtree = Vtree(self.label_size, vtree_type = 'balanced')
		sdd_mgr = SddManager(vtree = vtree)

		if self.dataset == 'bbn_modified':				
			ANIMAL, CONTACT_INFO, ADDRESS, PHONE, DISEASE, EVENT, HURRICANE, WAR, FAC, AIRPORT, ATTRACTION, BRIDGE, BUILDING, HIGHWAY_STREET, GAME, GPE, CITY, COUNTRY, STATE_PROVINCE, LANGUAGE, LAW, LOCATION, CONTINENT, LAKE_SEA_OCEAN, REGION, RIVER, ORGANIZATION, CORPORATION, EDUCATIONAL, GOVERNMENT, HOSPITAL, HOTEL, POLITICAL, RELIGIOUS, PERSON, PLANT, PRODUCT, VEHICLE, WEAPON, SUBSTANCE, CHEMICAL, DRUG, FOOD, WORK_OF_ART, BOOK, PAINTING, PLAY, SONG = sdd_mgr.vars
			# et_logical_formula = ANIMAL | CONTACT_INFO | (CONTACT_INFO & PHONE)| (CONTACT_INFO & ADDRESS) | DISEASE | EVENT | (EVENT & HURRICANE) | (EVENT & WAR) | (FAC & BUILDING) | FAC | (FAC & AIRPORT) | (FAC & HIGHWAY_STREET) | (FAC & ATTRACTION) | (FAC & BRIDGE) | (GPE & CITY) | GPE | (GPE & STATE_PROVINCE)| (GPE & COUNTRY) | GAME | (LOCATION & REGION)| LOCATION | (LOCATION & CONTINENT) | LAW | LANGUAGE | (LOCATION & LAKE_SEA_OCEAN) | (LOCATION & RIVER) | (ORGANIZATION & CORPORATION)| ORGANIZATION | (ORGANIZATION & GOVERNMENT) | (ORGANIZATION & POLITICAL) | (ORGANIZATION & EDUCATIONAL) | (ORGANIZATION & HOTEL) | (ORGANIZATION & RELIGIOUS) | (ORGANIZATION & HOSPITAL) | PERSON | PRODUCT | (PRODUCT & WEAPON) | (PRODUCT & VEHICLE) | PLANT | SUBSTANCE | (SUBSTANCE & FOOD) | (SUBSTANCE & DRUG) | (SUBSTANCE & CHEMICAL) | WORK_OF_ART | (WORK_OF_ART & BOOK)| (WORK_OF_ART & SONG) | (WORK_OF_ART & PAINTING) | (WORK_OF_ART & PLAY)
			# et_logical_formula = ( ~ANIMAL | ~CONTACT_INFO | ~DISEASE | ~EVENT | ~FAC | ~GAME | ~GPE | ~ LANGUAGE | ~ LAW | ~ LOCATION | ~ ORGANIZATION | ~ PERSON | ~ PLANT | ~ PRODUCT | ~ SUBSTANCE | ~ WORK_OF_ART) & (CONTACT_INFO | ~ ADDRESS  | ~ PHONE) & (EVENT  | ~ HURRICANE  | ~ WAR) & (FAC  | ~ AIRPORT  | ~ ATTRACTION  | ~ BRIDGE  | ~ BUILDING  | ~ HIGHWAY_STREET) & (GPE  | ~ CITY  | ~ COUNTRY  | ~ STATE_PROVINCE) & (LOCATION  | ~ CONTINENT  | ~ LAKE_SEA_OCEAN  | ~ REGION  | ~ RIVER ) & (ORGANIZATION  | ~ CORPORATION  | ~ EDUCATIONAL  | ~ GOVERNMENT  | ~ HOSPITAL  | ~ HOTEL  | ~ POLITICAL  | ~ RELIGIOUS ) & (PRODUCT  | ~ VEHICLE  | ~ WEAPON) & (SUBSTANCE  | ~ CHEMICAL  | ~ DRUG  | ~ FOOD) & (WORK_OF_ART  | ~ BOOK  | ~ PAINTING  | ~ PLAY  | ~ SONG)
			column1 = CONTACT_INFO | (~ADDRESS & ~ PHONE)
			column2 = EVENT | (~HURRICANE  & ~ WAR)
			column3 = FAC | (~AIRPORT  & ~ ATTRACTION  & ~ BRIDGE  & ~ BUILDING  & ~ HIGHWAY_STREET)
			column4 = GPE | (~CITY  & ~ COUNTRY  & ~ STATE_PROVINCE)
			column5 = LOCATION  | (~CONTINENT & ~ LAKE_SEA_OCEAN  & ~ REGION  & ~ RIVER )
			column6 = ORGANIZATION | (~CORPORATION  & ~ EDUCATIONAL  & ~ GOVERNMENT  & ~ HOSPITAL  & ~ HOTEL & ~ POLITICAL & ~ RELIGIOUS )
			column7 = PRODUCT  | (~ VEHICLE  & ~ WEAPON)
			column8 = SUBSTANCE | (~CHEMICAL  & ~ DRUG  & ~ FOOD)
			column9 = WORK_OF_ART | (~ BOOK & ~ PAINTING  & ~ PLAY  & ~ SONG)
			mutually_exclusive = self.exactly_one(ANIMAL,DISEASE,GAME,LANGUAGE,LAW,PERSON,PLANT,CONTACT_INFO, EVENT,FAC,GPE,LOCATION,ORGANIZATION,PRODUCT,SUBSTANCE,WORK_OF_ART)
			et_logical_formula = mutually_exclusive & column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9
		elif self.dataset == 'ontonotes_modified':	
			location,celestial,city,country,geography,body_of_water,park,structure,airport,government,hotel,sports_facility,transit,bridge,railway,road,organization,company,broadcast,news,education,government,military,political_party,sports_team,stock_exchange,other,art,broadcast,film,music,stage,writing,body_part,currency,event,election,holiday,natural_disaster,protest,sports_event,violent_conflict,food,health,treatment,heritage,legal,living_thing,animal,product,car,computer,software,weapon,religion,scientific,sports_and_leisure,person,artist,actor,author,music,athlete,business,doctor,education,student,teacher,legal,military,political_figure,title  = sdd_mgr.vars
			column1 = location | (~celestial & ~ city& ~country & ~geography & ~park & ~structure & ~transit )
			column2 = geography | ~body_of_water 
			column3 = structure | (~airport & ~government & ~hotel & ~sports_facility)
			column4 = transit| (~bridge  & ~railway  & ~road)
			column5 = organization | (~company & ~education  & ~government  & ~military  & ~political_party & ~sports_team & ~stock_exchange)
			column6 = company  | (~broadcast & ~news)
			column7 = other | ( ~art & ~body_part & ~currency & ~event & ~food & ~health & ~heritage & ~legal & ~living_thing & ~product & ~religion & ~scientific & ~sports_and_leisure)
			column8 = art | (~broadcast & ~film & ~music & ~stage & ~writing)
			column9 = event | (~election & ~holiday  & ~natural_disaster  &~protest & ~sports_event & ~violent_conflict)
			column10 = health | ~treatment
			column11 = living_thing | ~animal
			column12 = structure | (~car & ~computer & ~software & ~weapon)
			column13 = person | (~artist & ~athlete & ~business & ~doctor & ~education & ~legal & ~military & ~political_figure & ~title)
			column14 = artist | (~actor & ~author & ~music)
			column15 = education | (~student & ~teacher)
			et_logical_formula = column1 | column2 | column3 | column4 | column5 | column6 | column7 | column8 | column9 | column10 | column11 | column12 | column13 | column14 | column15 
		elif self.dataset == 'figer_50k':
			art , film , astral_body , award , biology , body_part , broadcast , tv_channel , broadcast_network , broadcast_program , building , airport , dam , hospital , hotel , library , power_station , restaurant , sports_facility , theater , chemistry , computer , algorithm , programming_language , disease , education , department , educational_degree , event , attack , election , military_conflict , natural_disaster , protest , sports_event , terrorist_attack , finance , currency , stock_exchange , food , game , geography , glacier , island , mountain , god , government , government , political_party , government_agency , internet , website , language , law , living_thing , livingthing , animal , location , body_of_water , bridge , cemetery , city , country , county , province , medicine , drug , medical_treatment , symptom , metropolitan_transit , transit_line , military , music , news_agency , newspaper , organization , airline , company , educational_institution , fraternity_sorority , sports_league , sports_team , terrorist_organization , park , people , ethnicity , person , actor , architect , artist , athlete , author , coach , director , doctor , engineer , monarch , musician , politician , religious_leader , soldier , terrorist , play , product , airplane , camera , car , computer , instrument , mobile_phone , ship , spacecraft , weapon , rail , railway , religion , religion , software , time , title , train , transit , transportation , road , visual_art , color , written_work = sdd_mgr.vars
			et_logical_formula = art | (art & film) | astral_body | award | biology | body_part | broadcast | (broadcast & tv_channel) | broadcast_network | broadcast_program | building | (building & airport) | (building & dam) | (building & hospital) | (building & hotel) | (building & library) | (building & power_station) | (building & restaurant) | (building & sports_facility) | (building & theater) | chemistry | computer | (computer & algorithm) | (computer & programming_language) | disease | education | (education & department) | (education & educational_degree) | event | (event & attack) | (event & election) | (event & military_conflict) | (event & natural_disaster) | (event & protest) | (event & sports_event) | (event & terrorist_attack) | finance | (finance & currency) | (finance & stock_exchange) | food | game | geography | (geography & glacier) | (geography & island) | (geography & mountain) | god | government | (government & political_party) | government_agency | internet | (internet & website) | language | law  | living_thing | livingthing | (livingthing & animal) | location | (location & body_of_water) | (location & bridge) | (location & cemetery) | (location & city) | (location & country) | (location & county) | (location & province) | medicine | (medicine & drug) | (medicine & medical_treatment) | (medicine & symptom) | metropolitan_transit | (metropolitan_transit & transit_line) | military | music | news_agency | newspaper | organization | (organization & airline) | (organization & company) | (organization & educational_institution) | (organization & fraternity_sorority) | (organization & sports_league) | (organization & sports_team) | (organization & terrorist_organization) | park | people | (people & ethnicity) | person | (person & actor) | (person & architect) | (person & artist) | (person & athlete) | (person & author) | (person & coach) | (person & director) | (person & doctor) | (person & engineer) | (person & monarch) | (person & musician) | (person & politician) | (person & religious_leader) | (person & soldier) | (person & terrorist) | play | product | (product & airplane) | (product & camera) | (product & car) | (product & computer) | (product & instrument) | (product & mobile_phone) | (product & ship) | (product & spacecraft) | (product & weapon) | rail | (rail & railway) | religion | (religion & religion) | software | time | title | train | transit | transportation | (transportation & road) | visual_art | (visual_art & color) | written_work
		elif self.dataset == 'BBN':
			ANIMAL , CONTACT_INFO , url , DISEASE , EVENT , HURRICANE , WAR , FACILITY , AIRPORT , ATTRACTION , BRIDGE , BUILDING , HIGHWAY_STREET , GAME , GPE , CITY , COUNTRY , STATE_PROVINCE , LANGUAGE , LAW , LOCATION , CONTINENT , LAKE_SEA_OCEAN , REGION , RIVER , ORGANIZATION , CORPORATION , EDUCATIONAL , GOVERNMENT , HOSPITAL , HOTEL , MUSEUM , POLITICAL , RELIGIOUS , PERSON , PLANT , PRODUCT , VEHICLE , WEAPON , SUBSTANCE , CHEMICAL , DRUG , FOOD , WORK_OF_ART , BOOK , PLAY , SONG = sdd_mgr.vars
			# et_logical_formula = (~ANIMAL| ~CONTACT_INFO| ~DISEASE | ~EVENT | ~FACILITY| ~GAME  | ~GPE | ~LANGUAGE | ~LAW | ~LOCATION | ~ORGANIZATION | ~PERSON | ~PLANT | ~PRODUCT | ~SUBSTANCE | ~WORK_OF_ART) & (CONTACT_INFO | ~url) & (EVENT | ~HURRICANE | ~WAR) & (FACILITY | ~AIRPORT | ~ATTRACTION | ~BRIDGE |~ BUILDING | ~HIGHWAY_STREET) & (GPE | ~CITY | ~COUNTRY | ~STATE_PROVINCE ) & (LOCATION | ~CONTINENT | ~LAKE_SEA_OCEAN | ~REGION | ~RIVER ) & (ORGANIZATION | ~CORPORATION | ~EDUCATIONAL | ~GOVERNMENT | ~HOSPITAL | ~HOTEL | ~MUSEUM | ~POLITICAL | ~RELIGIOUS ) & (PRODUCT | ~VEHICLE | ~WEAPON ) &(SUBSTANCE | ~CHEMICAL | ~DRUG | ~FOOD ) & (WORK_OF_ART | ~BOOK | ~PLAY | ~SONG )		
			column1 = CONTACT_INFO | ~url 
			column2 = EVENT | (~HURRICANE & ~WAR)
			column3 = FACILITY | (~AIRPORT & ~ATTRACTION & ~BRIDGE & ~BUILDING & ~HIGHWAY_STREET)
			column4 = GPE | (~CITY  & ~COUNTRY  & ~STATE_PROVINCE)
			column5 = LOCATION | (~CONTINENT & ~LAKE_SEA_OCEAN  & ~REGION  & ~RIVER )
			column6 = ORGANIZATION | (~CORPORATION & ~EDUCATIONAL & ~GOVERNMENT & ~HOSPITAL & ~HOTEL & ~MUSEUM & ~POLITICAL & ~RELIGIOUS)
			column7 = PRODUCT  | (~VEHICLE & ~WEAPON)
			column8 = SUBSTANCE | (~CHEMICAL & ~DRUG & ~FOOD)
			column9 = WORK_OF_ART | (~BOOK & ~PLAY  & ~SONG)
			mutually_exclusive = self.exactly_one(ANIMAL, CONTACT_INFO, DISEASE,EVENT, FACILITY, GAME, GPE, LANGUAGE, LAW, LOCATION, ORGANIZATION, PERSON, PLANT, PRODUCT, SUBSTANCE, WORK_OF_ART)
			et_logical_formula = mutually_exclusive + (column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9)
		# #Minimize SDD
		# et_logical_formula.ref()
		# sdd_mgr.minimize_limited()

		wmc_mgr = WmcManager(et_logical_formula, log_mode = False)  
		wmc_mgr.propagate()				
		weights = torch.cat((every_, every_))
		for i, p in enumerate(every_):	
			weights[i] = 1.0 - every_[len(every_)-1-i]
		weights = array('d', weights)

		#Consitioned every literal to get joint probability of every literal and theory	
		likelihood = []
		wmc = []
		lits = [None] + [sdd_mgr.literal(i) for i in range(1, sdd_mgr.var_count() + 1)]
		for i in range(0, sdd_mgr.var_count()):	
			wmc_mgr.set_literal_weights_from_array(weights)
			wmc.append(wmc_mgr.propagate())
			# print("wmc:", wmc)		
			# Condition on lits[i+1] to be true.
			wmc_mgr.set_literal_weight(lits[i+1], 1)
			wmc_mgr.set_literal_weight(-lits[i+1], 0)
			every_likelihood = wmc_mgr.propagate()
			likelihood.append(every_likelihood)

		# with open(here / "output" / "sdd.dot", "w") as out:
		# 	print(mgr.dot(), file=out)
		# with open(here / "output" / "vtree.dot", "w") as out:
		# 	print(vtree.dot(), file=out)
		# print("done")

		return  wmc, likelihood



	def calculate_loss(self, y_hat, batch_y):	
		if self.use_hierarchy:
			if self.use_marginal_ranking_loss:
				loss_fn = MarginLoss(0.9, 0.1, 0.5)
			else:
				loss_fn = nn.BCELoss()
		else:
			loss_fn = nn.BCEWithLogitsLoss()

	
		return loss_fn(y_hat,batch_y)

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





