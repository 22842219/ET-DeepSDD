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

		# if self.use_hierarchy:		
		# 	for index, every_ in enumerate(y_hat):
		# 		normalized_logits = torch.sigmoid(every_)
		# 		wmc = self.compute_wmc(normalized_logits)
		# 		# wmc = torch.FloatTensor(wmc).to(device)                 
		# 		y_hat[index] = torch.mul(normalized_logits, wmc)
		# 		torch.clamp(y_hat, min=0.0, max=1.0)	
		return y_hat

	def AtMostOne(self,literals, mgr):
		alpha = mgr.false()

		for lit in literals:
			alpha += ~lit			
		return alpha

	def implication(self,literals, mgr):

		alpha = mgr.false()

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
		vtree = Vtree(self.label_size)
		sdd_mgr = SddManager(vtree = vtree)	
		# match = re.search('ontonotes_modified.*', self.dataset)

		if self.dataset == 'bbn_modified':						
			ANIMAL,CONTACT_INFO,ADDRESS,PHONE,DISEASE,EVENT,HURRICANE,WAR,FAC,AIRPORT,ATTRACTION,BRIDGE,BUILDING,HIGHWAY_STREET,GAME,GPE,CITY,COUNTRY,STATE_PROVINCE,LANGUAGE,LAW,LOCATION,CONTINENT,LAKE_SEA_OCEAN,REGION,RIVER,ORGANIZATION,CORPORATION,EDUCATIONAL,GOVERNMENT,HOSPITAL,HOTEL,POLITICAL,RELIGIOUS,PERSON,PLANT,PRODUCT,VEHICLE,WEAPON,SUBSTANCE,CHEMICAL,DRUG,FOOD,WORK_OF_ART,BOOK,PAINTING,PLAY,SONG = sdd_mgr.vars
			column1 = CONTACT_INFO | (~ADDRESS & ~PHONE)
			column2 = EVENT | (~HURRICANE  & ~WAR)
			column3 = FAC | (~AIRPORT  & ~ATTRACTION  & ~BRIDGE  & ~BUILDING  & ~HIGHWAY_STREET)
			column4 = GPE | (~CITY  & ~COUNTRY  & ~STATE_PROVINCE)
			column5 = LOCATION  | (~CONTINENT & ~LAKE_SEA_OCEAN  & ~REGION  & ~RIVER )
			column6 = ORGANIZATION | (~CORPORATION  & ~EDUCATIONAL  & ~GOVERNMENT  & ~HOSPITAL  & ~HOTEL & ~POLITICAL & ~RELIGIOUS )
			column7 = PRODUCT  | (~VEHICLE  & ~WEAPON)
			column8 = SUBSTANCE | (~CHEMICAL  & ~DRUG  & ~FOOD)
			column9 = WORK_OF_ART | (~BOOK & ~PAINTING  & ~PLAY  & ~SONG)
			mutually_exclusive = self.exactly_one((ANIMAL, CONTACT_INFO, DISEASE, EVENT, FAC, GAME, GPE, LANGUAGE, LAW, LOCATION, ORGANIZATION, PERSON, PLANT, PRODUCT, SUBSTANCE, WORK_OF_ART), sdd_mgr)
			et_logical_formula = mutually_exclusive & column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9
			# et_logical_formula = (~ANIMAL | ~CONTACT_INFO | ~DISEASE | ~EVENT | ~FAC | ~GAME | ~GPE | ~ LANGUAGE | ~ LAW | ~ LOCATION | ~ ORGANIZATION | ~ PERSON | ~ PLANT | ~ PRODUCT | ~ SUBSTANCE | ~ WORK_OF_ART) & column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9
		elif self.dataset == 'bbn_original':
			ANIMAL,CONTACT_INFO,ADDRESS,PHONE,url,DISEASE,EVENT,HURRICANE,WAR,FAC,AIRPORT,ATTRACTION,BRIDGE,BUILDING,HIGHWAY_STREET,FACILITY,AIRPORT,ATTRACTION,BRIDGE,BUILDING,HIGHWAY_STREET,GAME,GPE,CITY,COUNTRY,STATE_PROVINCE,LANGUAGE,LAW,LOCATION,CONTINENT,LAKE_SEA_OCEAN,REGION,RIVER,ORGANIZATION,CORPORATION,EDUCATIONAL,GOVERNMENT,HOSPITAL,HOTEL,MUSEUM,POLITICAL,RELIGIOUS,PERSON,PLANT,PRODUCT,VEHICLE,WEAPON,SUBSTANCE,CHEMICAL,DRUG,FOOD,WORK_OF_ART,BOOK,PAINTING,PLAY,SONG = sdd_mgr.vars
			# et_logical_formula = ANIMAL| CONTACT_INFO| CONTACT_INFO & url| DISEASE| EVENT| EVENT & HURRICANE| EVENT & WAR| FACILITY| FACILITY & AIRPORT| FACILITY & ATTRACTION| FACILITY & BRIDGE| FACILITY & BUILDING| FACILITY & HIGHWAY_STREET| GAME| GPE| GPE & CITY| GPE & COUNTRY| GPE & STATE_PROVINCE| LANGUAGE| LAW| LOCATION| LOCATION & CONTINENT| LOCATION & LAKE_SEA_OCEAN| LOCATION & REGION| LOCATION & RIVER| ORGANIZATION| ORGANIZATION & CORPORATION| ORGANIZATION & EDUCATIONAL| ORGANIZATION & GOVERNMENT| ORGANIZATION & HOSPITAL| ORGANIZATION & HOTEL| ORGANIZATION & MUSEUM| ORGANIZATION & POLITICAL| ORGANIZATION & RELIGIOUS| PERSON| PLANT| PRODUCT| PRODUCT & VEHICLE| PRODUCT & WEAPON| SUBSTANCE| SUBSTANCE & CHEMICAL| SUBSTANCE & DRUG| SUBSTANCE & FOOD| WORK_OF_ART| WORK_OF_ART & BOOK| WORK_OF_ART & PLAY| WORK_OF_ART & SONG
			# et_logical_formula = (~ANIMAL| ~CONTACT_INFO| ~DISEASE | ~EVENT | ~FACILITY| ~GAME  | ~GPE | ~LANGUAGE | ~LAW | ~LOCATION | ~ORGANIZATION | ~PERSON | ~PLANT | ~PRODUCT | ~SUBSTANCE | ~WORK_OF_ART) & (CONTACT_INFO | ~url) & (EVENT | ~HURRICANE | ~WAR) & (FACILITY | ~AIRPORT | ~ATTRACTION | ~BRIDGE |~ BUILDING | ~HIGHWAY_STREET) & (GPE | ~CITY | ~COUNTRY | ~STATE_PROVINCE ) & (LOCATION | ~CONTINENT | ~LAKE_SEA_OCEAN | ~REGION | ~RIVER ) & (ORGANIZATION | ~CORPORATION | ~EDUCATIONAL | ~GOVERNMENT | ~HOSPITAL | ~HOTEL | ~MUSEUM | ~POLITICAL | ~RELIGIOUS ) & (PRODUCT | ~VEHICLE | ~WEAPON ) &(SUBSTANCE | ~CHEMICAL | ~DRUG | ~FOOD ) & (WORK_OF_ART | ~BOOK | ~PLAY | ~SONG )		
			column1 = CONTACT_INFO | (~ADDRESS & ~ PHONE & ~url)
			column2 = EVENT | (~HURRICANE & ~WAR)
			column3 = FAC | FACILITY | (~AIRPORT  & ~ ATTRACTION  & ~ BRIDGE  & ~ BUILDING  & ~ HIGHWAY_STREET)
			column4 = GPE | (~CITY  & ~ COUNTRY  & ~ STATE_PROVINCE)
			column5 = LOCATION | (~CONTINENT & ~LAKE_SEA_OCEAN  & ~REGION  & ~RIVER )
			column6 = ORGANIZATION | (~CORPORATION & ~EDUCATIONAL & ~GOVERNMENT & ~HOSPITAL & ~HOTEL & ~MUSEUM & ~POLITICAL & ~RELIGIOUS)
			column7 = PRODUCT  | (~VEHICLE & ~WEAPON)
			column8 = SUBSTANCE | (~CHEMICAL & ~DRUG & ~FOOD)
			column9 = WORK_OF_ART | (~BOOK & ~PAINTING &  ~PLAY  & ~SONG) 
			mutually_exclusive = self.exactly_one((ANIMAL, CONTACT_INFO, DISEASE, EVENT, FAC, GAME, GPE, LANGUAGE, LAW, LOCATION, ORGANIZATION, PERSON, PLANT, PRODUCT, SUBSTANCE, WORK_OF_ART), sdd_mgr)   
			et_logical_formula = mutually_exclusive & column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9 
			# et_logical_formula = (~ANIMAL| ~CONTACT_INFO| ~DISEASE| ~EVENT| ~FAC| ~FACILITY| ~GAME| ~GPE| ~LANGUAGE| ~LAW| ~LOCATION|  ~ORGANIZATION| ~PERSON| ~PLANT| ~PRODUCT| ~SUBSTANCE| ~WORK_OF_ART)& column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9 & column10 & column11

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
			column12 = product | (~car & ~computer & ~software & ~weapon)
			column13 = person | (~artist & ~athlete & ~business & ~doctor & ~education & ~legal & ~military & ~political_figure & ~title)
			column14 = artist | (~actor & ~author & ~music)
			column15 = education | (~student & ~teacher)
			mutually_exclusive = self.exactly_one((other, person, organization),sdd_mgr)
			et_logical_formula = mutually_exclusive | (column1 & column2 & column3 & column4 )| (column5 & column6 ) | (column7 & column8 & column9 & column10 & column11 &column12) |( column13 & column14 & column15 )
		elif self.dataset == 'ontonotes_original':
			location,celestial,city,country,geography,body_of_water,island,mountain,geograpy,island,park,structure,airport,government,hospital,hotel,restaurant,sports_facility,theater,transit,bridge,railway,road,organization,company,broadcast,news,education,government,military,music,political_party,sports_league,sports_team,stock_exchange,transit,other,art,broadcast,film,music,stage,writing,award,body_part,currency,event,accident,election,holiday,natural_disaster,protest,sports_event,violent_conflict,food,health,malady,treatment,heritage,internet,language,programming_language,legal,living_thing,animal,product,car,computer,mobile_phone,software,weapon,religion,scientific,sports_and_leisure,supernatural,person,artist,actor,author,director,music,athlete,business,coach,doctor,education,student,teacher,legal,military,political_figure,religious_leader,title = sdd_mgr.vars
			column1 = location | (~celestial & ~ city& ~country & ~geography & ~park & ~structure & ~transit )
			column2 = geography | (~body_of_water & ~island & ~mountain)
			column3 = structure | (~airport & ~government & ~hospital & ~hotel & ~restaurant & ~sports_facility  & ~theater )
			column4 = transit| (~bridge  & ~railway  & ~road)
			column5 = organization | (~company & ~education  & ~government  & ~military & ~music & ~political_party  & ~sports_league & ~sports_team & ~stock_exchange & ~transit)
			column6 = company  | (~broadcast & ~news)
			column7 = other | ( ~art & ~award & ~body_part & ~currency & ~event & ~food & ~health & ~heritage & ~internet & ~language & ~legal & ~living_thing & ~product & ~religion & ~scientific & ~sports_and_leisure & ~supernatural)
			column8 = art | (~broadcast & ~film & ~music & ~stage & ~writing)
			column9 = event | (~accident & ~election & ~holiday  & ~natural_disaster  &~protest & ~sports_event & ~violent_conflict)
			column10 = health | ( ~malady & ~treatment)
			column11 = language | ~programming_language
			column12 = living_thing | ~animal
			column13 = product | (~car & ~computer & ~software & ~weapon & ~mobile_phone)
			column14 = person | (~artist & ~athlete & ~business & ~coach & ~doctor & ~education & ~legal & ~military & ~political_figure & ~religious_leader & ~title)
			column15 = artist | (~actor & ~author & ~director & ~music)
			column16 = education | (~student & ~teacher)
			et_logical_formula = (column1 & column2 & column3 & column4 )| (column5 & column6 ) | (column7 & column8 & column9 & column10 & column11 &column12 & column13) |( column14 & column15 & column16 )
			# et_logical_formula = location|(location & celestial)|(location & city)|(location & country)|(location & geography)|(location & geography & body_of_water)|(location & geography & island)|(location & geography & mountain)|(location & geograpy)|(location & geograpy & island)|(location & park)|(location & structure)|(location & structure & airport)|(location & structure & government)|(location & structure & hospital)|(location & structure & hotel)|(location & structure & restaurant)|(location & structure & sports_facility)|(location & structure & theater)|(location & transit)|(location & transit & bridge)|(location & transit & railway)|(location & transit & road)|organization|(organization & company)|(organization & company & broadcast)|(organization & company & news)|(organization & education)|(organization & government)|(organization & military)|(organization & music)|(organization & political_party)|(organization & sports_league)|(organization & sports_team)|(organization & stock_exchange)|(organization & transit)|other|(other & art)|(other & art & broadcast)|(other & art & film)|(other & art & music)|(other & art & stage)|(other & art & writing)|(other & award)|(other & body_part)|(other & currency)|(other & event)|(other & event & accident)|(other & event & election)|(other & event & holiday)|(other & event & natural_disaster)|(other & event & protest)|(other & event & sports_event)|(other & event & violent_conflict)|(other & food)|(other & health)|(other & health & malady)|(other & health & treatment)|(other & heritage)|(other & internet)|(other & language)|(other & language & programming_language)|(other & legal)|(other & living_thing)|(other & living_thing & animal)|(other & product)|(other & product & car)|(other & product & computer)|(other & product & mobile_phone)|(other & product & software)|(other & product & weapon)|(other & religion)|(other & scientific)|(other & sports_and_leisure)|(other & supernatural)|person|(person & artist)|(person & artist & actor)|(person & artist & author)|(person & artist & director)|(person & artist & music)|(person & athlete)|(person & business)|(person & coach)|(person & doctor)|(person & education)|(person & education & student)|(person & education & teacher)|(person & legal)|(person & military)|(person & political_figure)|(person & religious_leader)|(person & title)
		elif self.dataset == 'figer_50k':
			art , film , astral_body , award , biology , body_part , broadcast , tv_channel , broadcast_network , broadcast_program , building , airport , dam , hospital , hotel , library , power_station , restaurant , sports_facility , theater , chemistry , computer , algorithm , programming_language , disease , education , department , educational_degree , event , attack , election , military_conflict , natural_disaster , protest , sports_event , terrorist_attack , finance , currency , stock_exchange , food , game , geography , glacier , island , mountain , god , government , government , political_party , government_agency , internet , website , language , law , living_thing , livingthing , animal , location , body_of_water , bridge , cemetery , city , country , county , province , medicine , drug , medical_treatment , symptom , metropolitan_transit , transit_line , military , music , news_agency , newspaper , organization , airline , company , educational_institution , fraternity_sorority , sports_league , sports_team , terrorist_organization , park , people , ethnicity , person , actor , architect , artist , athlete , author , coach , director , doctor , engineer , monarch , musician , politician , religious_leader , soldier , terrorist , play , product , airplane , camera , car , computer , instrument , mobile_phone , ship , spacecraft , weapon , rail , railway , religion , religion , software , time , title , train , transit , transportation , road , visual_art , color , written_work = sdd_mgr.vars
			column1 = art | ~film
			column2 = broadcast | ~tv_channel
			column3 = building | (~airport & ~dam & ~hospital & ~hotel & ~library & ~power_station & ~restaurant & ~sports_facility & ~theater)
			column4 = computer | (~algorithm & ~programming_language)
			column5 = education | (~department & ~educational_degree)
			column6 = event | (~attack & ~election & ~military_conflict & ~natural_disaster & ~protest & ~sports_event & ~terrorist_attack)
			column7 = finance | (~currency & ~stock_exchange)
			column8 = geography | (~glacier & ~island & ~mountain)
			column9 = government | ~political_party
			column10 = internet | ~website
			column11 = livingthing | ~animal
			column12 = location | (~body_of_water & ~bridge & ~cemetery & ~city & ~country & ~county & ~province)
			column13 = medicine | (~drug & ~medical_treatment & ~symptom)
			column14 = metropolitan_transit | (~transit_line)
			column15 = organization | (~airline & ~company & ~educational_institution & ~fraternity_sorority & ~sports_league & ~sports_team & ~terrorist_organization)
			column16 = people | ~ethnicity
			column17 = person | (~actor & ~architect & ~artist & ~athlete & ~author & ~coach & ~director & ~doctor & ~engineer & ~monarch & ~musician & ~politician & ~religious_leader & ~soldier & ~terrorist)
			column18 = product | (~airplane & ~camera & ~car & ~computer & ~instrument & ~mobile_phone & ~ship & ~spacecraft & ~weapon)
			column19 = rail | ~railway
			column20 = religion | ~religion
			column21 = transportation | ~road
			column22 = visual_art | ~color
			et_logical_formula =column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9 & column10 & column11 & column12 & column13 & column14 & column15 & column16 & column17 & column18 & column19 & column20 & column21 & column22 
			# et_logical_formula = art | (art & film) | astral_body | award | biology | body_part | broadcast | (broadcast & tv_channel) | broadcast_network | broadcast_program | building | (building & airport) | (building & dam) | (building & hospital) | (building & hotel) | (building & library) | (building & power_station) | (building & restaurant) | (building & sports_facility) | (building & theater) | chemistry | computer | (computer & algorithm) | (computer & programming_language) | disease | education | (education & department) | (education & educational_degree) | event | (event & attack) | (event & election) | (event & military_conflict) | (event & natural_disaster) | (event & protest) | (event & sports_event) | (event & terrorist_attack) | finance | (finance & currency) | (finance & stock_exchange) | food | game | geography | (geography & glacier) | (geography & island) | (geography & mountain) | god | government | (government & political_party) | government_agency | internet | (internet & website) | language | law  | living_thing | livingthing | (livingthing & animal) | location | (location & body_of_water) | (location & bridge) | (location & cemetery) | (location & city) | (location & country) | (location & county) | (location & province) | medicine | (medicine & drug) | (medicine & medical_treatment) | (medicine & symptom) | metropolitan_transit | (metropolitan_transit & transit_line) | military | music | news_agency | newspaper | organization | (organization & airline) | (organization & company) | (organization & educational_institution) | (organization & fraternity_sorority) | (organization & sports_league) | (organization & sports_team) | (organization & terrorist_organization) | park | people | (people & ethnicity) | person | (person & actor) | (person & architect) | (person & artist) | (person & athlete) | (person & author) | (person & coach) | (person & director) | (person & doctor) | (person & engineer) | (person & monarch) | (person & musician) | (person & politician) | (person & religious_leader) | (person & soldier) | (person & terrorist) | play | product | (product & airplane) | (product & camera) | (product & car) | (product & computer) | (product & instrument) | (product & mobile_phone) | (product & ship) | (product & spacecraft) | (product & weapon) | rail | (rail & railway) | religion | (religion & religion) | software | time | title | train | transit | transportation | (transportation & road) | visual_art | (visual_art & color) | written_work
		elif self.dataset == 'figer':
			art,film,astral_body,award,biology,body_part,broadcast,tv_channel,broadcast_network,broadcast_program,building,airport,dam,hospital,hotel,library,power_station,restaurant,sports_facility,theater,chemistry,computer,algorithm,programming_language,disease,education,department,educational_degree,event,attack,election,military_conflict,natural_disaster,protest,sports_event,terrorist_attack,finance,currency,stock_exchange,food,game,geography,glacier,island,mountain,god,government,government,political_party,government_agency,internet,website,language,law,living_thing,livingthing,animal,location,body_of_water,bridge,cemetery,city,country,county,province,medicine,drug,medical_treatment,symptom,metropolitan_transit,transit_line,military,music,news_agency,newspaper,organization,airline,company,educational_institution,fraternity_sorority,sports_league,sports_team,terrorist_organization,park,people,ethnicity,person,actor,architect,artist,athlete,author,coach,director,doctor,engineer,monarch,musician,politician,religious_leader,soldier,terrorist,play,product,airplane,camera,car,computer,engine_device,instrument,mobile_phone,ship,spacecraft,weapon,rail,railway,religion,religion,software,time,title,train,transit,transportation,road,visual_art,color,written_work = sdd_mgr.vars
			column1 = art | ~film
			column2 = broadcast | ~tv_channel
			column3 = building | (~airport & ~dam & ~hospital & ~hotel & ~library & ~power_station & ~restaurant & ~sports_facility & ~theater)
			column4 = computer | (~algorithm & ~programming_language)
			column5 = education | (~department & ~educational_degree)
			column6 = event | (~attack & ~election & ~military_conflict & ~natural_disaster & ~protest & ~sports_event & ~terrorist_attack)
			column7 = finance | (~currency & ~stock_exchange)
			column8 = geography | (~glacier & ~island & ~mountain)
			column9 = government | (~government & ~political_party)
			column10 = internet | ~website
			column11 = livingthing | ~animal
			column12 = location | (~body_of_water & ~bridge & ~cemetery & ~city & ~country & ~county & ~province)
			column13 = medicine | (~drug & ~medical_treatment & ~symptom)
			column14 = metropolitan_transit | ~transit_line
			column15 = organization | (~airline & ~company & ~educational_institution & ~fraternity_sorority & ~sports_league & ~sports_team & ~terrorist_organization)
			column16 = people | ~ethnicity
			column17 = person | (~actor & ~architect & ~artist & ~athlete & ~author & ~coach & ~director & ~doctor & ~engineer & ~monarch & ~musician & ~politician & ~religious_leader & ~soldier & ~terrorist)
			column18 = product | (~airplane & ~camera & ~car & ~computer & ~engine_device & ~instrument & ~mobile_phone & ~ship & ~spacecraft & ~weapon)
			column19 = rail | ~railway
			column20 = religion | ~religion
			column21 = transportation | ~road
			column22 = visual_art | ~color
			et_logical_formula =column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9 & column10 & column11 & column12 & column13 & column14 & column15 & column16 & column17 & column18 & column19 & column20 & column21 & column22 
		elif self.dataset == 'MWO-Small':				
			Abbreviation, Activity, Agent, Attribute, Cardinality, Consumable, Event, Item, Location, Observation, Observed_state, Qualitative, Quantitative, Specifier, Time, Typo, Unsure = sdd_mgr.vars
			column1 = Observation | (~Observed_state & ~Qualitative & ~Quantitative)
			et_logical_formula = self.exactly_one((Abbreviation, Activity, Agent, Attribute, Cardinality, Consumable, Event, Item, Location, Observation, Specifier, Time, Typo, Unsure),sdd_mgr)& column1



		weights = torch.cat((every_, every_))
		for i, p in enumerate(every_):	
			weights[i] = 1.0 - every_[len(every_)-1-i]
		weights = array('d', weights)

		# Consitioned every literal to get marginal weighted model couting
		conditioned_wmc = []		
		wmc = []			
		lits = [None] + [sdd_mgr.literal(i) for i in range(1, sdd_mgr.var_count() + 1)]	
		for i in range(0, sdd_mgr.var_count()):	
			wmc_mgr = et_logical_formula.wmc(log_mode = True) 
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

