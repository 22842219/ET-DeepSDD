import os
import pysdd
from pysdd.sdd import Vtree, SddManager, WmcManager
from pysdd.iterator import SddIterator
from array import array
from graphviz import Source
import math
import numpy as np
import math
from pathlib import Path
here = Path(__file__).parent

def main(label_size= 56, dataset =  'bbn_original') :

	vtree = Vtree(label_size)
	sdd_mgr = SddManager(vtree = vtree)

	if dataset == 'bbn_modified':				
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
		# et_logical_formula = column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9
		et_logical_formula = (~ANIMAL | ~CONTACT_INFO | ~DISEASE | ~EVENT | ~FAC | ~GAME | ~GPE | ~ LANGUAGE | ~ LAW | ~ LOCATION | ~ ORGANIZATION | ~ PERSON | ~ PLANT | ~ PRODUCT | ~ SUBSTANCE | ~ WORK_OF_ART) & column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9
		# et_logical_formula = ANIMAL |DISEASE |GAME |LANGUAGE |LAW |PERSON |PLANT |column1 |column2 |column3|column4|column5|column6|column7|column8|column9
	elif dataset == 'bbn_original':
		ANIMAL, CONTACT_INFO, ADDRESS, PHONE, url, DISEASE, EVENT, HURRICANE, WAR, FAC, AIRPORT, ATTRACTION, BRIDGE, BUILDING, HIGHWAY_STREET, FACILITY, AIRPORT, ATTRACTION, BRIDGE, BUILDING, HIGHWAY_STREET, GAME, GPE, CITY, COUNTRY, STATE_PROVINCE, LANGUAGE, LAW, LOCATION, CONTINENT, LAKE_SEA_OCEAN, REGION, RIVER, ORGANIZATION, CORPORATION, EDUCATIONAL, GOVERNMENT, HOSPITAL, HOTEL, MUSEUM, POLITICAL, RELIGIOUS, PERSON, PLANT, PRODUCT, VEHICLE, WEAPON, SUBSTANCE, CHEMICAL, DRUG, FOOD, WORK_OF_ART, BOOK, PAINTING, PLAY, SONG  = sdd_mgr.vars
		et_logical_formula = ANIMAL| CONTACT_INFO| CONTACT_INFO & url| DISEASE| EVENT| EVENT & HURRICANE| EVENT & WAR| FACILITY| FACILITY & AIRPORT| FACILITY & ATTRACTION| FACILITY & BRIDGE| FACILITY & BUILDING| FACILITY & HIGHWAY_STREET| GAME| GPE| GPE & CITY| GPE & COUNTRY| GPE & STATE_PROVINCE| LANGUAGE| LAW| LOCATION| LOCATION & CONTINENT| LOCATION & LAKE_SEA_OCEAN| LOCATION & REGION| LOCATION & RIVER| ORGANIZATION| ORGANIZATION & CORPORATION| ORGANIZATION & EDUCATIONAL| ORGANIZATION & GOVERNMENT| ORGANIZATION & HOSPITAL| ORGANIZATION & HOTEL| ORGANIZATION & MUSEUM| ORGANIZATION & POLITICAL| ORGANIZATION & RELIGIOUS| PERSON| PLANT| PRODUCT| PRODUCT & VEHICLE| PRODUCT & WEAPON| SUBSTANCE| SUBSTANCE & CHEMICAL| SUBSTANCE & DRUG| SUBSTANCE & FOOD| WORK_OF_ART| WORK_OF_ART & BOOK| WORK_OF_ART & PLAY| WORK_OF_ART & SONG
		# et_logical_formula = (~ANIMAL| ~CONTACT_INFO| ~DISEASE | ~EVENT | ~FACILITY| ~GAME  | ~GPE | ~LANGUAGE | ~LAW | ~LOCATION | ~ORGANIZATION | ~PERSON | ~PLANT | ~PRODUCT | ~SUBSTANCE | ~WORK_OF_ART) & (CONTACT_INFO | ~url) & (EVENT | ~HURRICANE | ~WAR) & (FACILITY | ~AIRPORT | ~ATTRACTION | ~BRIDGE |~ BUILDING | ~HIGHWAY_STREET) & (GPE | ~CITY | ~COUNTRY | ~STATE_PROVINCE ) & (LOCATION | ~CONTINENT | ~LAKE_SEA_OCEAN | ~REGION | ~RIVER ) & (ORGANIZATION | ~CORPORATION | ~EDUCATIONAL | ~GOVERNMENT | ~HOSPITAL | ~HOTEL | ~MUSEUM | ~POLITICAL | ~RELIGIOUS ) & (PRODUCT | ~VEHICLE | ~WEAPON ) &(SUBSTANCE | ~CHEMICAL | ~DRUG | ~FOOD ) & (WORK_OF_ART | ~BOOK | ~PLAY | ~SONG )		
		# column1 = CONTACT_INFO | ~url 
		# column2 = EVENT | (~HURRICANE & ~WAR)
		# column3 = FACILITY | (~AIRPORT & ~ATTRACTION & ~BRIDGE & ~BUILDING & ~HIGHWAY_STREET)
		# column4 = GPE | (~CITY  & ~COUNTRY  & ~STATE_PROVINCE)
		# column5 = LOCATION | (~CONTINENT & ~LAKE_SEA_OCEAN  & ~REGION  & ~RIVER )
		# column6 = ORGANIZATION | (~CORPORATION & ~EDUCATIONAL & ~GOVERNMENT & ~HOSPITAL & ~HOTEL & ~MUSEUM & ~POLITICAL & ~RELIGIOUS)
		# column7 = PRODUCT  | (~VEHICLE & ~WEAPON)
		# column8 = SUBSTANCE | (~CHEMICAL & ~DRUG & ~FOOD)
		# column9 = WORK_OF_ART | (~BOOK & ~PLAY  & ~SONG)
		# et_logical_formula = ~ANIMAL| ~DISEASE| ~GAME| ~LANGUAGE| ~LAW| ~PERSON| ~PLANT| ~column1| ~column2| ~column3| ~column4| ~column5| ~column6| ~column7| ~column8| ~column9
		# et_logical_formula = self.exactly_one((ANIMAL, DISEASE, GAME, LANGUAGE, LAW, PERSON, PLANT, column9, column8, column7, column6, column5, column4, column3, column2, column1), sdd_mgr)
	elif dataset == 'ontonotes_modified':	
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
	elif dataset == 'figer_50k':
		art , film , astral_body , award , biology , body_part , broadcast , tv_channel , broadcast_network , broadcast_program , building , airport , dam , hospital , hotel , library , power_station , restaurant , sports_facility , theater , chemistry , computer , algorithm , programming_language , disease , education , department , educational_degree , event , attack , election , military_conflict , natural_disaster , protest , sports_event , terrorist_attack , finance , currency , stock_exchange , food , game , geography , glacier , island , mountain , god , government , government , political_party , government_agency , internet , website , language , law , living_thing , livingthing , animal , location , body_of_water , bridge , cemetery , city , country , county , province , medicine , drug , medical_treatment , symptom , metropolitan_transit , transit_line , military , music , news_agency , newspaper , organization , airline , company , educational_institution , fraternity_sorority , sports_league , sports_team , terrorist_organization , park , people , ethnicity , person , actor , architect , artist , athlete , author , coach , director , doctor , engineer , monarch , musician , politician , religious_leader , soldier , terrorist , play , product , airplane , camera , car , computer , instrument , mobile_phone , ship , spacecraft , weapon , rail , railway , religion , religion , software , time , title , train , transit , transportation , road , visual_art , color , written_work = sdd_mgr.vars
		et_logical_formula = art | (art & film) | astral_body | award | biology | body_part | broadcast | (broadcast & tv_channel) | broadcast_network | broadcast_program | building | (building & airport) | (building & dam) | (building & hospital) | (building & hotel) | (building & library) | (building & power_station) | (building & restaurant) | (building & sports_facility) | (building & theater) | chemistry | computer | (computer & algorithm) | (computer & programming_language) | disease | education | (education & department) | (education & educational_degree) | event | (event & attack) | (event & election) | (event & military_conflict) | (event & natural_disaster) | (event & protest) | (event & sports_event) | (event & terrorist_attack) | finance | (finance & currency) | (finance & stock_exchange) | food | game | geography | (geography & glacier) | (geography & island) | (geography & mountain) | god | government | (government & political_party) | government_agency | internet | (internet & website) | language | law  | living_thing | livingthing | (livingthing & animal) | location | (location & body_of_water) | (location & bridge) | (location & cemetery) | (location & city) | (location & country) | (location & county) | (location & province) | medicine | (medicine & drug) | (medicine & medical_treatment) | (medicine & symptom) | metropolitan_transit | (metropolitan_transit & transit_line) | military | music | news_agency | newspaper | organization | (organization & airline) | (organization & company) | (organization & educational_institution) | (organization & fraternity_sorority) | (organization & sports_league) | (organization & sports_team) | (organization & terrorist_organization) | park | people | (people & ethnicity) | person | (person & actor) | (person & architect) | (person & artist) | (person & athlete) | (person & author) | (person & coach) | (person & director) | (person & doctor) | (person & engineer) | (person & monarch) | (person & musician) | (person & politician) | (person & religious_leader) | (person & soldier) | (person & terrorist) | play | product | (product & airplane) | (product & camera) | (product & car) | (product & computer) | (product & instrument) | (product & mobile_phone) | (product & ship) | (product & spacecraft) | (product & weapon) | rail | (rail & railway) | religion | (religion & religion) | software | time | title | train | transit | transportation | (transportation & road) | visual_art | (visual_art & color) | written_work
	
	if dataset == 'MWO-Small':				
		Abbreviation, Activity, Agent, Attribute, Cardinality, Consumable, Event, Item, Location, Observation, Observed_state, Qualitative, Quantitative, Specifier, Time, Typo, Unsure = sdd_mgr.vars
		column1 = Observation | (~Observed_state & ~Qualitative & ~Quantitative)
		et_logical_formula = self.exactly_one((column1, Abbreviation, Activity, Agent, Attribute, Cardinality, Consumable, Event, Item, Location, Specifier, Time, Typo, Unsure), sdd_mgr)   

	# #Minimize SDD
	# et_logical_formula.ref()
	# sdd_mgr.minimize_limited()

	# weights = torch.cat((every_, every_))
	# for i, p in enumerate(every_):	
	# 	weights[i] = 1.0 - every_[len(every_)-1-i]
	# weights = array('d', weights)
	# wmc_mgr = WmcManager(et_logical_formula, log_mode = False)
	# wmc_mgr.set_literal_weights_from_array(weights)
	# wmc = wmc_mgr.propagate()
		
	

	# Consitioned every literal to get marginal weighted model couting
	marginal_wmc = []		
	wmc = []			
	lits = [None] + [sdd_mgr.literal(i) for i in range(1, sdd_mgr.var_count() + 1)]	
	for i in range(0, sdd_mgr.var_count()):	
		wmc_mgr = et_logical_formula.wmc(log_mode = False) 
		# wmc_mgr.set_literal_weights_from_array(weights)						
		wmc.append(wmc_mgr.propagate())
		# print("wmc:", wmc)
		# Condition on lits[i+1] to be true.
		wmc_mgr.set_literal_weight(lits[i+1], 1)
		wmc_mgr.set_literal_weight(-lits[i+1], 0)
		every_marginal_wmc = wmc_mgr.propagate()	#number of models where lits[i+1] being true
		marginal_wmc.append(every_marginal_wmc)
	


	# with open(here / "output" / "marginal_wmc", "a") as out:
	# 	print(marginal_wmc, file=out)
	with open(here / "output" / "bbn_original_baseline_sdd_wmc", "a") as out:
		print(wmc, file=out)
	with open(here / "output" / "bbn_original_baseline_sdd_marginal_wmc", "a") as out:
		print(marginal_wmc, file=out)
	# with open(here / "output" / "sdd.d
	# with open(here / "output" / "sdd.dot", "w") as out:
	# 	print(sdd_mgr.dot(), file=out)
	# with open(here / "output" / "vtree.dot", "w") as out:
	# 	print(vtree.dot(), file=out)
	# print("done")

	



	return    marginal_wmc, wmc, marginal_pr

# def main():
# 	parser = argparse.ArgumentParser(description = " wmc, marginal_wmc")
# 	parser.add_argument('-v', '--label_size', help = 'label size')
# 	parser.add_argument('-d', '--dataset')

# 	args = parser.parse_args()
# 	marginal_wmc, wmc = compute_wmc(args.label_size, args.dataset)


if __name__ == "__main__":
	main()