import pysdd
from pysdd.sdd import Vtree, SddManager, WmcManager
from pysdd.iterator import SddIterator
from array import array
from graphviz import Source
import sys, os

from pathlib import Path
here = Path(__file__).parent
print(here
	)

def AtMostOne(literals, mgr):
		alpha = mgr.false()

		for lit in literals:
			alpha += ~lit			
		return alpha

def implication(literals, mgr):

	alpha = mgr.false()

	beta0 = literals[0]
	for lit in literals[1:]:
		beta = ~lit | beta0  			
		alpha = alpha | beta
	return alpha

def exactly_one(lits, mgr):
	alpha = mgr.false()
	for lit in lits:
		beta = lit
		for lit2 in lits:
			if lit2!=lit:
				beta = beta & ~lit2
		alpha = alpha | beta
	return alpha



def main(argv):
	"""
	Creates the vtree and sdd format files of each dataset. 
	"""
	args = _argparser().parse_args(argv[1:])
	# sdd = args.sdd
	dataset = args.dataset
	label_size = args.label_size
	# set up vtree and manager
	vtree = Vtree(label_size)
	sdd_mgr = SddManager(vtree = vtree)	
	folder ='{}/{}/{}/'.format(here, "sdd_input", dataset)
	if not os.path.exists(os.path.dirname(folder)):
		try:
			os.makedirs(os.path.dirname(folder))
		except OSError as exc:
			if exc.errno != errno.EEXITST:
				raise

	# construct the formula 
	if dataset == 'bbn_modified':						
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
		mutually_exclusive = exactly_one((ANIMAL, CONTACT_INFO, DISEASE, EVENT, FAC, GAME, GPE, LANGUAGE, LAW, LOCATION, ORGANIZATION, PERSON, PLANT, PRODUCT, SUBSTANCE, WORK_OF_ART), sdd_mgr)
		et_logical_formula = mutually_exclusive & column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9
		# et_logical_formula = (~ANIMAL | ~CONTACT_INFO | ~DISEASE | ~EVENT | ~FAC | ~GAME | ~GPE | ~ LANGUAGE | ~ LAW | ~ LOCATION | ~ ORGANIZATION | ~ PERSON | ~ PLANT | ~ PRODUCT | ~ SUBSTANCE | ~ WORK_OF_ART) & column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9
	elif dataset == 'bbn_original':
		ANIMAL,CONTACT_INFO,ADDRESS,PHONE,url,DISEASE,EVENT,HURRICANE,WAR,FAC,AIRPORT,ATTRACTION,BRIDGE,BUILDING,HIGHWAY_STREET,FACILITY,AIRPORT,ATTRACTION,BRIDGE,BUILDING,HIGHWAY_STREET,GAME,GPE,CITY,COUNTRY,STATE_PROVINCE,LANGUAGE,LAW,LOCATION,CONTINENT,LAKE_SEA_OCEAN,REGION,RIVER,ORGANIZATION,CORPORATION,EDUCATIONAL,GOVERNMENT,HOSPITAL,HOTEL,MUSEUM,POLITICAL,RELIGIOUS,PERSON,PLANT,PRODUCT,VEHICLE,WEAPON,SUBSTANCE,CHEMICAL,DRUG,FOOD,WORK_OF_ART,BOOK,PAINTING,PLAY,SONG = sdd_mgr.vars
		column1 = CONTACT_INFO | (~ADDRESS & ~ PHONE & ~url)
		column2 = EVENT | (~HURRICANE & ~WAR)
		column3 = FAC | FACILITY | (~AIRPORT  & ~ ATTRACTION  & ~ BRIDGE  & ~ BUILDING  & ~ HIGHWAY_STREET)
		column4 = GPE | (~CITY  & ~ COUNTRY  & ~ STATE_PROVINCE)
		column5 = LOCATION | (~CONTINENT & ~LAKE_SEA_OCEAN  & ~REGION  & ~RIVER )
		column6 = ORGANIZATION | (~CORPORATION & ~EDUCATIONAL & ~GOVERNMENT & ~HOSPITAL & ~HOTEL & ~MUSEUM & ~POLITICAL & ~RELIGIOUS)
		column7 = PRODUCT  | (~VEHICLE & ~WEAPON)
		column8 = SUBSTANCE | (~CHEMICAL & ~DRUG & ~FOOD)
		column9 = WORK_OF_ART | (~BOOK & ~PAINTING &  ~PLAY  & ~SONG) 
		mutually_exclusive = exactly_one((ANIMAL, CONTACT_INFO, DISEASE, EVENT, FAC, GAME, GPE, LANGUAGE, LAW, LOCATION, ORGANIZATION, PERSON, PLANT, PRODUCT, SUBSTANCE, WORK_OF_ART), sdd_mgr)   
		et_logical_formula = mutually_exclusive & column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9 
		# et_logical_formula = (~ANIMAL| ~CONTACT_INFO| ~DISEASE| ~EVENT| ~FAC| ~FACILITY| ~GAME| ~GPE| ~LANGUAGE| ~LAW| ~LOCATION|  ~ORGANIZATION| ~PERSON| ~PLANT| ~PRODUCT| ~SUBSTANCE| ~WORK_OF_ART)& column1 & column2 & column3 & column4 & column5 & column6 & column7 & column8 & column9 & column10 & column11

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
		column12 = product | (~car & ~computer & ~software & ~weapon)
		column13 = person | (~artist & ~athlete & ~business & ~doctor & ~education & ~legal & ~military & ~political_figure & ~title)
		column14 = artist | (~actor & ~author & ~music)
		column15 = education | (~student & ~teacher)
		mutually_exclusive = exactly_one((other, person, organization),sdd_mgr)
		et_logical_formula = mutually_exclusive | (column1 & column2 & column3 & column4 )| (column5 & column6 ) | (column7 & column8 & column9 & column10 & column11 &column12) |( column13 & column14 & column15 )
	elif dataset == 'ontonotes_original':
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
	elif dataset == 'figer_50k':
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
	elif dataset == 'figer':
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
	elif dataset == 'MWO-Small':				
		Abbreviation, Activity, Agent, Attribute, Cardinality, Consumable, Event, Item, Location, Observation, Observed_state, Qualitative, Quantitative, Specifier, Time, Typo, Unsure = sdd_mgr.vars
		column1 = Observation | (~Observed_state & ~Qualitative & ~Quantitative)
		et_logical_formula = exactly_one((Abbreviation, Activity, Agent, Attribute, Cardinality, Consumable, Event, Item, Location, Observation, Specifier, Time, Typo, Unsure),sdd_mgr)& column1


	print("saving sdd and vtree ... ")
	with open(folder+"sdd.dot", "w") as out:
		print(et_logical_formula.dot(), file=out)
	with open(folder +"vtree.dot", "w") as out:
		print(vtree.dot(), file=out)

	wmc_mgr = et_logical_formula.wmc(log_mode = True)
	sdd_nodes = wmc_mgr.node

	vtree.save(bytes(here/folder/"et.vtree"))
	sdd_mgr.save(bytes(here/folder/"et.sdd"), sdd_nodes)

	print("done")



def _argparser():
	import argparse
	parser = argparse.ArgumentParser()
	# parser.add_argument('sdd', help="")
	parser.add_argument('--dataset', '-d', type=str, help='dataset.')
	parser.add_argument('--label_size', '-n', type=int, help='label size.')
	return parser


if __name__ == '__main__':
    main(sys.argv)
