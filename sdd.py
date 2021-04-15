import pysdd
from pysdd.sdd import Vtree, SddManager, WmcManager
from pysdd.iterator import SddIterator
from array import array
from graphviz import Source

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



def main():
	# set up vtree and manager
	vtree = Vtree(72)
	sdd_mgr = SddManager(vtree = vtree)	

	# construct the formula 
	print("constructing SDD ... ")
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

	print("saving sdd and vtree ... ")
	with open("sdd_output/sdd.dot", "w") as out:
		print(et_logical_formula.dot(), file=out)
	with open("sdd_output/vtree.dot", "w") as out:
		print(vtree.dot(), file=out)

	wmc_mgr = et_logical_formula.wmc(log_mode = True)
	sdd_nodes = wmc_mgr.node

	vtree.save(b"sdd_output/et_logical_formula.vtree")
	sdd_mgr.save(b"sdd_output/et_logical_formula.sdd", sdd_nodes)

	print("done")


if __name__ == "__main__":
	main()