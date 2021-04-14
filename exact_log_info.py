from pathlib import Path
here = Path(__file__).parent
import re


infile = r"/home/ziyu/Desktop/bbn_modified_log.txt"

important = []
keep_phrases = ["Epoch",
  "Micro F1"]


with open(infile) as f:
	for line in f:	
		for phrase in keep_phrases:
			if phrase in line:
				important.append(line)
				break

# print(important)
with open(here / "outputs" / "bbn_modified_log", "a") as out:
	print(important, file=out)




