import torch
from torch.utils.data import DataLoader
import sys, random
import numpy as np
import pickle as pkl
import codecs, jsonlines
from logger import logger

import data_utils as dutils
from data_utils import CategoryHierarchy, EntityTypingDataset, MentionTypingDataset
from train import train_without_loading
from evaluate import evaluate_without_loading
from bert_encoder import  get_contextualizer
from load_config import load_config, device
cf = load_config()

from pathlib import Path
here = Path(__file__).parent

# Ensure deterministic behavior
torch.manual_seed(0xDEADBEEF)
np.random.seed(0xDEADBEEF)
random.seed(0xDEADBEEF)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Sentence(object):
	def __init__(self, 		
				tokens, 
				token_ids,
				labels,  
				wordpieces, 				 
				wordpiece_ids, 
				token_idxs_to_wp_idxs):
	
		self.tokens = tokens
		self.token_ids = token_ids
		self.labels = labels

		self.wordpieces = wordpieces
		self.wordpiece_ids = wordpiece_ids

		self.token_idxs_to_wp_idxs = token_idxs_to_wp_idxs

	# Return the data corresponding to this sentence.
	def data(self):
		return self.tokens, self.token_ids, self.labels, self.wordpieces, self.wordpiece_ids, self.token_idxs_to_wp_idxs

	# Print out a summary of the sentence.
	def __repr__(self):
		s =  "tokens:             %s\n" % self.tokens
		s += "token_ids:          %s\n" % self.token_ids 
		s += "labels:           %s\n" % self.labels 
		s += "wordpieces:         %s\n" % self.wordpieces 
		s += "wordpiece_ids:   %s\n" % self.wordpiece_ids 
		s += "Token map:          %s\n" % self.token_idxs_to_wp_idxs
		return s

# The Mention stores one single training/testing example for the Mention-level model.
# It is a subclass of the Sentence class.
class Mention(Sentence):
	def __init__(self, 		
				tokens, 
				token_ids,
				labels,  
				wordpieces, 				 
				wordpiece_ids, 
				token_idxs_to_wp_idxs, 
				start, 
				end):
		super(Mention, self).__init__(tokens, token_ids, labels, wordpieces, wordpiece_ids, token_idxs_to_wp_idxs)		


		self.token_mention_start = start
		self.token_mention_end = end		

		self.wp_mention_start = self.token_idxs_to_wp_idxs.index(start)
		self.wp_mention_end = self.token_idxs_to_wp_idxs.index(end)

		# if self.wp_mention_start == -1:
		# 	self.wp_mention_start = self.wordpieces.index(u"[CLS]")

		self.build_context_wordpieces()			
		
	def build_context_wordpieces(self):
		token_span 	 = cf.MODEL_OPTIONS['context_window']
		mention_span = cf.MODEL_OPTIONS['mention_window']
		wp_l_span = max(0, self.token_mention_start - token_span)
		wp_r_span  = min(self.token_mention_end + token_span, len(self.token_ids)) 
	
	
		self.ctx_left = self.token_ids[max(0, self.token_mention_start - token_span) : self.token_mention_start]
		self.ctx_right = self.token_ids[self.token_mention_end : min(self.token_mention_end + token_span, len(self.token_ids))]
		self.ctx_mention = self.token_ids[self.token_mention_start : self.token_mention_end]	
		self.ctx_all = self.ctx_left + self.ctx_mention + self.ctx_right
		
		self.wp_ids_left = self.wordpiece_ids[self.token_idxs_to_wp_idxs.index(wp_l_span) : self.wp_mention_start]
		self.wp_ids_right = self.wordpiece_ids[self.wp_mention_end : self.token_idxs_to_wp_idxs.index(wp_r_span)]	
		self.wp_ids_mention = self.wordpiece_ids[self.wp_mention_start : self.wp_mention_end]	
		self.wp_ids_all = self.wp_ids_left + self.wp_ids_mention + self.wp_ids_right

		self.wp_ids_left_padding = self.wp_ids_left[:token_span] + [0]*(token_span - len(self.wp_ids_left ))
		self.wp_ids_right_padding = self.wp_ids_right[:token_span] + [0]*(token_span - len(self.wp_ids_right ))
		self.wp_ids_mention_padding = self.wp_ids_mention[:mention_span] + [0]*(mention_span - len(self.wp_ids_mention ))
		self.wp_ids_all_padding =  self.wp_ids_left_padding + self.wp_ids_mention_padding + self.wp_ids_right_padding

	def is_valid(self):
		maxlen 	 = cf.MODEL_OPTIONS['context_window']
		maxlen_m = cf.MODEL_OPTIONS['mention_window'] 
		return len(self.ctx_left) == maxlen and len(self.ctx_right) == maxlen and (self.token_mention_end - self.token_mention_start) <= maxlen_m
	def __repr__(self):
		s =  "Tokens:             %s\n" % self.tokens	
		s += "Token mention start/end:  %s-%s\n" % (self.token_mention_start, self.token_mention_end)
		s += "=========================\n"
		s += "Mentioned entity Left context:       %s\n" % ([self.tokens[t] for t in self.ctx_left])
		s += "Mentioned entity Right context:      %s\n" % ([self.tokens[t] for t in self.ctx_right])
		s += "Mentioned entity:    %s\n" % ([self.tokens[t] for t in self.ctx_mention])
		s += "=========================\n"
		s += "Token map:  %s\n" % (self.token_idxs_to_wp_idxs)
		s += "Wordpieces:         %s\n" % self.wordpieces 
		s += "Wordpieces mention start/end:  %s-%s\n" % (self.wp_mention_start, self.wp_mention_end)
		s += "=========================\n"
		s += "Mentioned entity Left wordpieces: %s\n" % ([self.wordpieces[t] for t in self.wp_ids_left])
		s += "Mentioned entity Right wordpieces: %s\n" % ([self.wordpieces[t] for t in self.wp_ids_right])
		s += "Mentioned entity wordpieces:    %s\n" % ([self.wordpieces[t] for t in self.wp_ids_mention])
		s += "=========================\n\n\n"
				
		return s

def build_hierarchy(filepaths):
	logger.info("Building category hierarchy.")
	hierarchy = CategoryHierarchy()
	for ds_name, filepath in filepaths.items():
		with jsonlines.open(filepath, "r") as reader:
			for i, line in enumerate(reader):
				for m in line['mentions']:
					labels = set(m['labels'])
					for l in labels:
						hierarchy.add_category(l, "test" if ds_name == "dev" else ds_name) # Treat the dev set as the test set for the purpose of the hierarchy categories
				if type(cf.MAX_SENTS[ds_name]) == int and i >= cf.MAX_SENTS[ds_name]:
					break
	hierarchy.freeze_categories() 
	logger.info("Category hierarchy contains %d categories." % len(hierarchy))

	return hierarchy

def build_dataset(filepath, hierarchy, ds_name, contextualizer, tokenizer):

	total_sents = 0
	total_mentions = 0
	total_wordpieces = 0

	sentences = []	
	mentions = []
	unknown_words = []

	with jsonlines.open(filepath, "r") as reader:
		for i, line in enumerate(reader):
			tokens = line['tokens']
			total_sents += 1

			wordpiece_ids_without_special_symbols, wordpiece_ids, token_idxs_to_wp_idxs = contextualizer.tokenize_with_mapping(tokens)	
			wordpieces = tokenizer.convert_ids_to_tokens(wordpiece_ids)
			token_ids = tokenizer.convert_tokens_to_ids(tokens)	

			for i, item in enumerate(token_ids):
				if item == 100 and tokens[i] not in unknown_words:
					unknown_words.append(tokens[i])
			for m in line['mentions']:								
				start = m['start']
				end = m['end']
				if start == -1 and end == -1:
					continue
				labels = hierarchy.categories2onehot(m['labels'])
				mention = Mention(tokens, 
								  token_ids,
								  labels,								   
								  wordpieces,  
								  wordpiece_ids, 
								  token_idxs_to_wp_idxs, 
								  start, 
								  end)
				mentions.append(mention)
				total_mentions += 1
				total_wordpieces += len(wordpiece_ids)

			print("\r%s" % total_sents, end="")
			if type(cf.MAX_SENTS[ds_name]) == int and len(mentions) >= cf.MAX_SENTS[ds_name]:
				break
	
	print("saving unknown words ... ")
	with open(here/"unknown_words.txt", "w") as out:
		for line in unknown_words:
			out.write(line)
			out.write('\n' )

	logger.info("Building data loader...")

	data_xl, data_xr, data_xa, data_xm, data_y = [], [], [], [], []

	for i, mention in enumerate(mentions):
		data_xl.append(np.asarray(mention.wp_ids_left_padding))
		data_xr.append(np.asarray(mention.wp_ids_right_padding))
		data_xa.append(np.asarray(mention.wp_ids_all_padding))
		data_xm.append(np.asarray(mention.wp_ids_mention_padding))
		data_y.append(np.asarray(mention.labels))		
	dataset = MentionTypingDataset(data_xl, data_xr, data_xa, data_xm, data_y)

	return dataset, total_wordpieces



def main():

	dataset_filenames = {
		"train": cf.TRAIN_FILENAME,
		"dev": cf.DEV_FILENAME,
		"test": cf.TEST_FILENAME,
	}
	# 1. Construct the Hierarchy by looking through each dataset for unique labels.
	hierarchy = build_hierarchy(dataset_filenames)
	logger.info("Hierarchy contains %d categories unique to the test set." % len(hierarchy.get_categories_unique_to_test_dataset()))
	
	# 2. Build a data loader for each dataset (train, test).
	# A 'data loader' is an Pytorch object that stores a dataset in a numeric format.
	contextualizer = get_contextualizer("bert-base-cased", device = device)
	tokenizer = contextualizer.get_tokenizer()

	# vocab = tokenizer.save_vocabulary(save_directory= "/home/ziyu/Desktop/CODE/ET-DeepSDD")

	data_loaders = {}
	for ds_name, filepath in dataset_filenames.items():
		logger.info("Loading %s dataset from %s." % (ds_name, filepath))
		dataset, total_wordpieces = build_dataset(filepath, hierarchy, ds_name, contextualizer,tokenizer)
		data_loader = DataLoader(dataset, batch_size=cf.BATCH_SIZE, pin_memory=True)
		data_loaders[ds_name] = data_loader
		logger.info("The %s dataset was built successfully." % ds_name)

		logger.info("Dataset contains %i wordpieces (including overly long sentences)." % total_wordpieces)
		if ds_name == "train":
			total_wordpieces_train = total_wordpieces

	print(hierarchy.category_counts['train'])

	BYPASS_SAVING = False
	if BYPASS_SAVING:
		logger.info("Bypassing file saving - training model directly")
		train_without_loading(data_loaders, hierarchy, total_wordpieces_train)

		logger.info("Evaluating directly")
		evaluate_without_loading(data_loaders, hierarchy, total_wordpieces_train)
		return

	logger.info("Saving data loaders to file...")
	dutils.save_obj_to_pkl_file(data_loaders, 'data loaders', cf.ASSET_FOLDER + '/data_loaders.pkl')

	logger.info("Savinghierarchy to file...")
	dutils.save_obj_to_pkl_file(hierarchy, 'hierarchy', cf.ASSET_FOLDER + '/hierarchy.pkl')
	dutils.save_obj_to_pkl_file(total_wordpieces_train, 'total_wordpieces', cf.ASSET_FOLDER + '/total_wordpieces.pkl')


if __name__ == "__main__":
	main()
