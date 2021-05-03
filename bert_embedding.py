import sys, os
import torch
import json
import datasets
from datasets import load_dataset
from transformers import BertTokenizer, BertForTokenClassification, BertModel, BertConfig
from pathlib import Path
here = Path(__file__).parent
		

def main(argv):
	"""
	Creates dataset. 
	"""
	args = _argparser().parse_args(argv[1:])
	dataset = args.dataset
	tokens = []
	mentions = []
	labels=[]

	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
	bert_model = BertModel.from_pretrained('bert-base-cased')
	bert_model.to('cuda')

	#Loading data

	_URL ='{}/{}/{}/'.format(here, "data", dataset)
	_URLs = {
	"train": _URL + "train.json",
	"dev": _URL + "dev.json",
	"test": _URL + "test.json",
	}
	datasets = load_dataset('json', data_files= _URLs)
	for dataset in datasets:
		if dataset == 'train':
			for example in datasets[dataset]:				
				tokenized_input = tokenizer(example["tokens"], is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
				tokenized_input = tokenized_input.to('cuda')
				outputs = bert_model(**tokenized_input)
				#The first element is the hidden state of the last layer of the Bert model
				encoded_layers = outputs[0]				
				y_hat = torch.sigmoid(encoded_layers)
	return y_hat



def _argparser():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', '-d', type=str, help='dataset.')

	return parser


if __name__ == '__main__':
    main(sys.argv)



