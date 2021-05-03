import sys, os
import torch
import json
import datasets
from datasets import load_dataset
from pathlib import Path
from bert_encoder import BertEncoder

here = Path(__file__).parent
torch.cuda.empty_cache()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')		


def main(argv):
	"""
	Creates dataset. 
	"""
	args = _argparser().parse_args(argv[1:])
	dataset = args.dataset
	# batch_size = args.batch_size
	encoded_layers = BertEncoder()

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
			bert_embedding = encoded_layers(datasets[dataset][:10])
			print("bert_embedding:", bert_embedding.size())			
			y_hat = torch.sigmoid(bert_embedding)
			print("normalized y_hat:", y_hat)
	return y_hat



def _argparser():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', '-d', type=str, help='dataset.')
	# parser.add_argument('--batch_size', '-d', type=str, help='batch_size.')
	return parser


if __name__ == '__main__':
    main(sys.argv)



