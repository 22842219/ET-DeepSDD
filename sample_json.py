import sys, os, re
import json
import numpy as np
import math

from pathlib import Path
here = Path(__file__).parent

def main(argv):
	"""
	Creates the data distribution of each sample rate from the raw whole dataset. 
	In order to have the fair enough comparision and the test if our model is sensitive with the volume of dataset or not,
	we use the same validation and test datsets.
	The sample rate considered: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00.
	"""
	args = _argparser().parse_args(argv[1:])
	sample_json = args.sample_json
	# dataset = args.dataset
	datasets = ['figer_50k']
	sample_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	basefile = r"/home/ziyu/Desktop/CODE/phd project1_entity typing/ET-DeepSDD/data/datasets/"
	for dataset in datasets:
		infile = '{}/{}'.format(basefile, dataset)
		for file in os.listdir(infile):
			filepath = '{}/{}'.format(infile,file)
			with open(filepath) as f:
				records = [json.loads(x) for x in f]

			count = len(records)
			print("count:", count)

			for r in sample_ratio:
				folder = '_{}/'.format(r)
				folderpath = infile + folder
				# print("filepath:", filepath)

				if not os.path.exists(os.path.dirname(folderpath)):
					try:
						os.makedirs(os.path.dirname(folderpath))
					except OSError as exc:
						if exc.errno != errno.EEXITST:
							raise

				if file == "train.json":
					sample_count = math.ceil(count * r)
				else:
					sample_count = count
				sample_indexes = np.random.choice(
					count,
					sample_count)

				sample_records = []
				for sample_index in sample_indexes:
					sample_record = records[sample_index]
					sample_records.append(sample_record)

				assert len(sample_records) == sample_count

				with open(here / "data" /"datasets"/ folderpath/file,"a") as f:
					for record in sample_records:
						f.write(json.dumps(record) + "\n")
				print("Sampled {} records from {} original modified records.".format( sample_count, count))

def _argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_json', help="")
    # parser.add_argument('--dataset', '-d', type=str, default=1,
                        # help='dataset.')
    return parser


if __name__ == '__main__':
    main(sys.argv)
