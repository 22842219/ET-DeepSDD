import data_utils as dutils
from data_utils import Vocab, CategoryHierarchy, EntityTypingDataset, batch_to_wordpieces, wordpieces_to_bert_embs, load_embeddings
from bert_serving.client import BertClient
from logger import logger
from model import MentionLevelModel
import torch.optim as optim
from progress_bar import ProgressBar
import time, json
import torch
from torch.autograd import Variable
from load_config import load_config, device
cf = load_config()
from evaluate import ModelEvaluator
import pandas as pd
import os
from pathlib import Path
here = Path(__file__).parent


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/ontonotes_modified_sdd_March_KLD_coe0.5')


torch.manual_seed(123)
torch.backends.cudnn.deterministic=True


# Train the model, evaluating it every 10 epochs.
def train(model, data_loaders, word_vocab, wordpiece_vocab, hierarchy, epoch_start = 1):


	logger.info("Training model.")

	# Set up a new Bert Client, for encoding the wordpieces
	if cf.EMBEDDING_MODEL == "bert":
		bc = BertClient()
	else:
		bc = None

	modelEvaluator = ModelEvaluator(model, data_loaders['dev'], word_vocab, wordpiece_vocab, hierarchy, bc)
	
	#optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.LEARNING_RATE, momentum=0.9)
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.LEARNING_RATE)#, eps=1e-4, amsgrad=True)#, momentum=0.9)
	model.cuda()


	num_batches = len(data_loaders["train"])
	progress_bar = ProgressBar(num_batches = num_batches, max_epochs = cf.MAX_EPOCHS, logger = logger)
	avg_loss_list = []

	epoch_losses = []
	avg_loss_png= []
	# Train the model
	for epoch in range(epoch_start, cf.MAX_EPOCHS + 1):
		epoch_start_time = time.time()


		for (i, (batch_xl, batch_xr, batch_xa, batch_xm, batch_y)) in enumerate(data_loaders["train"]):
			#torch.cuda.empty_cache()
			#if i > 1:
			#	continue
			# 1. Convert the batch_x from wordpiece ids into wordpieces
			wordpieces_l = batch_to_wordpieces(batch_xl, wordpiece_vocab)
			wordpieces_r = batch_to_wordpieces(batch_xr, wordpiece_vocab)
			#wordpieces_a = batch_to_wordpieces(batch_xa, wordpiece_vocab)
			wordpieces_m = batch_to_wordpieces(batch_xm, wordpiece_vocab)

			# 2. Encode the wordpieces into Bert vectors
			bert_embs_l  = wordpieces_to_bert_embs(wordpieces_l, bc).to(device)
			bert_embs_r  = wordpieces_to_bert_embs(wordpieces_r, bc).to(device)				
			#bert_embs_a  = wordpieces_to_bert_embs(wordpieces_a, bc).to(device)
			bert_embs_m  = wordpieces_to_bert_embs(wordpieces_m, bc).to(device)

			batch_y = batch_y.float().to(device)

			# 3. Feed these Bert vectors to our model
			model.zero_grad()
			model.train()

			y_hat = model(bert_embs_l, bert_embs_r, None, bert_embs_m)	

			loss = model.calculate_loss(y_hat, batch_y)
			writer.add_scalar("Loss/train", loss, epoch)

			
			for j, row in enumerate(batch_y):
				labels = hierarchy.onehot2categories(batch_y[j])
				with open(here / "outputs" / "batch_y", "a") as out:
					print(labels, file=out)
					print(batch_y[j], file=out)


			# 4. Backpropagate
			# loss.backward() computes dloss/dx for every parameter x which has requires_grad = True. x.grad += dloss/dx
			# optimizer.step updates the value of x using the gradient x.grad.			
			loss.backward()
			optimizer.step()
			epoch_losses.append(loss)


			# epoch_semantic_losses.append(semantic_loss)

			# semantic_loss.backward()
			# optimizer_.step()

			# 5. Draw the progress bar
			progress_bar.draw_bar(i, epoch, epoch_start_time)	

		avg_loss = sum(epoch_losses) / float(len(epoch_losses))
		avg_loss_list.append(avg_loss)
		progress_bar.draw_completed_epoch(avg_loss, avg_loss_list, epoch, epoch_start_time)	
		avg_loss_png.append((epoch, avg_loss))
		modelEvaluator.evaluate_every_n_epochs(1, epoch)

	# import sys
	# import matplotlib
	# matplotlib.use('Agg')
	# import matplotlib.pyplot as plt

	# plt.figure(figsize=(10,5))
	# plt.title("Average Loss During Training")
	# plt.plot(avg_loss_png,label="avg_loss/Train")
	# plt.xlabel("epoch")
	# plt.ylabel("avg_loss")
	# plt.legend()
	# # plt.show()

	# # plt.plot('epoch', 'avg_loss', data=avg_loss_png)
	# plt.savefig(sys.stdout.buffer)


def create_model(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces):
	model = MentionLevelModel(	embedding_dim = cf.EMBEDDING_DIM,
						hidden_dim = cf.HIDDEN_DIM,

						vocab_size = len(wordpiece_vocab),
						label_size = len(hierarchy),
						dataset = cf.DATASET,
						model_options = cf.MODEL_OPTIONS,
						total_wordpieces = total_wordpieces,
						category_counts = hierarchy.get_train_category_counts(),
						context_window = cf.MODEL_OPTIONS['context_window'],
						attention_type = cf.MODEL_OPTIONS['attention_type'],
						mention_window = cf.MODEL_OPTIONS['mention_window'],
						use_context_encoders = cf.MODEL_OPTIONS['use_context_encoders'],
						hierarchy_matrix = hierarchy.hierarchy_matrix)

	return model


def train_without_loading(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces):
	model = create_model(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces)
	model.cuda()
	train(model, data_loaders, word_vocab, wordpiece_vocab, hierarchy)


def main():		

	logger.info("Loading files...")

	data_loaders = dutils.load_obj_from_pkl_file('data loaders', cf.ASSET_FOLDER + '/data_loaders.pkl')
	word_vocab = dutils.load_obj_from_pkl_file('word vocab', cf.ASSET_FOLDER + '/word_vocab.pkl')
	wordpiece_vocab = dutils.load_obj_from_pkl_file('wordpiece vocab', cf.ASSET_FOLDER + '/wordpiece_vocab.pkl')
	total_wordpieces = dutils.load_obj_from_pkl_file('total wordpieces', cf.ASSET_FOLDER + '/total_wordpieces.pkl')
	hierarchy = dutils.load_obj_from_pkl_file('hierarchy', cf.ASSET_FOLDER + '/hierarchy.pkl')
	
	logger.info("Building model.")
	model = create_model(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces)		
	model.cuda()
	train(model, data_loaders, word_vocab, wordpiece_vocab, hierarchy)
	writer.flush()

		

if __name__ == "__main__":
	main()
