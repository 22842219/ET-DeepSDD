import numpy as np
import random
import time, json, os
import torch
from torch.autograd import Variable
# import torch.optim as optim
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter

from progress_bar import ProgressBar
import data_utils as dutils
from data_utils import CategoryHierarchy, EntityTypingDataset
from logger import logger
from model import MentionLevelModel
from bert_encoder import  get_contextualizer
from evaluate import ModelEvaluator
from load_config import load_config, device
cf = load_config()

from pathlib import Path
here = Path(__file__).parent	

def train(model, data_loaders,  hierarchy, writer, epoch_start = 1):

	logger.info("Training model.")
	modelEvaluator = ModelEvaluator(model, data_loaders['dev'], hierarchy, writer)	
	# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.LEARNING_RATE)  
	optimizer = AdamW(
        params=model.parameters(),
        lr = cf.LEARNING_RATE,
        weight_decay= 0.1)  
	model.cuda()

	num_batches = len(data_loaders["train"])
	progress_bar = ProgressBar(num_batches = num_batches, max_epochs = cf.MAX_EPOCHS, logger = logger)
	avg_loss_list = []

	epoch_losses = []
	avg_loss_png= []

	for epoch in range(epoch_start, cf.MAX_EPOCHS + 1):
		epoch_start_time = time.time()

		for (i, (batch_xl, batch_xr, batch_xa, batch_xm, batch_y)) in enumerate(data_loaders["train"]):	
			
			batch_y = batch_y.float().to(device) 
			# Feed these Bert vectors to our model
			model.zero_grad()
			model.train()
			y_hat = model(batch_xl, batch_xr, batch_xa, batch_xm)
			loss = model.calculate_loss(y_hat, batch_y)

			# Backpropagate
			loss.backward()
			optimizer.step()
			epoch_losses.append(loss)

			# Write loss to tensorboard
			writer.add_scalar("Loss/Train", loss, epoch)

			# Draw the progress bar
			progress_bar.draw_bar(i, epoch, epoch_start_time)			

		avg_loss = sum(epoch_losses) / float(len(epoch_losses))
		avg_loss_list.append(avg_loss)
		progress_bar.draw_completed_epoch(avg_loss, avg_loss_list, epoch, epoch_start_time)	
		avg_loss_png.append((epoch, avg_loss))
		modelEvaluator.evaluate_every_n_epochs(1, epoch)


def create_model(data_loaders,  hierarchy, total_wordpieces):
	model = MentionLevelModel(	embedding_dim = cf.EMBEDDING_DIM,
						hidden_dim = cf.HIDDEN_DIM,						
						label_size = len(hierarchy),
						dataset = cf.DATASET,
						model_options = cf.MODEL_OPTIONS,
						total_wordpieces = total_wordpieces,
						context_window = cf.MODEL_OPTIONS['context_window'],
						attention_type = cf.MODEL_OPTIONS['attention_type'],
						mention_window = cf.MODEL_OPTIONS['mention_window'],
						use_context_encoders = cf.MODEL_OPTIONS['use_context_encoders'],
						hierarchy_matrix = hierarchy.hierarchy_matrix)

	return model


def train_without_loading(data_loaders, hierarchy, total_wordpieces):
	model = create_model(data_loaders, hierarchy, total_wordpieces)
	model.cuda()
	train(model, data_loaders, hierarchy)


def main():
	# Ensure deterministic behavior
	dutils.set_seed()

	logger.info("Loading files...")
	data_loaders = dutils.load_obj_from_pkl_file('data loaders', cf.ASSET_FOLDER + '/data_loaders.pkl')
	total_wordpieces = dutils.load_obj_from_pkl_file('total wordpieces', cf.ASSET_FOLDER + '/total_wordpieces.pkl')
	hierarchy = dutils.load_obj_from_pkl_file('hierarchy', cf.ASSET_FOLDER + '/hierarchy.pkl')
	
	logger.info("Building model.")
	model = create_model(data_loaders, hierarchy, total_wordpieces)		
	model.cuda()

	metric_folder ='{}/{}/{}/{}/'.format(here, 'runs', model.dataset, model.model_name)
	if not os.path.exists(os.path.dirname(metric_folder)):
		try:
			os.makedirs(os.path.dirname(metric_folder))
		except OSError as exc:
			if exc.errno != errno.EEXITST:
				raise
	writer = SummaryWriter(metric_folder)

	train(model, data_loaders, hierarchy, writer)
	writer.flush()


if __name__ == "__main__":
	main()
