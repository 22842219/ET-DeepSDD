
import time, json, os
import torch
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torch.optim.adamw import AdamW

from progress_bar import ProgressBar
import data_utils as dutils
from data_utils import Vocab, CategoryHierarchy, EntityTypingDataset, batch_to_wordpieces
from logger import logger
from model import MentionLevelModel
from bert_encoder import  get_contextualizer
from evaluate import ModelEvaluator
from load_config import load_config, device
cf = load_config()


from pathlib import Path
here = Path(__file__).parent

from torch.utils.tensorboard import SummaryWriter
writer = writer = SummaryWriter('runs/ontonotes_modified/bilstm_emb300')

torch.manual_seed(123)
torch.backends.cudnn.deterministic=True


# Train the model, evaluating it every 10 epochs.
def train(model, data_loaders, word_vocab, wordpiece_vocab, hierarchy, epoch_start = 1):
	logger.info("Training model.")

	# Set up a pre_trained bert model, for encoding the wordpieces
	if cf.EMBEDDING_MODEL == "bert":
		bc =  get_contextualizer("bert-base-cased", device='cuda:0')
	else:
		bc = None

	modelEvaluator = ModelEvaluator(model, data_loaders['dev'], word_vocab, wordpiece_vocab, hierarchy, bc)
	
	optimizer = Optimizer.AdamW(
        params=model.parameters(),
        lr=cf.LEARNING_RATE,
        weight_decay=0.1
    )
    
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
			# # 1. Convert the batch_x from wordpiece ids into bert embedding vectors

			bert_embs_l = bc.encode(batch_xl, frozen=True)					
			bert_embs_r = bc.encode(batch_xr, frozen=True)		
			bert_embs_m = bc.encode(batch_xm, frozen=True)		
			
			batch_y = batch_y.float().to(device)	

			# 2. Feed these Bert vectors to our model
			model.zero_grad()
			model.train()

			y_hat = model(bert_embs_l, bert_embs_r, None, bert_embs_m)

			loss = model.calculate_loss(y_hat, batch_y)

			# 3. Backpropagate
			loss.backward()
			optimizer.step()
			epoch_losses.append(loss)

			# 4. Draw the progress bar
			progress_bar.draw_bar(i, epoch, epoch_start_time)
			

		avg_loss = sum(epoch_losses) / float(len(epoch_losses))
		avg_loss_list.append(avg_loss)
		progress_bar.draw_completed_epoch(avg_loss, avg_loss_list, epoch, epoch_start_time)	
		avg_loss_png.append((epoch, avg_loss))
		modelEvaluator.evaluate_every_n_epochs(1, epoch)


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

	folder ='{}/{}/{}/'.format(here, "predictions", model.dataset)
	if not os.path.exists(os.path.dirname(folder)):
		try:
			os.makedirs(os.path.dirname(folder))
		except OSError as exc:
			if exc.errno != errno.EEXITST:
				raise

	train(model, data_loaders, word_vocab, wordpiece_vocab, hierarchy)
	writer.flush()

		

if __name__ == "__main__":
	main()
