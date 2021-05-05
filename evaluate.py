import torch
import json, sys, random, re, matplotlib
from sklearn.metrics import f1_score, classification_report, accuracy_score
from colorama import Fore, Back, Style

from logger import logger
import nfgec_evaluate 
import data_utils as dutils
from data_utils import batch_to_wordpieces, build_token_to_wp_mapping, load_embeddings
from load_config import load_config, device
cf = load_config()


from pathlib import Path
here = Path(__file__).parent

from torch.utils.tensorboard import SummaryWriter
writer = writer = SummaryWriter('runs/ontonotes_modified/sdd_KLD_coe0.005')


torch.manual_seed(123)
torch.backends.cudnn.deterministic=True

class ModelEvaluator():

	def __init__(self, model, data_loader, word_vocab, wordpiece_vocab, hierarchy, bc, mode="train"):
		self.model = model 
		self.data_loader = data_loader 
		self.word_vocab = word_vocab 
		self.wordpiece_vocab = wordpiece_vocab
		self.hierarchy = hierarchy
		self.bc = bc
		self.mode = mode
		self.best_f1_and_epoch = [0.0, -1]


	# Evaluate a given model via F1 score over the entire test corpus.
	def evaluate_model(self, epoch):		
		if cf.EMBEDDING_MODEL == "bert":
			bc =  get_contextualizer("bert-base-cased", device='cuda:0')
		else:
			bc = None

		self.model.zero_grad()
		self.model.eval()	
		all_tys   = None
		all_preds = None		
		true_and_prediction = []
		accuracy_scores = []
		average_scores = []
		num_batches = len(self.data_loader)
		labels_set = []

		# Convert all one-hot to categories
		def build_true_and_preds(self, tys, preds):
			true_and_prediction = []
			empty = 0
			for i, row in enumerate(tys):	
				true_cats = self.hierarchy.onehot2categories(tys[i])		
				pred_cats = self.hierarchy.onehot2categories(preds[i])
				true_and_prediction.append((true_cats,pred_cats))
				if pred_cats == []:
					empty += 1
			if empty > 0:
				logger.warn("There were %d empty predictions." % empty)
			return true_and_prediction	
	
		
		for (i, (batch_xl, batch_xr, batch_xa, batch_xm, batch_y)) in enumerate(self.data_loader):
			
			batch_true_and_predictions = []
			entities = []
			
			
			# Convert the batch_x from wordpiece ids into pretrained bert embedding
			bert_embs_l = self.bc.encode(batch_xl, frozen=True)	
			bert_embs_r = self.bc.encode(batch_xr, frozen=True)		
			bert_embs_m = self.bc.encode(batch_xm, frozen=True)			
							
			mention_preds = self.model.evaluate(bert_embs_l, bert_embs_r, None, bert_embs_m)
			batch_y = batch_y.float().to(device)

			for j, row in enumerate(batch_y):				

				labels = self.hierarchy.onehot2categories(batch_y[j])
				preds = self.hierarchy.onehot2categories(mention_preds[j])	

				if labels not in labels_set:
					labels_set.append(labels)
				true_and_prediction.append((labels, preds))
				batch_true_and_predictions.append((labels, preds))
				every_entity_mention = ' '.join(wordpieces_m[j])
				every_entity_mention = re.sub("'", "", every_entity_mention)
				entities.append(every_entity_mention)


			s = ""		
			for i, every_tuple in enumerate(batch_true_and_predictions):
				s += "Entity Mention:"
				s += " ".join(entities[i])	
				s += "\n"				
				s += "Predicted: "
				ps = ", ".join(["%s%s%s" % (Fore.GREEN if pred in every_tuple[0] else Fore.RED, pred, Style.RESET_ALL) for pred in every_tuple[1]])							
				s += ps
				s += "\n"	
				s += "Actual: "							
				s += ", ".join(every_tuple[0])
				s += "\n\n"

				for pred in every_tuple[1]:
					if pred in every_tuple[0]:
						with open(here / "predictions" /self.model.dataset/ "(partial)predicted", "a") as out:
							print(entities[i], file=out)
							print(every_tuple, file=out)
					else:
						with open(here / "predictions" /self.model.dataset/"unpredicted", "a") as out:
							print(entities[i], file=out)
							print(every_tuple, file=out)

			logger.info("\n" + s)	
				
			sys.stdout.write("\rEvaluating batch %d / %d" % (i, num_batches))

			with open(here / "predictions" /self.model.dataset/"labels_set", "w") as out:
				print(len(labels_set), file = out)
				print(labels_set, file=out)
		

		

		print("")
		print(len(true_and_prediction))
		micro, macro, acc = nfgec_evaluate.loose_micro(true_and_prediction)[2], nfgec_evaluate.loose_macro(true_and_prediction)[2], nfgec_evaluate.strict(true_and_prediction)[2]
		logger.info("                  Micro F1: %.4f\tMacro F1: %.4f\tAcc: %.4f" % (micro, macro, acc))
		writer.add_scalar("Accuracy/train", acc, epoch)
		accuracy_scores.append(acc)
		average_scores.append((acc + macro + micro) / 3)
		return (acc + macro + micro) / 3

	# Save the best model to the best model directory, and save a small json file with some details (epoch, f1 score).
	def save_best_model(self, f1_score, epoch):
		logger.info("Saving model to %s." % cf.BEST_MODEL_FILENAME)
		torch.save(self.model.state_dict(), cf.BEST_MODEL_FILENAME)

		logger.info("Saving model details to %s." % cf.BEST_MODEL_JSON_FILENAME)
		model_details = {
			"epoch": epoch,
			"f1_score": f1_score
		}
		with open(cf.BEST_MODEL_JSON_FILENAME, 'w') as f:
			json.dump(model_details, f)

	# Determine whether the given f1 is better than the best f1 score so far.
	def is_new_best_f1_score(self, f1):
		return f1 > self.best_f1_and_epoch[0]

	# Determine whether there has been no improvement to f1 over the past n epochs.
	def no_improvement_in_n_epochs(self, n, epoch):
		return epoch - self.best_f1_and_epoch[1] >= n

	# Evaluate the model every n epochs.
	def evaluate_every_n_epochs(self, n, epoch):		
		if epoch % n == 0 or epoch == cf.MAX_EPOCHS:
			f1 = self.evaluate_model(epoch)

			if self.is_new_best_f1_score(f1):
				self.best_f1_and_epoch = [f1, epoch]
				logger.info("New best average F1 score achieved!        (%s%.4f%s)" % (Fore.YELLOW, f1, Style.RESET_ALL))
				self.save_best_model(f1, epoch)
			elif epoch > 15 and self.no_improvement_in_n_epochs(cf.STOP_CONDITION, epoch):#:cf.STOP_CONDITION):
				logger.info("No improvement to F1 score in past %d epochs. Stopping early." % cf.STOP_CONDITION)
				logger.info("Best F1 Score: %.4f" % self.best_f1_and_epoch[0])

				main()
				exit()
			if epoch == cf.MAX_EPOCHS:
				logger.info("Training complete.")
				main()
				exit()
				
def create_model(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces):

	from model import  MentionLevelModel
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

def evaluate_without_loading(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces):
	from model import E2EETModel, MentionLevelModel
	from bert_encoder import  get_contextualizer
	import jsonlines
	
	bc = get_contextualizer("bert-base-cased", device='cuda:0')
	logger.info("Loading files...")	
	logger.info("Building model.")
	model = create_model(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces)	
	model.cuda()
	model.load_state_dict(torch.load(cf.BEST_MODEL_FILENAME))
	modelEvaluator = ModelEvaluator(model, data_loaders['test'], word_vocab, wordpiece_vocab, hierarchy, bc, mode="test")	
	with jsonlines.open(cf.BEST_MODEL_JSON_FILENAME, "r") as reader:
		for line in reader:
			f1_score, epoch = line['f1_score'], line['epoch']
	modelEvaluator.evaluate_model(epoch)	


def main():

	from model import  MentionLevelModel
	from bert_encoder import  get_contextualizer
	import jsonlines	
	bc = get_contextualizer("bert-base-cased", device='cuda:0')	

	logger.info("Loading files...")
	data_loaders = dutils.load_obj_from_pkl_file('data loaders', cf.ASSET_FOLDER + '/data_loaders.pkl')
	word_vocab = dutils.load_obj_from_pkl_file('word vocab', cf.ASSET_FOLDER + '/word_vocab.pkl')
	wordpiece_vocab = dutils.load_obj_from_pkl_file('wordpiece vocab', cf.ASSET_FOLDER + '/wordpiece_vocab.pkl')
	hierarchy = dutils.load_obj_from_pkl_file('hierarchy', cf.ASSET_FOLDER + '/hierarchy.pkl')
	total_wordpieces = dutils.load_obj_from_pkl_file('total wordpieces', cf.ASSET_FOLDER + '/total_wordpieces.pkl')	
	
	logger.info("Building model.")
	model = create_model(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces)	
	model.cuda()
	model.load_state_dict(torch.load(cf.BEST_MODEL_FILENAME))


	modelEvaluator = ModelEvaluator(model, data_loaders['test'], word_vocab, wordpiece_vocab, hierarchy, bc, mode="test")	
	with jsonlines.open(cf.BEST_MODEL_JSON_FILENAME, "r") as reader:
		for line in reader:
			f1_score, epoch = line['f1_score'], line['epoch']
			# writer.add_scalar("F1",f1_score , epoch)

	modelEvaluator.evaluate_model(epoch)	
	writer.flush()	

if __name__ == "__main__":
	main()
		
