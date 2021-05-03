import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BertEncoder(nn.Module):
    def __init__(self, language='en', use_finetune=False):

        super(BertEncoder, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained('bert-base-cased').to(device)
        
        self.use_finetune = use_finetune
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def forward(self, dataset):
        """
        dataset: The datasets object itself is DatasetDict, which contains one key for the training, validation and test set.
        tokenized_input ={input_ids, token_type_ids, attention_mask}    
        """
        tokenized_input = self.tokenizer(dataset["tokens"],
                                         is_split_into_words=True, 
                                         padding=True, 
                                         truncation=True, 
                                         max_length=100, 
                                         return_tensors="pt").to(device)
        #The first element is the hidden state of the last layer of the Bert model
        if self.use_finetune:
            encoded_layers = self.model()[0]
        else:
            with torch.no_grad():
                encoded_layers = self.model(**tokenized_input)[0]  

        return encoded_layers