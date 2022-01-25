import transformers
import torch
from torch import nn

class RegressorModelOutput(transformers.file_utils.ModelOutput):

    def __init__(self,prediction,loss=None):
        super().__init__()
        #print("PREDICTION",prediction.squeeze())
        self["prediction"]=prediction.squeeze()
        if loss is not None:
            self["loss"]=loss

class BertRegressor(transformers.BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = transformers.BertModel(config)
        #The output layer that takes the mean-pooled representation and gives an output
        self.ff3=nn.Linear(config.hidden_size,1)

    def forward(self, input_ids, attention_mask,target=None):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        #logits = outputs.last_hidden_state[:,0,:]
        ### Average all outputs that are not masked by attention_mask (includes CLS, does it really matter...?)
        h=outputs.last_hidden_state*attention_mask.unsqueeze(-1) #zeros out outputs of masked tokens
        h=h.sum(-2) #sum up along the token dimension
        h=h/attention_mask.sum(-1).unsqueeze(-1) #and divide by attmask sums to get average
        logits = h
        output = self.ff3(logits)
        if target is not None: #Have a target? Calculate loss too!
            mo=RegressorModelOutput(output, nn.MSELoss(reduction="mean")(torch.squeeze(target,-1), torch.squeeze(output,-1)))
        else:
            mo=RegressorModelOutput(output) #No target, get just the output
        return mo
        
