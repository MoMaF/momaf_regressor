import transformers
import torch
from torch import nn

class RegressorModelOutput(transformers.file_utils.ModelOutput):

    def __init__(self,prediction,loss=None):
        super().__init__()
        self["prediction"]=prediction
        if loss is not None:
            self["loss"]=loss

class BertRegressor(transformers.BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = transformers.BertModel(config)
        #The output layer that takes the [CLS] representation and gives an output
        self.cls_layer1 = nn.Linear(config.hidden_size,128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128,128)
        self.tanh1 = nn.Tanh()
        self.ff2 = nn.Linear(128,1)

    def forward(self, input_ids, attention_mask,target=None):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        output = self.tanh1(output)
        output = self.ff2(output)
        if target is not None:
            mo=RegressorModelOutput(output, nn.MSELoss()(torch.squeeze(target,-1), torch.squeeze(output,-1)))
        else:
            mo=RegressorModelOutput(output)
        return mo
        
