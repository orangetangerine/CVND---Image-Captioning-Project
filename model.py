import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)
        
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (torch.zeros(self.n_layers, 1, self.hidden_size).cuda(),
                torch.zeros(self.n_layers, 1, self.hidden_size).cuda())
    
    def forward(self, features, captions):
        # _, self.hidden = self.lstm(features.view(len(features), 1, -1))
        
        embeds = self.embedding(captions)
        # print(embeds[:, :-1, :].shape)
        # print(embeds.shape)
        # print(features.unsqueeze(1).shape)
        
        input_ = torch.cat((features.unsqueeze(1), embeds[:, :-1, :]), 1)
        # print(input_.shape)

        lstm_out, self.hidden = self.lstm(input_)
        outputs = self.hidden2tag(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = torch.zeros(max_len, dtype=torch.int32) # prevent calling append
        # print('inputs.shape before loop', inputs.shape)
        for i in range(max_len):
            output, states = self.lstm(inputs, states)
            output = self.hidden2tag(output) # (1, 1, vocab_size)
            # print('hidden2tag shape', output.shape)
            # print(output.max(2)) # value, index
            idx = output.max(2)[1].squeeze() # index of word with max probability in vocab_size, known as word_id
            res[i] = idx.item()
            
            inputs = self.embedding(idx) # id to embedding
            inputs = inputs.unsqueeze(0).unsqueeze(0) # (1, 1, vocab_id)
            # print('inputs.shape in loop', inputs.shape)
        return res.numpy().tolist()