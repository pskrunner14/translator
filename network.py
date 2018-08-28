import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 15

# TODO (1) Implement batch encodings and decodings

"""
Encoder RNN Model
"""
class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderRNN, self).__init__()
        
        # Model Config
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding Layer
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        
        # LSTM Layers
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, 
                            self.num_layers, bidirectional=True)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq).view(1, 1, -1)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        forward_output, backward_output = \
            output[:, :, :self.hidden_size], output[:, :, self.hidden_size:]
        output = torch.cat((forward_output, backward_output))
        return output, hidden
    
    def init_hidden(self):
        return (torch.zeros(self.num_layers * 2, 1, 
                self.hidden_size, device=torch.device('cuda')),
                torch.zeros(self.num_layers * 2, 1, 
                self.hidden_size, device=torch.device('cuda')))

"""
Attention Decoder RNN Model
"""
class AttnDecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, output_size, num_layers=1, 
                dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        
        # Model Config
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers
        
        # Embedding Layer
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
        # Attention Mechanism
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_p)
        
        # LSTM Layers
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, 
                            self.num_layers, bidirectional=True)
        # Output Dense    
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, input_seq, hidden, encoder_outputs):
        embedded = self.embedding(input_seq).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax( self.attn(
            torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), 
                                 encoder_outputs.unsqueeze(0))
        
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        forward_output, backward_output = \
            output[:, :, :self.hidden_size], output[:, :, self.hidden_size:]
        output = torch.cat((forward_output, backward_output))
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
    
    def init_hidden(self):
        return (torch.zeros(self.num_layers * 2, 1, 
                self.hidden_size, device=torch.device('cuda')),
                torch.zeros(self.num_layers * 2, 1, 
                self.hidden_size, device=torch.device('cuda')))