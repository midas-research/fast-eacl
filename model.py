import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=False, bidirectional=False):
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional
    def forward(self, inputs, timestamps, reverse=False):
        b, seq, embed = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)
        h = h.cuda()
        c = c.cuda()
        outputs = []
        hidden_state_h = []
        hidden_state_c = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))  # short term mem
            c_s2 = c_s1 * timestamps[:, s: s + 1].expand_as(c_s1)  # discounted short term mem
            c_l = c - c_s1  # long term mem
            c_adj = c_l + c_s2  # adjusted = long + disc short term mem
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(o)
            hidden_state_c.append(c)
            hidden_state_h.append(h)
        if reverse:
            outputs.reverse()
            hidden_state_c.reverse()
            hidden_state_h.reverse()
        outputs = torch.stack(outputs, 1)
        hidden_state_c = torch.stack(hidden_state_c, 1)
        hidden_state_h = torch.stack(hidden_state_h, 1)
        return outputs, (h, c)

class Attention(torch.nn.Module):
    def __init__(self, in_shape, use_attention=True, maxlen=None):
        super(Attention, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.W1 = torch.nn.Linear(in_shape, in_shape)
            self.W2 = torch.nn.Linear(in_shape, in_shape)
            self.V = torch.nn.Linear(in_shape, 1)
        if maxlen != None:
            self.arange = torch.arange(maxlen)
    def forward(self, full, last, lens=None, dim=1):
        if self.use_attention:
            score = self.V(F.tanh(self.W1(last) + self.W2(full)))
            attention_weights = F.softmax(score, dim=dim)
            context_vector = attention_weights * full
            context_vector = torch.sum(context_vector, dim=dim)
            return context_vector
        else:
            return torch.mean(full, dim=dim)

class FAST(nn.Module):
    def __init__(self):
        super(FAST, self).__init__()
        self.text_lstm = [nn.LSTM(768,64) for _ in range(no_stocks)]
        for i,textlstm in enumerate(self.text_lstm):
            self.add_module('textlstm{}'.format(i), textlstm)
        self.time_lstm = [TimeLSTM(768,64) for _ in range(no_stocks)]
        for i,timelstm in enumerate(self.time_lstm):
            self.add_module('timelstm{}'.format(i), timelstm)
        self.day_lstm = [nn.LSTM(64,64) for _ in range(no_stocks)]
        for i,daylstm in enumerate(self.day_lstm):
            self.add_module('daylstm{}'.format(i), daylstm)
        self.text_attention = [Attention(64,10) for _ in range(no_stocks)]
        for i,textattention in enumerate(self.text_attention):
            self.add_module('textattention{}'.format(i), textattention)
        self.day_attention = [Attention(64,5) for _ in range(no_stocks)]
        for i,dayattention in enumerate(self.day_attention):
            self.add_module('dayattention{}'.format(i), dayattention)
        self.linear_stock = nn.Linear(64,1)
    def forward(self, text_input, time_inputs, no_stocks):
        list_1 = []
        op_size = 64
        for i in range(text_input.size(0)):
            list_2 = []
            len_lookback_window = text_input.size(1)
            num_text = text_input.size(2)
            embb_dims = text_input.size(3)
            for j in range(len_lookback_window):
                y, (temp,_) = self.time_lstm[i](text_input[i,j,:,:].reshape(1,num_text,embb_dims), time_inputs[i,j,:].reshape(1,num_text))
                y = self.text_attention[i](y, temp, num_text)
                list_2.append(y)
            text_vectors = torch.Tensor((1,len_lookback_window,op_size))
            text_vectors = torch.cat(list_2)
            text, (temp2,_) = self.day_lstm[i](text_vectors.reshape(1,len_lookback_window,op_size))            
            text = self.text_attention[i](text, temp2, len_lookback_window)
            list_1.append(text.reshape(1,op_size))
        ft_vec = torch.Tensor((no_stocks,op_size))
        ft_vec = torch.cat(list_1)
        op = F.leaky_relu(self.linear_stock(ft_vec))
        return op