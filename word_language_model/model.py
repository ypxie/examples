import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers):
        super(RNNModel, self).__init__()
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.W_emb =  nn.Parameter( (torch.randn(ntoken, ninp)) )

        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=False)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.W_emb.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        #emb = self.encoder(input)

        batchsize, timestep = input.size()[1], input.size()[0]
        vec_input = input.view(-1)
        emb = torch.index_select(self.W_emb, 0, vec_input).view(timestep,batchsize, -1) # emb = N*ninp

        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

def matmul(emb, W):
    # N*T*nin, W: nin*nhid
    W_linear = W.view(1, *W.size())
    W_linear = W_linear.expand(emb.size()[0],*(W_linear.size()[1:]) )
    return torch.bmm(emb, W_linear)


class _RNNModel(object):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers):
        self.__dict__.update(locals())
        initrange = 0.1

        #self.W_emb =  nn.Parameter( (torch.randn(ntoken, ninp)).uniform_(-initrange, initrange) )
        #self.W_emb =  torch.autograd.Variable( (torch.randn(ntoken, ninp)).uniform_(-initrange, initrange) , requires_grad = True)

        self.W_rnn =  nn.Parameter( (torch.randn(ninp, nhid)).uniform_(-initrange, initrange) )
        self.U_rnn =  nn.Parameter( (torch.eye(nhid, nhid)))
        self.b_rnn = nn.Parameter( (torch.zeros(1, nhid)))
        self.W_decode = nn.Parameter( (torch.randn(nhid, ntoken)).uniform_(-initrange, initrange) )
        self.b_decode = nn.Parameter(torch.zeros(1, ntoken))

        self._parameters = [self.W_emb,self.W_rnn, self.U_rnn, self.b_rnn , self.W_decode, self.b_decode]
        self._cuda = False

    def parameters(self):
        return self._parameters

    def cuda(self):
        self._cuda = True
        for param in self._parameters:
            param.grad.data = param.grad.data.cuda()
            param.data = param.data.cuda()

    def zero_grad(self):
        for p in self._parameters:
            p.grad.data.zero_()


    def forward(self, input, state=None):
        # input is timesetep*N
        if self._cuda:
            input = input.cuda()

        batchsize, timestep = input.size()[1], input.size()[0]

        vec_input = input.view(-1)
        emb = torch.index_select(self.W_emb, 0, vec_input).view(timestep,batchsize, -1) # emb = N*ninp
        inp = matmul(emb, self.W_rnn)
        #inp = torch.transpose(input, 0,1) # T * N * nhid
        state = torch.autograd.Variable( torch.zeros(inp.size()[1:])) if state is None else state
        out = nn.Parameter(torch.zeros(timestep, batchsize, self.W_decode.size()[1]) )
        #out = torch.autograd.Variable(torch.zeros(timestep, batchsize, self.W_decode.size()[1]) , requires_grad = True)

        if self._cuda:
            state = state.cuda()
            out = out.cuda()

        for step in range(inp.size()[0]):
            this_input = inp[step] # N * nhid
            this_input = torch.addmm(this_input, state, self.U_rnn)
            state = F.tanh(this_input  + self.b_rnn.expand_as(this_input) )

            out[step] = torch.addmm(out[step], state, self.W_decode)
            out[step] = out[step]  + self.b_decode.expand_as(out[step]) # we don't do softmax here.
        self.out = out

        return out, state

    def init_hidden(self, batch_size):
        hid = torch.autograd.Variable(torch.zeros(batch_size, self.nhid))
        if self._cuda:
            return hid.cuda()
        else:
            return hid


    def __call__(self, input, state = None):
        return self.forward(input, state)













