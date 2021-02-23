import math, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class TRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, prefix='R1'):
        super(TRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.prefix = prefix
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        try: 
            self.dh = int(re.search(r'(?<=Dh)\d+', self.prefix).group(0))
        except:
            self.dh = 1

        try: 
            self.r = list(map(int,re.search(r'(?<=R)[\d,]+', self.prefix).group(0).split(',')))
        except:
            self.r = [1]

        self.order = len(self.r)+1
        self.outer_size = [1+self.hidden_size*self.dh]*self.order
        self.cores = []
        for i,j,k in zip([1]+self.r, self.outer_size, self.r+[self.hidden_size]):
            w = nn.Parameter(torch.Tensor(i, j, k), requires_grad=True)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5)) 
            self.cores.append(w)

        self.x_h_w = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=True)
        self.h_o_w = nn.Parameter(torch.Tensor(hidden_size, output_size), requires_grad=True)
        self.x_o_w = nn.Parameter(torch.Tensor(input_size, output_size), requires_grad=True)
        self.h_o_b  = nn.Parameter(torch.Tensor(1, output_size), requires_grad=True)
        self.x_o_b  = nn.Parameter(torch.Tensor(1, output_size), requires_grad=True)
        self.x_h_b = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        stdv = 1.0 / math.sqrt(hidden_size)
        nn.init.uniform_(self.x_h_w, -stdv, stdv)
        nn.init.uniform_(self.h_o_w, -stdv, stdv)
        nn.init.uniform_(self.x_o_w, -stdv, stdv)
        nn.init.zeros_(self.x_h_b)
        nn.init.zeros_(self.h_o_b)
        nn.init.zeros_(self.x_o_b)
        
        if 'A' in self.prefix:
            act = re.search(r'(?<=A)[a-z]+', self.prefix).group(0)
            if act == 'tanh': self.activation = nn.Tanh()
            elif act == 'relu': self.activation = nn.ReLU()
            elif act == 'sigmoid': self.activation = nn.Sigmoid()
            else: raise NotImplementedError()
        else:
            self.activation = nn.Identity()   

    def forward(self, inputs, hidden_state=None):
        # inputs [ 1 batch, hidden features]
        batch = inputs.size(0)
        assert batch == 1, 'No implemention for mini-batch.'

        if hidden_state is None:
            hidden_state = []
            init_h = torch.zeros(batch, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
            for _ in range(self.dh):
                hidden_state.append(init_h)
    
        h_0 = torch.cat(hidden_state+[torch.ones(1,self.hidden_size)], dim=1) # [ 1 batch, h ]

        h_1 = torch.matmul(h_0.unsqueeze(-1), h_0)

        if self.order >= 3:
            for _ in range(self.order-2):
                h_1 = torch.matmul(h_1.unsqueeze(-1), h_0)
        h_1 = h_1.unsqueeze(-1)

        for c in self.cores:
            h_1_shape, c_shape = list(h_1.size()), list(c.size())
            h_1_view_t, c_view_t = h_1_shape.pop(-1), c_shape.pop(0)
            h_1_shape[-1]*=h_1_view_t
            c_shape[0]*=c_view_t
            h_1 = torch.matmul(h_1.view(h_1_shape), c.view(c_shape))

        h_1 = h_1 + torch.matmul(inputs, self.x_h_w) + self.x_h_b
            
        h_1 = self.activation(h_1)
        
        z_out = torch.matmul(h_1, self.h_o_w) + self.h_o_b + torch.matmul(inputs, self.x_o_w) + self.x_o_b
        z_out = self.sigmoid(z_out)

        hidden_state.pop(0)
        hidden_state.append(h_1)
        
        return z_out, hidden_state

# ==================== MODEL ====================
class TRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, prefix):
        super(TRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trnn_cell = TRNNCell(input_size, hidden_size, output_size, prefix=prefix)

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.size(0)
        batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps, batch_size, self.output_size, dtype=inputs.dtype, device=inputs.device)
        for times in range(time_steps):
            outputs[times], hidden_state = self.trnn_cell(inputs[times], hidden_state)

        return outputs, hidden_state
