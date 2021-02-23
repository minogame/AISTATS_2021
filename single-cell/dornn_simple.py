import re, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class DORNNCell(nn.Module):
    """
    Prefix:
    C(ombined): (X H)' as input
    F(ixed): The P is fixed, otherwise learnable
    N(eurIPS) + (rank) + (f)eature/(l)ayer: The fractional represent, feature-wise or layer-wise
    H(yper-power): The P is generated using hyper-net
    A(ctivation): Use activation functions for h_0
    O(U)tterProduct: Use outter product on (X H)
    D(uplicated states): There exists multipy inputs/hidden steps.
    """
    def __init__(self, input_size, hidden_size, output_size, prefix='C'):
        super(DORNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.prefix = prefix
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        try: 
            self.r = int(re.search(r'(?<=U)[\d\.]+', self.prefix).group(0))
        except:
            self.r = 2
        try: 
            self.n = int(re.search(r'(?<=N)\d+', self.prefix).group(0))
        except:
            self.n = 2
        try: 
            self.p = float(re.search(r'(?<=F)[\d\.]+', self.prefix).group(0))
        except:
            self.p = 1.0
        try: 
            self.s = re.search(r'(?<=N)\d+[fl]', self.prefix).group(0)
        except:
            self.s = 'f'
        try: 
            self.dx = int(re.search(r'(?<=Dx)\d+', self.prefix).group(0))
        except:
            self.dx = 1
        try: 
            self.dh = int(re.search(r'(?<=Dh)\d+', self.prefix).group(0))
        except:
            self.dh = 1
        try:
            self.hidden_size_p = int(re.search(r'(?<=Dp)\d+', self.prefix).group(0))
        except:
            self.hidden_size_p=5

        if 'f' in self.s:
            self.p_size = self.hidden_size
        elif 'l' in self.s:
            self.p_size = 1

        if 'A' in self.prefix:
            act = re.search(r'(?<=A)[a-z]+', self.prefix).group(0)
            if act == 'tanh': self.activation = nn.Tanh()
            elif act == 'relu': self.activation = nn.ReLU()
            elif act == 'sigmoid': self.activation = nn.Sigmoid()
            else: raise NotImplementedError()
        else:
            self.activation = nn.Identity()   

class DORNNCell_Z(DORNNCell):
    def __init__(self, input_size, hidden_size, output_size, prefix='C'):
        super(DORNNCell_Z, self).__init__(input_size, hidden_size, output_size, prefix)
        stdv = 1.0 / math.sqrt(hidden_size)

        self.x_h_w = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=True)
        self.h_h_w = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
        self.h_o_w = nn.Parameter(torch.Tensor(hidden_size, output_size), requires_grad=True)
        self.x_o_w = nn.Parameter(torch.Tensor(input_size, output_size), requires_grad=True)
        nn.init.uniform_(self.x_h_w, -stdv, stdv)
        nn.init.uniform_(self.h_h_w, -stdv, stdv)
        nn.init.uniform_(self.h_o_w, -stdv, stdv)
        nn.init.uniform_(self.x_o_w, -stdv, stdv)

        self.x_h_b = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.h_h_b = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.h_o_b = nn.Parameter(torch.Tensor(1, output_size), requires_grad=True)
        self.x_o_b = nn.Parameter(torch.Tensor(1, output_size), requires_grad=True)
        nn.init.zeros_(self.x_h_b)
        nn.init.zeros_(self.h_h_b)
        nn.init.zeros_(self.h_o_b)
        nn.init.zeros_(self.x_o_b)

    def forward(self, inputs, hidden_state=None):
        if hidden_state is None:
            h_0 = torch.zeros(inputs.size(0), self.hidden_size, dtype=inputs.dtype, device=inputs.device)
        else:
            h_0 = hidden_state[0]

        h_1 = self.tanh(torch.matmul(h_0, self.h_h_w) + torch.matmul(inputs, self.x_h_w) + self.h_h_b + self.x_h_b)
        z_out = torch.matmul(h_1, self.h_o_w) + torch.matmul(inputs, self.x_o_w) + self.h_o_b + self.x_o_b
        
        return z_out, (h_1, )


class DORNNCell_U(DORNNCell):
    def __init__(self, input_size, hidden_size, output_size, prefix='C'):
        super(DORNNCell_U, self).__init__(input_size, hidden_size, output_size, prefix)

        self.hx_h_w = nn.Parameter(torch.Tensor(int(math.pow(hidden_size+input_size, self.r)), hidden_size), requires_grad=True)
        self.h_o_w  = nn.Parameter(torch.Tensor(hidden_size, output_size), requires_grad=True)
        self.x_o_w = nn.Parameter(torch.Tensor(input_size, output_size), requires_grad=True)
        nn.init.kaiming_uniform_(self.hx_h_w, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.h_o_w, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.x_o_w, a=math.sqrt(5)) 

        self.hx_h_b = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.h_o_b  = nn.Parameter(torch.Tensor(1, output_size), requires_grad=True)
        self.x_o_b  = nn.Parameter(torch.Tensor(1, output_size), requires_grad=True)
        nn.init.zeros_(self.hx_h_b)
        nn.init.zeros_(self.h_o_b)
        nn.init.zeros_(self.x_o_b)


    def forward(self, inputs, hidden_state=None):
        if hidden_state is None:
            h_0 = torch.zeros(inputs.size(0), self.hidden_size, dtype=inputs.dtype, device=inputs.device)
        else:
            h_0 = hidden_state[0]

        assert 'C' in self.prefix, "Outer product model only support C type."

        hx = torch.cat([h_0, inputs], dim=1) # hx: [batch, hs+is]
        h_1 = torch.matmul(hx.view(-1, 1), hx.view(1, -1))

        if self.r >= 3:
            for _ in range(self.r-2):
                h_1 = torch.matmul(h_1.view(-1, 1), hx.view(1, -1))

        h_1 = torch.matmul(h_1.view(inputs.size(0), -1), self.hx_h_w)
        h_1 = h_1 + self.hx_h_b     
            
        h_1 = self.activation(h_1)
        
        z_out = torch.matmul(h_1, self.h_o_w) + self.h_o_b + torch.matmul(inputs, self.x_o_w) + self.x_o_b
        z_out = self.sigmoid(z_out)
        
        return z_out, (h_1, )


class DORNNCell_N(DORNNCell):
    def __init__(self, input_size, hidden_size, output_size, prefix='C'):
        super(DORNNCell_N, self).__init__(input_size, hidden_size, output_size, prefix)
        stdv = 1.0 / math.sqrt(hidden_size)

        if 'C' in self.prefix:
            self.hx_h_w = [ nn.Parameter(torch.Tensor(hidden_size+input_size, hidden_size), requires_grad=True) for _ in range(self.n) ]
            self.hx_h_b = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
            for w in self.hx_h_w:
                # nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                nn.init.uniform_(w, -stdv, stdv) 
            nn.init.zeros_(self.hx_h_b)

        else:
            self.h_h_w = [ nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True) for _ in range(self.n) ]
            self.h_h_b = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
            for w in self.h_h_w:
                # nn.init.kaiming_uniform_(w, a=math.sqrt(5)) 
                nn.init.uniform_(w, -stdv, stdv)
            nn.init.zeros_(self.h_h_b)

            self.x_h_w = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=True)
            self.x_h_b  = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
            # nn.init.kaiming_uniform_(self.x_h_w, a=math.sqrt(5)) 
            nn.init.uniform_(self.x_h_w, -stdv, stdv) 
            nn.init.zeros_(self.x_h_b)


        if 'F' in self.prefix:
            self.p_h = nn.Parameter(torch.Tensor(1, self.p_size), requires_grad=False)
            nn.init.constant_(self.p_h, self.p)
        else:
            self.p_h = nn.Parameter(torch.Tensor(1, self.p_size), requires_grad=True)
            nn.init.constant_(self.p_h, 1.0)

        self.h_o_w  = nn.Parameter(torch.Tensor(hidden_size, output_size), requires_grad=True)
        self.x_o_w = nn.Parameter(torch.Tensor(input_size, output_size), requires_grad=True)
        self.h_o_b  = nn.Parameter(torch.Tensor(1, output_size), requires_grad=True)
        self.x_o_b  = nn.Parameter(torch.Tensor(1, output_size), requires_grad=True)
        # nn.init.kaiming_uniform_(self.h_o_w, a=math.sqrt(5)) 
        # nn.init.kaiming_uniform_(self.x_o_w, a=math.sqrt(5)) 
        nn.init.uniform_(self.h_o_w, -stdv, stdv) 
        nn.init.uniform_(self.x_o_w, -stdv, stdv) 
        nn.init.zeros_(self.h_o_b)
        nn.init.zeros_(self.x_o_b)

    def forward(self, inputs, hidden_state=None):
        if hidden_state is None:
            h_0 = torch.zeros(inputs.size(0), self.hidden_size, dtype=inputs.dtype, device=inputs.device)
        else:
            h_0 = hidden_state[0]

        hx = torch.cat([h_0, inputs], dim=1)

        h_1 = []
        if 'C' in self.prefix:
            for w in self.hx_h_w: 
                h_1.append(torch.matmul(hx, w))
            h_1 = torch.cat(h_1)
            # h_1 = torch.pow(torch.abs(h_1), self.p_h) + self.hx_h_b
            h_1 = torch.sign(h_1)*torch.pow(torch.abs(h_1), self.p_h) 
            h_1 = torch.sum(h_1, 0, keepdim=True) + self.hx_h_b

        else:
            for w in self.h_h_w: 
                h_1.append(torch.matmul(h_0, w))
            h_1 = torch.cat(h_1)
            # h_1 = torch.pow(torch.abs(h_1), self.p_h) + torch.matmul(inputs, self.x_h_w) + self.h_h_b + self.x_h_b
            h_1 = torch.sum(torch.sign(h_1)*torch.pow(torch.abs(h_1), self.p_h)) \
                    + torch.matmul(inputs, self.x_h_w) + self.h_h_b + self.x_h_b

        # h_1 = torch.sum(h_1, 0, keepdim=True)/self.n
        h_1 = self.activation(h_1)

        z_out = torch.matmul(h_1, self.h_o_w) + torch.matmul(inputs, self.x_o_w) + self.h_o_b + self.x_o_b
        z_out = self.sigmoid(z_out)
        
        return z_out, (h_1,)

class DORNNCell_H(DORNNCell):

    def __init__(self, input_size, hidden_size, output_size, prefix='C'):
        super(DORNNCell_H, self).__init__(input_size, hidden_size, output_size, prefix)
        x_size, h_size = self.dx * input_size, self.dh * hidden_size

        if 'C' in self.prefix:
            self.hx_h_w = [ nn.Parameter(torch.Tensor(h_size+x_size, hidden_size), requires_grad=True) for _ in range(self.n) ]
            self.hx_h_b = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
            for w in self.hx_h_w:
                nn.init.kaiming_uniform_(w, a=math.sqrt(5)) 
            nn.init.zeros_(self.hx_h_b)

        else:
            self.h_h_w = [ nn.Parameter(torch.Tensor(h_size, hidden_size), requires_grad=True) for _ in range(self.n) ]
            self.h_h_b = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
            for w in self.h_h_w:
                nn.init.kaiming_uniform_(w, a=math.sqrt(5)) 
            nn.init.zeros_(self.h_h_b)

            self.x_h_w = nn.Parameter(torch.Tensor(x_size, hidden_size), requires_grad=True)
            self.x_h_b  = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
            nn.init.kaiming_uniform_(self.x_h_w, a=math.sqrt(5)) 
            nn.init.zeros_(self.x_h_b)

        self.pxh_p_w = nn.Parameter(torch.Tensor(self.hidden_size_p, self.p_size), requires_grad=True)
        self.pxh_p_w0 = nn.Parameter(torch.Tensor(h_size+x_size+self.p_size, self.hidden_size_p), requires_grad=True)
        self.pxh_p_b  = nn.Parameter(torch.Tensor(1, self.p_size), requires_grad=True)
        self.pxh_p_b0  = nn.Parameter(torch.Tensor(1, self.hidden_size_p), requires_grad=True)
        nn.init.kaiming_uniform_(self.pxh_p_w, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pxh_p_w0, a=math.sqrt(5))
        nn.init.zeros_(self.pxh_p_b)
        nn.init.zeros_(self.pxh_p_b0)

        self.h_o_w  = nn.Parameter(torch.Tensor(hidden_size, output_size), requires_grad=True)
        self.x_o_w = nn.Parameter(torch.Tensor(input_size, output_size), requires_grad=True)
        self.h_o_b  = nn.Parameter(torch.Tensor(1, output_size), requires_grad=True)
        self.x_o_b  = nn.Parameter(torch.Tensor(1, output_size), requires_grad=True)
        nn.init.kaiming_uniform_(self.h_o_w, a=math.sqrt(5)) 
        nn.init.kaiming_uniform_(self.x_o_w, a=math.sqrt(5)) 
        nn.init.zeros_(self.h_o_b)
        nn.init.zeros_(self.x_o_b)

    def forward(self, inputs, hidden_state=None):
        # inputs [ 1 batch, hidden features]
        if hidden_state is None:
            hidden_state = []
            init_h = torch.zeros(inputs.size(0), self.hidden_size, dtype=inputs.dtype, device=inputs.device)
            init_x = torch.zeros(inputs.size(0), self.input_size, dtype=inputs.dtype, device=inputs.device)
            init_p = torch.ones(inputs.size(0), self.p_size, dtype=inputs.dtype, device=inputs.device)

            for _ in range(self.dx):
                hidden_state.append(init_x)
            for _ in range(self.dh):
                hidden_state.append(init_h)
            hidden_state.append(init_p)
        
        h_0, x_0, p_0 = hidden_state[0:self.dx], hidden_state[self.dx:self.dx+self.dh], hidden_state[-1]
        x_0.pop(0)
        x_0.append(inputs)

        hx = torch.cat(h_0+x_0, dim=1)
        hxp = torch.cat(h_0+x_0+[p_0], dim=1)

        p_1 = torch.matmul(torch.tanh(torch.matmul(hxp, self.pxh_p_w0)+self.pxh_p_b0), self.pxh_p_w) + self.pxh_p_b
        p_1 = self.sigmoid(p_1)

        h_1 = []
        if 'C' in self.prefix:
            for w in self.hx_h_w: 
                h_1.append(torch.matmul(hx, w))
            h_1 = torch.cat(h_1)
            # h_1 = torch.pow(torch.abs(h_1), p_1) + self.hx_h_b
            h_1 = torch.sign(h_1)*torch.pow(torch.abs(h_1), p_1)
            h_1 = torch.sum(h_1, 0, keepdim=True)+ self.hx_h_b

        else:
            for w in self.h_h_w: 
                h_1.append(torch.matmul(h_0, w))
            h_1 = torch.cat(h_1)
            # h_1 = torch.pow(torch.abs(h_1), p_1) + torch.matmul(inputs, self.x_h_w) + self.h_h_b + self.x_h_b
            h_1 = torch.sum(torch.sign(h_1)*torch.pow(torch.abs(h_1), p_1)) + torch.matmul(inputs, self.x_h_w)+ self.h_h_b + self.x_h_b

        # h_1 = torch.sum(h_1, 0, keepdim=True)/self.n
        h_1 = self.activation(h_1)

        z_out = torch.matmul(h_1, self.h_o_w) + torch.matmul(inputs, self.x_o_w) + self.h_o_b + self.x_o_b
        z_out = self.sigmoid(z_out)

        h_0.pop(0)
        h_0.append(h_1)
        
        return z_out, h_0+x_0+[p_1]

# ==================== MODEL ====================
class DORNN(nn.Module):
    """mRNN with fixed d for time series prediction"""
    def __init__(self, input_size, hidden_size, output_size, prefix):
        super(DORNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        if 'U' in prefix:
            self.dornn_cell = DORNNCell_U(input_size, hidden_size, output_size, prefix=prefix)
        elif 'N' in prefix:       
            if 'H' in prefix:
                self.dornn_cell = DORNNCell_H(input_size, hidden_size, output_size, prefix=prefix)
            else:
                self.dornn_cell = DORNNCell_N(input_size, hidden_size, output_size, prefix=prefix)
        elif 'Z' in prefix:
                self.dornn_cell = DORNNCell_Z(input_size, hidden_size, output_size, prefix=prefix)
        else:
            raise NotImplementedError()

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.size(0)
        batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps, batch_size, self.output_size, dtype=inputs.dtype, device=inputs.device)
        for times in range(time_steps):
            outputs[times], hidden_state = self.dornn_cell(inputs[times], hidden_state)

        return outputs, hidden_state


