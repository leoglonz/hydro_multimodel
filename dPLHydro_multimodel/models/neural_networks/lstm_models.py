# import math

# from sympy import Q
# import torch
# import torch.nn.functional as F
# from models.neural_networks.dropout import DropMask, createMask
# from torch.nn import Parameter



# class CudnnLstm(torch.nn.Module):
#     def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod="drW"):
#         super(CudnnLstm, self).__init__()
#         self.name = 'CudnnLstm'
#         self.inputSize = inputSize
#         self.hiddenSize = hiddenSize
#         self.dr = dr
#         self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
#         self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
#         self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
#         self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
#         self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]
#         self.cuda()

#         self.reset_mask()
#         self.reset_parameters()

#     def _apply(self, fn):
#         ret = super(CudnnLstm, self)._apply(fn)
#         return ret

#     def __setstate__(self, d):
#         super(CudnnLstm, self).__setstate__(d)
#         self.__dict__.setdefault("_data_ptrs", [])
#         if "all_weights" in d:
#             self._all_weights = d["all_weights"]
#         if isinstance(self._all_weights[0][0], str):
#             return
#         self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

#     def reset_mask(self):
#         self.maskW_ih = createMask(self.w_ih, self.dr)
#         self.maskW_hh = createMask(self.w_hh, self.dr)

#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hiddenSize)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)

#     def forward(self, input, hx=None, cx=None, tRange = None, doDropMC=False, dropoutFalse=False):
#         # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
#         if dropoutFalse and (not doDropMC):
#             doDrop = False
#         elif self.dr > 0 and (doDropMC is True or self.training is True):
#             doDrop = True
#         else:
#             doDrop = False

#         batchSize = input.size(1)

#         if hx is None:
#             hx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)
#         if cx is None:
#             cx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)

#         # cuDNN backend - disabled flat weight
#         # handle = torch.backends.cudnn.get_handle()
#         if doDrop is True:
#             self.reset_mask()
#             weight = [
#                 DropMask.apply(self.w_ih, self.maskW_ih, True),
#                 DropMask.apply(self.w_hh, self.maskW_hh, True),
#                 self.b_ih,
#                 self.b_hh,
#             ]
#         else:
#             weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

#         if tRange is not None:
#             inputx = input[tRange, :, : ]
#         else:
#             inputx = input
#         # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
#         #     input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
#         #     self.hiddenSize, 1, False, 0, self.training, False, (), None)
#         if torch.__version__ < "1.8":
#             output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
#                 inputx,
#                 weight,
#                 4,
#                 None,
#                 hx,
#                 cx,
#                 2,  # 2 means LSTM
#                 self.hiddenSize,
#                 1,
#                 False,
#                 0,
#                 self.training,
#                 False,
#                 (),
#                 None,
#             )
#         else:
#             output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
#                 input,
#                 weight,
#                 4,
#                 None,
#                 hx,
#                 cx,
#                 2,  # 2 means LSTM
#                 self.hiddenSize,
#                 0,
#                 1,
#                 False,
#                 0,
#                 self.training,
#                 False,
#                 (),
#                 None,
#             )
#         # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(  #torch._C._VariableFunctions._cudnn_rnn(
#         #     input.cuda(), weight, 4, None, hx.cuda(), cx.cuda(), 2,  # 2 means LSTM
#         #     self.hiddenSize, 1, False, 0, self.training, False, (), None)   # 4 was False before
#         return output, (hy, cy)

#     @property
#     def all_weights(self):
#         return [
#             [getattr(self, weight) for weight in weights]
#             for weights in self._all_weights
#         ]


# class CudnnLstmModel(torch.nn.Module):
#     def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
#         super(CudnnLstmModel, self).__init__()
#         self.name = 'CudnnLstmModel'
#         self.nx = nx
#         self.ny = ny
#         self.hiddenSize = hiddenSize
#         self.ct = 0
#         self.nLayer = 1
#         self.linearIn = torch.nn.Linear(nx, hiddenSize)
#         self.lstm = CudnnLstm(inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        
#         self.linearOut = torch.nn.Linear(hiddenSize, ny)
#         # self.seqModeInit(batchSize)

#         # self.activation_sigmoid = torch.nn.Sigmoid()

#     def seqModeInit(self, input):
#         batchSize = input.size(0)
#         self.hx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)
#         self.cx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)
    
#     def forward(self, x, doDropMC=False, dropoutFalse=False, tRange=None, seqMode=True):
#         x0 = F.relu(self.linearIn(x))
#         self.batchSize = x0.size(0)
#         self.seqMode = seqMode  # True for sequence to sequence, False for step by step


#         if not tRange:
#             self.seqModeInit(x0) ### TODO fix because it doesn't give the right dimensions.
        
#         ########## Temporary:
#         self.seqMode = True
#         ##########

#         if self.seqMode:
#             outLSTM, (self.hn, self.cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
#         else:
#             ## NOTE: may need a way to clear hx and cx after batched runs
#             outLSTM, (self.hx, self.cx) = self.lstm(x0, #hx=self.hx, cx=self.cx, 
#                                                     tRange=tRange, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
#         out = self.linearOut(outLSTM)

#         return out


import math

import torch
import torch.nn.functional as F
from models.neural_networks.dropout import DropMask, createMask
from torch.nn import Parameter










class CudnnLstm(torch.nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod="drW"):
        super(CudnnLstm, self).__init__()
        self.name = 'CudnnLstm'
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]
        self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault("_data_ptrs", [])
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True),
                self.b_ih,
                self.b_hh,
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
        #     input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
        #     self.hiddenSize, 1, False, 0, self.training, False, (), None)
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hiddenSize,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hiddenSize,
                0,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(  #torch._C._VariableFunctions._cudnn_rnn(
        #     input.cuda(), weight, 4, None, hx.cuda(), cx.cuda(), 2,  # 2 means LSTM
        #     self.hiddenSize, 1, False, 0, self.training, False, (), None)   # 4 was False before
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]


class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.name = 'CudnnLstmModel'
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        # self.activation_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        out = self.linearOut(outLSTM)
        return out
    