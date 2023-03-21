from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from typeguard import check_argument_types
from wenet.utils.common import get_activation


class TransducerJoint(torch.nn.Module):

    def __init__(self,
                 voca_size: int,
                 enc_output_size: int,
                 pred_output_size: int,
                 join_dim: int,
                 blank: int,
                 prejoin_linear: bool = True,
                 postjoin_linear: bool = False,
                 joint_mode: str = 'add',
                 activation: str = "tanh",
                 ILMT: bool = False,
                 ILMA: bool = False,):
        assert check_argument_types()
        # TODO(Mddct): concat in future
        assert joint_mode in ['add']
        super().__init__()

        self.activatoin = get_activation(activation)
        self.prejoin_linear = prejoin_linear
        self.postjoin_linear = postjoin_linear
        self.joint_mode = joint_mode
        self.blank = blank
        assert self.blank == 0
        self.ILMT = ILMT
        self.ILMA = ILMA

        if not self.prejoin_linear and not self.postjoin_linear:
            assert enc_output_size == pred_output_size == join_dim
        # torchscript compatibility
        self.enc_ffn: Optional[nn.Linear] = None
        self.pred_ffn: Optional[nn.Linear] = None
        if self.prejoin_linear:
            self.enc_ffn = nn.Linear(enc_output_size, join_dim)
            self.pred_ffn = nn.Linear(pred_output_size, join_dim)
        # torchscript compatibility
        self.post_ffn: Optional[nn.Linear] = None
        if self.postjoin_linear:
            self.post_ffn = nn.Linear(enc_output_size, join_dim)

        self.ffn_out = nn.Linear(join_dim, voca_size)

        if self.ILMA:
            self.ffn_out_ori = nn.Linear(join_dim, voca_size)
            self.ffn_out_ori.requires_grad = False

    def forward(self, enc_out: torch.Tensor, pred_out: torch.Tensor):
        """
        Args:
            enc_out (torch.Tensor): [B, T, E]
            pred_out (torch.Tensor): [B, T, P]
        Return:
            [B,T,U,V]
        """
        if (self.prejoin_linear and self.enc_ffn is not None
                and self.pred_ffn is not None):
            if not self.ILMA:
                enc_out = self.enc_ffn(enc_out)  # [B,T,E] -> [B,T,V]
            pred_out = self.pred_ffn(pred_out)

        if self.ILMT:
            ILM_out = F.linear(pred_out, self.ffn_out.weight[:, self.blank:], 
                                self.ffn_out.bias[self.blank:])
        if self.ILMA:
            ILM_ori = F.linear(pred_out, self.ffn_out_ori.weight[:, self.blank:], 
                                self.ffn_out_ori.bias[self.blank:])
        if not self.ILMA: 
            enc_out = enc_out.unsqueeze(2)  # [B,T,V] -> [B,T,1,V]
            pred_out = pred_out.unsqueeze(1)  # [B,U,V] -> [B,1 U, V]

            # TODO(Mddct): concat joint
            _ = self.joint_mode
            out = enc_out + pred_out  # [B,T,U,V]

            if self.postjoin_linear and self.post_ffn is not None:
                out = self.post_ffn(out)

            out = self.activatoin(out)
            out = self.ffn_out(out)
        else:
            out = None
        if self.ILMA:
            return (out, (ILM_out, ILM_ori))
        if self.ILMT:
            return (out, ILM_out)
        return out
