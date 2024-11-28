import torch
from torch import nn
import torch.nn.functional as F

import copy

def _get_clones(module, N, layer_share=False):
    # import ipdb; ipdb.set_trace()
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PSDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=512,
        d_ffn=1024,
        n_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
            self,
            tgt,
            tgt_query_pos,
            memory, # hwd bs dmodel
    ):
        # self attn
        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attn
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos),
            memory,
            memory,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt

class PSDecoder(nn.Module):   
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []
        self.norm = norm
        
    def forward(
            self,
            tgt,
            memory,
    ):
        # import pdb;pdb.set_trace()
        output = tgt
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            output = layer(
                tgt=output,
                tgt_query_pos=None,
                memory=memory,
            )
            intermediate.append(self.norm(output))

        return [itm_out.transpose(0, 1) for itm_out in intermediate]

class PlaneSliceAttentionTransformer(nn.Module):
    def __init__(
            self, 
            hidden_size=512,
            num_decoder_layers=6,
            nhead=8,
            ):
        super().__init__()
        self.hidden_size = hidden_size
        decoder_layer = PSDecoderLayer(
            hidden_size,
            nhead,
        )
        decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder = PSDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
        )
        self.ps_pos = nn.Parameter(torch.empty(hidden_size, 3, 128))
        nn.init.normal_(self.ps_pos)

    def forward(self, queries, img_embed, plane, slice):
        # queries: bs 300 768
        # img_embed: bs 64(seq len) 768
        bs = img_embed.shape[0]
        plane_mapping = {
            'Axial': 0,
            'Coronal': 1,
            'Sagittal': 2,
        }

        for b in range(bs):
            pos_b = self.ps_pos[None,:,plane_mapping[plane[b]],int(slice[b])//2] # 1 768
            img_embed[b] = img_embed[b]+pos_b
            queries[b] = queries[b]+pos_b

        # import pdb;pdb.set_trace()

        output = self.decoder(
            tgt = queries.transpose(0,1), # bs nq dmodel -> nq bs dmodel
            memory = img_embed.transpose(0,1), # bs len_seq dmodel -> \sum{hwd} bs dmodel
        )
        res = output[-1]

        return res