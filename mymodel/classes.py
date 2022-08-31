import torch
from torch import nn
from mymodel import MAX_LEN


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super(MultiHeadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # Q:    [batch_size, query_len, hid_dim]
        # K:    [batch_size, key_len, hid_dim]
        # V:    [batch_size, value_len, hid_dim]

        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q:    [batch_size, n_heads, query_len, head_dim]
        # K:    [batch_size, n_heads, key_len, head_dim]
        # V:    [batch_size, n_heads, value_len, head_dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # attention:    [batch_size, n_heads, query_len, key_len]
        attention = (Q @ K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # attention:    [batch_size, n_heads, query_len, head_dim]
        attention = self.dropout(torch.softmax(attention, dim=-1)) @ V

        # attention:    [batch_size, query_len, hid_dim]
        attention = attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hid_dim)

        return self.fc_o(attention)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.hid_dim = hid_dim
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super(DecoderLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.enc_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.pff = PositionWiseFeedForward(hid_dim, pf_dim, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.pff_layer_norm = nn.LayerNorm(hid_dim)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg_ = trg
        trg = self.dropout(self.self_attention(trg, trg, trg, trg_mask)) + trg_
        trg = self.self_attn_layer_norm(trg)

        trg_ = trg
        trg = self.enc_attention(trg, enc_src, enc_src, src_mask)
        trg = self.dropout(trg) + trg_
        trg = self.enc_attn_layer_norm(trg)

        trg_ = trg
        trg = self.pff(trg)
        trg = self.dropout(trg) + trg_
        trg = self.pff_layer_norm(trg)

        return trg


class Decoder(nn.Module):
    def __init__(self, output_size, n_layers, hid_dim, n_heads, pf_dim, dropout, device, max_length=MAX_LEN):
        super(Decoder, self).__init__()
        self.tok_emb = nn.Embedding(output_size, hid_dim)
        self.pos_emb = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.fc_o = nn.Linear(hid_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.device = device
        #

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg:      [batch_size, trg_len]
        # enc_src:  [batch_size, src_len, hid_dim]
        # trg_mask: [batch_size, 1, trg_len, trg_len]
        # src_mask: [batch_size, 1, 1, src_len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # pos:      [batch_size, trg_len]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # trg:      [batch_size, trg_len, hid_dim]
        trg = self.dropout(self.tok_emb(trg) * self.scale + self.pos_emb(pos))

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # return: [batch_size, trg_len, output_size] ,token multi-class
        return self.fc_o(trg)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.trained_epochs = 0
        self.best_loss = float('inf')
        self.train_ppl = []
        self.valid_ppl = []

    def make_src_mask(self, src: torch.Tensor):
        # src_mask = [batch_size, src_len]
        # src_mask = [batch_size, 1, 1, src_len]
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg: torch.Tensor):
        # trg = [batch size, trg len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask

    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, src_trg]
        src_mask = self.make_src_mask(src.input_ids)
        trg_mask = self.make_trg_mask(trg)

        # 使用albert, src保留tokenizer的结构化输出作为输入，而trg只需要给输入input_ids
        # src: [batch_size, src_len, hid_dim]
        # 从albert的输出获取last_hidden_state
        src = self.encoder(src.input_ids, src.attention_mask, src.token_type_ids).last_hidden_state
        trg = self.decoder(trg, src, trg_mask, src_mask)

        # trg:  [batch_size, trg_len, vocab_size]
        return trg
