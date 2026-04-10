#!/usr/bin/env python3
"""Semantic Action Parser — TextConformer + Multi-Head Model"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Head 클래스 정의
HEAD_CLASSES = {
    'exec_type': ['query_then_respond', 'control_then_confirm', 'query_then_judge', 'direct_respond', 'clarify'],
    'fn': ['light_control', 'heat_control', 'ac_control', 'vent_control', 'gas_control',
           'door_control', 'curtain_control', 'elevator_call', 'security_mode',
           'schedule_manage', 'weather_query', 'news_query', 'traffic_query', 'energy_query', 'info_query'],
    'room': ['living', 'kitchen', 'bedroom_main', 'bedroom_sub', 'all', 'external', 'none', 'ambiguous'],
    'param_type': ['none', 'temperature', 'brightness', 'mode', 'speed', 'direction', 'time', 'keyword'],
    'param_direction': ['none', 'up', 'down', 'set', 'on', 'off', 'open', 'close', 'stop'],
    'api': ['none', 'inbase_device', 'weather_api', 'news_api', 'traffic_api', 'energy_api', 'local_info_api'],
    'judge': ['none', 'outdoor_activity', 'clothing', 'air_quality', 'cost_trend'],
    'multi_action': ['single', 'composite'],
}

HEAD_NAMES = list(HEAD_CLASSES.keys())
HEAD_L2I = {h: {l:i for i,l in enumerate(labels)} for h, labels in HEAD_CLASSES.items()}
HEAD_I2L = {h: {i:l for i,l in enumerate(labels)} for h, labels in HEAD_CLASSES.items()}
HEAD_NC = {h: len(labels) for h, labels in HEAD_CLASSES.items()}

# Context tokens (이전 턴의 fn 결과)
CONTEXT_TOKENS = ['NONE'] + HEAD_CLASSES['fn']
CTX_L2I = {t:i for i,t in enumerate(CONTEXT_TOKENS)}

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, ff_dim=1024, kernel_size=31, dropout=0.1):
        super().__init__()
        # FFN 1/2
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        # Self-Attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        # Convolution
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2, groups=d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout))
        # FFN 1/2
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + self.attn_drop(attn_out)
        conv_in = self.conv_norm(x).permute(0, 2, 1)
        x = x + self.conv(conv_in).permute(0, 2, 1)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class SemanticActionParser(nn.Module):
    def __init__(self, pretrained_emb_weights, d_model=256, num_layers=3,
                 num_heads=4, ff_dim=1024, kernel_size=31, max_len=32, dropout=0.1):
        super().__init__()
        vocab_size, emb_dim = pretrained_emb_weights.shape

        # Embeddings
        self.token_emb = nn.Embedding.from_pretrained(pretrained_emb_weights, freeze=True, padding_idx=0)
        self.proj = nn.Linear(emb_dim, d_model)
        self.context_emb = nn.Embedding(len(CONTEXT_TOKENS), d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Encoder
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, ff_dim, kernel_size, dropout)
            for _ in range(num_layers)
        ])

        # Classification Heads
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model // 2, nc)
            )
            for name, nc in HEAD_NC.items()
        })

        # Value Pointer Head
        self.has_value = nn.Linear(d_model, 2)
        self.start_ptr = nn.Linear(d_model, 1)
        self.end_ptr = nn.Linear(d_model, 1)

        self.max_len = max_len

    def forward(self, token_ids, context_id=None):
        B = token_ids.shape[0]
        seq_len = min(token_ids.shape[1], self.max_len)

        # Token embedding + projection
        x = self.proj(self.token_emb(token_ids[:, :seq_len].long()))  # [B, T, d]

        # Context embedding
        if context_id is None:
            context_id = torch.zeros(B, dtype=torch.long, device=token_ids.device)
        ctx = self.context_emb(context_id).unsqueeze(1)  # [B, 1, d]

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d]

        # Concat: [CLS, CTX, tokens]
        x = torch.cat([cls, ctx, x], dim=1)  # [B, 2+T, d]

        # Encoder
        for block in self.blocks:
            x = block(x)

        # CLS output for classification
        cls_out = x[:, 0, :]  # [B, d]

        # Classification heads
        logits = {name: head(cls_out) for name, head in self.heads.items()}

        # Value pointer (on token positions, excluding CLS and CTX)
        token_out = x[:, 2:, :]  # [B, T, d]
        logits['has_value'] = self.has_value(cls_out)
        logits['start_ptr'] = self.start_ptr(token_out).squeeze(-1)  # [B, T]
        logits['end_ptr'] = self.end_ptr(token_out).squeeze(-1)  # [B, T]

        return logits


def compute_loss(logits, labels, head_weights=None):
    """Multi-task loss"""
    if head_weights is None:
        head_weights = {
            'exec_type': 2.0, 'fn': 2.0, 'room': 1.5, 'param_type': 1.0,
            'api': 1.0, 'judge': 1.5, 'multi_action': 1.0, 'has_value': 1.0,
        }

    loss = 0
    for head_name in HEAD_NAMES:
        w = head_weights.get(head_name, 1.0)
        loss += w * F.cross_entropy(logits[head_name], labels[head_name])

    # has_value
    if 'has_value' in labels:
        loss += head_weights.get('has_value', 1.0) * F.cross_entropy(logits['has_value'], labels['has_value'])

    return loss
