#!/usr/bin/env python3
"""5-Head CNN Model — ko-sbert frozen embeddings + CNN 4L + 5 classification heads"""
import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_CLASSES = {
    'fn': ['light_control', 'heat_control', 'ac_control', 'vent_control', 'gas_control',
           'door_control', 'curtain_control', 'elevator_call', 'security_mode',
           'schedule_manage', 'weather_query', 'news_query', 'traffic_query',
           'energy_query', 'home_info', 'system_meta', 'market_query',
           'medical_query', 'vehicle_manage', 'unknown'],
    'exec_type': ['query_then_respond', 'control_then_confirm', 'query_then_judge',
                  'direct_respond', 'clarify'],
    'param_direction': ['none', 'up', 'down', 'set', 'on', 'off', 'open', 'close', 'stop'],
    'param_type': ['none', 'temperature', 'brightness', 'mode', 'speed'],
    'judge': ['none', 'outdoor_activity', 'clothing', 'air_quality', 'cost_trend'],
}

HEAD_NAMES = list(HEAD_CLASSES.keys())
HEAD_L2I = {h: {l: i for i, l in enumerate(labels)} for h, labels in HEAD_CLASSES.items()}
HEAD_I2L = {h: {i: l for i, l in enumerate(labels)} for h, labels in HEAD_CLASSES.items()}
HEAD_NC = {h: len(labels) for h, labels in HEAD_CLASSES.items()}


class CNNMultiHead(nn.Module):
    def __init__(self, pretrained_emb_weights, d_model=256, max_len=32, dropout=0.1):
        super().__init__()
        vocab_size, emb_dim = pretrained_emb_weights.shape

        # Frozen embedding + projection
        self.token_emb = nn.Embedding.from_pretrained(pretrained_emb_weights, freeze=True, padding_idx=0)
        self.proj = nn.Linear(emb_dim, d_model)
        self.drop_in = nn.Dropout(dropout)

        # CNN 4 layers with different kernel sizes
        self.conv1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout))
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout))
        self.conv3 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout))
        self.conv4 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout))

        # 5 classification heads
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, nc)
            )
            for name, nc in HEAD_NC.items()
        })

        self.max_len = max_len

    def forward(self, token_ids):
        seq_len = min(token_ids.shape[1], self.max_len)
        x = self.proj(self.token_emb(token_ids[:, :seq_len].long()))  # [B, T, d]
        x = self.drop_in(x)

        # CNN expects [B, d, T]
        x = x.permute(0, 2, 1)

        x = x + self.conv1(x)
        x = x + self.conv2(x)
        x = x + self.conv3(x)
        x = x + self.conv4(x)

        # Global mean pooling → [B, d]
        x = x.mean(dim=2)

        # Classification heads
        logits = {name: head(x) for name, head in self.heads.items()}
        return logits


def compute_loss(logits, labels, head_weights=None):
    if head_weights is None:
        head_weights = {
            'fn': 2.0, 'exec_type': 2.0, 'param_direction': 1.5,
            'param_type': 1.0, 'judge': 1.5,
        }

    loss = 0
    for head_name in HEAD_NAMES:
        if head_name in labels:
            w = head_weights.get(head_name, 1.0)
            loss += w * F.cross_entropy(logits[head_name], labels[head_name])
    return loss
