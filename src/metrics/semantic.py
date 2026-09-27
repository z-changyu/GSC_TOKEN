"""Semantic metrics intentionally remain separate from task and communication metrics."""


def cosine_similarity(a, b):
    import torch
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

