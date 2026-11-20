"""Circular joint axis/phase supervision and continuous decoding."""

import torch


def labels(target, eligible, enabled):
    family = target.abs().reshape(-1, 3, 2).sum(-1).argmax(-1)
    face = family[:, None] * 2 + torch.arange(2, device=target.device)[None]
    angle = target.gather(1, face)
    observed = eligible.gather(1, face)
    coordinate = (angle + 40).div(10).remainder(9)
    low = coordinate.floor().long()
    high = (low + 1) % 9
    fraction = coordinate - low
    moving = ((angle.abs() > 1e-4) & observed).any(-1) & enabled
    strong = ((angle.abs() > 5) & observed).any(-1) & enabled
    rigid = (target.abs().amax(-1) <= 1e-4) & enabled
    return family, observed, low, high, fraction, moving, strong, rigid


def loss_function(logits, target, eligible, enabled):
    family, observed, low, high, fraction, moving, strong, rigid = labels(target, eligible, enabled)
    row = torch.arange(len(target), device=logits.device)
    conditional = logits[row, family].flatten(1).log_softmax(-1).reshape(-1, 9, 9)
    logaxis = logits.flatten(2).logsumexp(-1).log_softmax(-1)
    both = observed.all(-1)
    only_pos = observed[:, 0] & ~observed[:, 1]
    only_neg = observed[:, 1] & ~observed[:, 0]
    phase = torch.zeros(len(target), device=logits.device)
    for a, wa in [(low[:, 0], 1 - fraction[:, 0]), (high[:, 0], fraction[:, 0])]:
        for b, wb in [(low[:, 1], 1 - fraction[:, 1]), (high[:, 1], fraction[:, 1])]:
            phase -= conditional[row, a, b] * wa * wb * both
        phase -= conditional.logsumexp(-1)[row, a] * wa * only_pos
    for b, wb in [(low[:, 1], 1 - fraction[:, 1]), (high[:, 1], fraction[:, 1])]:
        phase -= conditional.logsumexp(-2)[row, b] * wb * only_neg
    phase -= logaxis[row, family] * strong
    global_logp = logits.flatten(1).log_softmax(-1).reshape(-1, 3, 9, 9)
    zero = -global_logp[:, :, 4, 4].logsumexp(-1)
    parts = []
    if moving.any():
        parts.append(phase[moving].mean())
    if rigid.any():
        parts.append(zero[rigid].mean())
    return sum(parts) / len(parts) if parts else logits.sum() * 0


def decode(logits):
    # Marginal-axis choice avoids penalizing a correct axis merely because its
    # hidden opposite-layer angle has a diffuse conditional distribution.
    axis = logits.flatten(2).logsumexp(-1).argmax(-1)
    row = torch.arange(len(logits), device=logits.device)
    selected = logits[row, axis]
    index = selected.flatten(1).argmax(-1)
    a = index // 9
    b = index % 9
    bins = logits.new_tensor(list(range(-40, 41, 10)))
    hard = torch.stack((bins[a], bins[b]), -1)
    offsets = logits.new_tensor([-1.0, 0.0, 1.0])
    aa = (a[:, None] + offsets.long()[None]) % 9
    bb = (b[:, None] + offsets.long()[None]) % 9
    window = selected[row[:, None, None], aa[:, :, None], bb[:, None, :]]
    probability = window.flatten(1).softmax(-1).reshape(-1, 3, 3)
    delta_a = (probability * offsets[None, :, None]).sum((-1, -2))
    delta_b = (probability * offsets[None, None, :]).sum((-1, -2))
    continuous = (hard + 10 * torch.stack((delta_a, delta_b), -1) + 45).remainder(90) - 45
    face = axis[:, None] * 2 + torch.arange(2, device=logits.device)[None]
    output = logits.new_zeros(len(logits), 6).scatter(1, face, continuous)
    output_hard = logits.new_zeros(len(logits), 6).scatter(1, face, hard)
    return output, output_hard, axis


def decode_hypotheses(logits):
    """Decode each axis with exactly the inference decoder, without GT routing.

    Returns Bx3x6 angles and Bx3 marginal axis probabilities. The hard angle-bin
    selection matches inference; local angle offsets and axis probabilities
    retain gradients. No averages between incompatible 3-D shapes are formed.
    """
    b = len(logits)
    conditional = torch.full_like(logits[:, None].expand(-1, 3, -1, -1, -1), -10000.)
    axis = torch.arange(3, device=logits.device)
    conditional[:, axis, axis] = logits
    angles = decode(conditional.reshape(b * 3, 3, 9, 9))[0].reshape(b, 3, 6)
    probability = logits.flatten(2).logsumexp(-1).softmax(-1)
    return angles, probability
