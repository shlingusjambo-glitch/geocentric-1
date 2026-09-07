"""Probe missing, irrelevant and contradictory evidence, including unseen entities."""
import argparse
import copy
import json
from pathlib import Path

import torch

from geocentric.model import GeocentricGPT, GPTConfig


def make_batch(generator, mode, size=64, heldout=False):
    low, span = (48, 16) if heldout else (0, 48)
    q = torch.randint(0, span, (size,), generator=generator) + low
    other = (q - low + torch.randint(1, span, (size,), generator=generator)) % span + low
    third = (q - low + torch.randint(1, span, (size,), generator=generator)) % span + low
    a = torch.randint(0, 16, (size,), generator=generator) + 72
    b = (a - 72 + torch.randint(1, 16, (size,), generator=generator)) % 16 + 72
    e1, e2, target = q.clone(), other.clone(), a.clone()
    if mode == 'irrelevant':
        e1, e2, target = other, third, torch.full_like(a, 7)
    elif mode == 'conflict':
        e2, target = q.clone(), torch.full_like(a, 7)
    elif mode == 'missing':
        target = torch.full_like(a, 7)
    swap = torch.rand(size, generator=generator) < .5
    x = torch.stack((torch.zeros_like(q), torch.where(swap, e2, e1) + 8,
                     torch.ones_like(q), torch.where(swap, b, a), torch.full_like(q, 2),
                     torch.where(swap, e1, e2) + 8, torch.ones_like(q),
                     torch.where(swap, a, b), torch.full_like(q, 3), q + 8,
                     torch.full_like(q, 4)), dim=1)
    if mode == 'missing':
        x[:, [1, 3, 5, 7]] = 7
    return x, target


@torch.no_grad()
def score(model, heldout):
    generator = torch.Generator().manual_seed(999)
    result = {}
    for mode in ('supported', 'missing', 'irrelevant', 'conflict'):
        x, target = make_batch(generator, mode, size=512, heldout=heldout)
        guess = model(x)[0][:, -1].argmax(-1)
        result[mode] = {'accuracy': float((guess == target).float().mean()),
                        'abstention': float((guess == 7).float().mean()),
                        'answer_rate': float(((guess >= 72) & (guess < 88)).float().mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seeds', nargs='+', type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument('--output', default='runs/mneme-hard-grounding.json')
    parser.add_argument('--balanced', action='store_true', help='Use 50% supported cases and a wider model')
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 1000)
        base = GeocentricGPT(GPTConfig(vocab_size=88, block_size=12,
                                       n_layer=4 if args.balanced else 2,
                                       n_head=4 if args.balanced else 2,
                                       n_embd=128 if args.balanced else 64))
        models = {'answer_only': base, 'mixed_evidence': copy.deepcopy(base)}
        optimizers = {name: torch.optim.AdamW(m.parameters(), lr=.003) for name, m in models.items()}
        for step in range(args.steps):
            supported_x, supported_y = make_batch(gen, 'supported')
            # Same token/update budget. The curriculum substitutes negative cases.
            sizes = [32, 8, 12, 12] if args.balanced else [16, 16, 16, 16]
            mixed = [make_batch(gen, mode, size=size) for mode, size in
                     zip(('supported', 'missing', 'irrelevant', 'conflict'), sizes)]
            mixed_x, mixed_y = torch.cat([x for x, _ in mixed]), torch.cat([y for _, y in mixed])
            for name in list(models)[::1 if step % 2 else -1]:
                x, targets = (supported_x, supported_y) if name == 'answer_only' else (mixed_x, mixed_y)
                labels = torch.full_like(x, -100); labels[:, -1] = targets
                optimizers[name].zero_grad(set_to_none=True)
                loss = models[name](x, labels=labels, return_logits=False)[1]
                loss.backward(); torch.nn.utils.clip_grad_norm_(models[name].parameters(), 1.)
                optimizers[name].step()
        results.append({'seed': seed, 'arms': {name: {'seen_entities': score(m.eval(), False),
                                                       'unseen_entities': score(m, True)}
                                              for name, m in models.items()}})
    report = {'settings': vars(args), 'results': results,
              'limitations': 'Synthetic atomic tokens, not natural-language claims. Equal batch sizes/steps; '
                             'curricula differ in supported/unsupported exposure. New sampled contexts and values; '
                             'unseen entity tokens were excluded from training. All outcomes retained, including over-refusal.'}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
