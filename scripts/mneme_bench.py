"""Controlled synthetic factual recall experiment; not a general intelligence test."""
import argparse
import copy
import json
import time

import torch
import torch.nn.functional as F

from geocentric.epicycle import mneme_loss, equant_loss
from geocentric.model import GeocentricGPT, GPTConfig


@torch.no_grad()
def score(model, answers, device):
    model.eval()
    entities = torch.arange(64, device=device)
    # Facts are trained; the two-token prefix is held out from training.
    plain = torch.stack((torch.zeros_like(entities), entities + 8,
                         torch.full_like(entities, 2)), dim=1)
    novel = torch.cat((torch.tensor([[4, 5]], device=device).expand(64, -1), plain), dim=1)
    result = {}
    for name, query in [('recall', plain), ('new_prefix', novel)]:
        logits = model(query)[0][:, -1]
        correct = logits.argmax(-1) == answers.to(device) + 72
        result[name] = float(correct.float().mean())
        result[name + '_rare'] = float(correct[8:].float().mean())
        result[name + '_common'] = float(correct[:8].float().mean())
        result[name + '_nll'] = float(F.cross_entropy(logits.float(), answers.to(device) + 72))
    sentences = torch.tensor([[0, entity + 8, 2, int(answers[entity]) + 72, 3,
                               4, 5, 4, 5, 4, 5, 6] for entity in range(64)], device=device)
    logits = model(sentences[:, :-1])[0]
    result['grammar_nll'] = float(F.cross_entropy(logits[:, 3:].reshape(-1, 136).float(),
                                                 sentences[:, 4:].reshape(-1)))
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 29, 47])
    parser.add_argument('--imbalance', type=float, default=20)
    parser.add_argument('--retention_steps', type=int, default=0)
    parser.add_argument('--compare_equant', action='store_true')
    parser.add_argument('--output', default='runs/mneme.json')
    args = parser.parse_args()
    if args.steps < 2 or args.retention_steps < 0 or args.imbalance < 1:
        parser.error('steps >= 2, retention_steps >= 0 and imbalance >= 1 required')
    torch.set_num_threads(2)
    results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 1000)
        answers = torch.randperm(64, generator=gen)
        model = GeocentricGPT(GPTConfig(vocab_size=136, block_size=16, n_layer=2,
                                        n_head=2, n_embd=64))
        models = {'ordinary_ce': model, 'mneme': copy.deepcopy(model)}
        if args.compare_equant:
            models.update(equant=copy.deepcopy(model), mneme_equant=copy.deepcopy(model))
        optimizers = {name: torch.optim.AdamW(m.parameters(), lr=.003) for name, m in models.items()}
        distribution = torch.ones(64); distribution[:8] = args.imbalance
        times = {name: 0. for name in models}
        for step in range(args.steps + args.retention_steps):
            if step == args.steps:
                distribution[8:] = 0  # retention: only the eight common facts recur
            entities = torch.multinomial(distribution, 16, replacement=True, generator=gen)
            prefix = int(torch.randint(0, 3, (), generator=gen))
            rows = [[*([4] if prefix == 1 else [5] if prefix == 2 else []),
                     0, int(entity) + 8, 2, int(answers[entity]) + 72, 3,
                     4, 5, 4, 5, 4, 5, 6] for entity in entities]
            tokens = torch.tensor(rows)
            x, labels = tokens[:, :-1], tokens[:, 1:]
            for name in list(models)[::1 if step % 2 else -1]:
                start = time.perf_counter()
                m, optimizer = models[name], optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                _, losses = m(x, labels=labels, loss_reduction='none', return_logits=False)
                selecting = step >= args.steps * .15
                if name == 'mneme_equant':
                    loss = mneme_loss(losses, labels, 136, equant_keep=.65 if selecting else None)
                elif name == 'equant' and selecting:
                    loss = equant_loss(losses, labels)
                else:
                    loss = mneme_loss(losses, labels, 136) if name == 'mneme' else losses.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
                optimizer.step()
                times[name] += time.perf_counter() - start
            if step + 1 in sorted(set([args.steps // 2, args.steps, args.steps + args.retention_steps])):
                results.append({'seed': seed, 'step': step + 1,
                                'phase': 'retention' if step >= args.steps else 'acquisition',
                                'arms': {name: {**score(m, answers, 'cpu'), 'seconds': times[name]}
                                         for name, m in models.items()}})
    from pathlib import Path
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    report = {'settings': vars(args), 'torch': torch.__version__, 'results': results,
              'limitations': 'Synthetic atomic-token facts with an imbalanced exposure distribution. '
                             'Same architecture, initialization, examples, steps and optimizer per seed. '
                             'Measures learned fact recall, not unseen facts, reasoning, or real-world truthfulness. '
                             'New-prefix probes reuse learned facts in unseen prefix combinations.'}
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
