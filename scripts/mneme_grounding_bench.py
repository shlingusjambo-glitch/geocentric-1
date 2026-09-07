"""Evidence/abstention curriculum test on held-out synthetic question entities."""
import argparse
import copy
import json
from pathlib import Path

import torch

from geocentric.model import GeocentricGPT, GPTConfig


def batch(entities, answers):
    return torch.stack((torch.zeros_like(entities), entities + 8,
                        torch.ones_like(entities), answers,
                        torch.full_like(entities, 2), entities + 8,
                        torch.full_like(entities, 3)), dim=1)


@torch.no_grad()
def evaluate(model):
    model.eval()
    entities = torch.arange(48, 64).repeat_interleave(16)
    answers = torch.arange(72, 88).repeat(16)
    present = model(batch(entities, answers))[0][:, -1].argmax(-1)
    absent = model(batch(entities, torch.full_like(answers, 7)))[0][:, -1].argmax(-1)
    return {'supported_accuracy': float((present == answers).float().mean()),
            'unsupported_abstention': float((absent == 7).float().mean()),
            'unsupported_answer_rate': float(((absent >= 72) & (absent < 88)).float().mean()),
            'over_refusal': float((present == 7).float().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seeds', nargs='+', type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument('--output', default='runs/mneme-grounding.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed + 1000)
        original = GeocentricGPT(GPTConfig(vocab_size=88, block_size=8,
                                           n_layer=2, n_head=2, n_embd=64))
        models = {'answer_only': original, 'evidence_withholding': copy.deepcopy(original)}
        optimizers = {name: torch.optim.AdamW(m.parameters(), lr=.003) for name, m in models.items()}
        for step in range(args.steps):
            entities = torch.randint(0, 48, (32,), generator=generator)
            answers = torch.randint(72, 88, (32,), generator=generator)
            for name in list(models)[::1 if step % 2 else -1]:
                m, optimizer = models[name], optimizers[name]
                evidence, targets = answers.clone(), answers.clone()
                if name == 'evidence_withholding':
                    evidence[::2] = 7; targets[::2] = 7
                x = batch(entities, evidence)
                labels = torch.full_like(x, -100); labels[:, -1] = targets
                optimizer.zero_grad(set_to_none=True)
                loss = m(x, labels=labels, return_logits=False)[1]
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
                optimizer.step()
        results.append({'seed': seed, 'arms': {name: evaluate(m) for name, m in models.items()}})
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    report = {'settings': vars(args), 'results': results,
              'limitations': 'Synthetic tokenized evidence-present versus missing-evidence task. '
                             'Held-out question entities, shared answer vocabulary, 256 probes per condition. '
                             'Training arms receive equal updates and batch sizes; evidence exposure differs by design. '
                             'Tests curriculum principle, not the natural-language dataset builder or general hallucination elimination. '
                             'No corrupted, adversarial, contradictory or partially relevant evidence is tested.'}
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
