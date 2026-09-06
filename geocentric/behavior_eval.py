"""Auditable held-out answers with narrow exact-match factual scoring, not an LLM judge."""
from __future__ import annotations

import gc
import json
from pathlib import Path

from geocentric.alignment import prompt_key, sha256


def normalize_answer(text):
    # Preserve internal punctuation: 1.5 must never receive credit for 15.
    return ' '.join(text.casefold().split()).strip().rstrip('.!?')


def read_cases(path):
    cases = []
    for line, raw in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if (not isinstance(row, dict) or row.get('category') not in ('factual', 'refusal', 'benign', 'uncertainty')
                or not isinstance(row.get('prompt'), str) or not row['prompt'].strip()):
            raise ValueError(f'Invalid evaluation case on line {line}')
        if row['category'] == 'factual':
            answers = row.get('answers')
            if (not isinstance(answers, list) or not answers
                    or any(not isinstance(a, str) or not normalize_answer(a) for a in answers)):
                raise ValueError(f'Factual case {line} requires nonempty accepted answers')
        cases.append(row)
    if not cases:
        raise ValueError('Evaluation data is empty')
    return cases


def check_behavior(model_dir, data_path, output, *, compare_dir=None, max_new_tokens=128):
    from geocentric.checkpoint import load_model_and_tokenizer, _find_checkpoint
    from geocentric.generate import build_chat_prompt, generate_text
    from geocentric.device import cleanup, select_device

    if max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ValueError('Choose a new report path; existing reports are not overwritten')
    cases = read_cases(data_path)
    directories = [Path(model_dir).expanduser().resolve()]
    if compare_dir:
        directories.append(Path(compare_dir).expanduser().resolve())
    # Both variants share a parent manifest. Exact normalized user prompts may not leak.
    heldout = {prompt_key([{'role': 'user', 'content': case['prompt']}]) for case in cases}
    for directory in directories:
        manifest_path = directory.parent / 'alignment.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            training = set()
            for key in manifest.get('training_prompt_keys', []):
                messages = [{'role': role, 'content': text} for role, text in json.loads(key) if role == 'user']
                training.add(prompt_key(messages))
                # Also reject matching any individual turn in multi-turn training data.
                training.update(prompt_key([m]) for m in messages)
            if training & heldout:
                raise ValueError('Evaluation overlaps alignment training prompts; use held-out cases')
    report = {'data_sha256': sha256(data_path), 'max_new_tokens': max_new_tokens,
              'scoring': 'Normalized exact match on factual cases only; other responses require human review. '
                         'No score proves hallucination-free behavior. Semantic paraphrases can fail exact match. '
                         'Prompt overlap detection covers exact normalized alignment prompts, not pretraining overlap.',
              'models': []}
    device = select_device()
    for directory in directories:
        checkpoint = _find_checkpoint(directory, None)
        model, tokenizer, stage = load_model_and_tokenizer(directory, device=device, with_stage=True)
        responses = []
        try:
            for case in cases:
                prompt = (build_chat_prompt([{'role': 'user', 'content': case['prompt']}])
                          if stage in ('sft', 'vision') else case['prompt'])
                prompt_tokens = len(tokenizer.encode(prompt).ids)
                budget = min(max_new_tokens, model.config.block_size - prompt_tokens)
                if budget < 1:
                    raise ValueError('Evaluation prompt exceeds checkpoint context; shorten the prompt')
                response = generate_text(model, tokenizer, prompt, max_new_tokens=budget,
                                         temperature=0., device=device)
                item = {**case, 'response': response, 'generation_budget': budget,
                        'review': 'pending'}
                if case['category'] == 'factual':
                    item['exact_match'] = normalize_answer(response) in {normalize_answer(a) for a in case['answers']}
                responses.append(item)
        finally:
            del model, tokenizer
            gc.collect()
            cleanup(device)
        factual = [r for r in responses if r['category'] == 'factual']
        report['models'].append({'directory': str(directory), 'checkpoint': checkpoint.name, 'stage': stage,
                                 'checkpoint_sha256': sha256(checkpoint), 'responses': responses,
                                 'factual_cases': len(factual),
                                 'factual_exact_match': sum(r['exact_match'] for r in factual) / len(factual) if factual else None})
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(f'Behavior report saved to {target}. Review refusals, benign answers, and uncertainty manually.')
    return report
