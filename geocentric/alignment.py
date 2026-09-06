"""Explicit, optional post-SFT alignment; the source checkpoint is never rewritten."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

from geocentric.checkpoint import _find_checkpoint, checkpoint_stage, find_tokenizer_path

CATEGORIES = frozenset(("refusal", "benign", "uncertainty"))
POLICY = ("Refuse assistance that facilitates harm or wrongdoing; answer legitimate sensitive "
          "questions helpfully; acknowledge missing evidence and uncertainty honestly.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prompt_key(messages):
    return json.dumps([(m['role'], ' '.join(m['content'].lower().split()))
                       for m in messages if m['role'] != 'assistant'], ensure_ascii=False)


def read_alignment_data(path):
    rows = []
    for line, raw in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f'Line {line}: expected a JSON object')
        messages = row.get('messages')
        if row.get('category') not in CATEGORIES:
            raise ValueError(f'Line {line}: category must be refusal, benign, or uncertainty')
        if (not isinstance(messages, list) or len(messages) < 2
                or any(not isinstance(m, dict) for m in messages)
                or messages[-1].get('role') != 'assistant'
                or not any(m.get('role') == 'user' for m in messages)):
            raise ValueError(f'Line {line}: provide user and assistant messages ending in assistant')
        for message in messages:
            if (message.get('role') not in ('system', 'user', 'assistant')
                    or not isinstance(message.get('content'), str) or not message['content'].strip()):
                raise ValueError(f'Line {line}: invalid message')
        rows.append({'category': row['category'], 'messages': messages})
    missing = CATEGORIES - {row['category'] for row in rows}
    if missing:
        raise ValueError(f'Alignment requires all three categories; missing {sorted(missing)}')
    return rows


def align_safety(model_dir, output_dir, data_path, *, checkpoint=None, epochs=1,
                 batch_size=1, gradient_accumulation_steps=4, learning_rate=1e-5,
                 dtype_name='auto', yes=False):
    from geocentric.data import SFTDataset
    from geocentric.tokenizer_train import load_tokenizer
    from geocentric.train_sft import sft
    import torch

    if (epochs < 1 or batch_size < 1 or gradient_accumulation_steps < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('Epochs, batches, accumulation and learning rate must be positive')
    if dtype_name not in ('auto', 'fp32', 'fp16', 'bf16'):
        raise ValueError('Unsupported dtype')
    src, out = Path(model_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if out.exists() or src == out or src in out.parents:
        raise ValueError('Choose a new output directory outside the source model directory')
    selected = _find_checkpoint(src, checkpoint).resolve()
    if selected.parent != src or not selected.is_file():
        raise ValueError('Checkpoint must be a file directly inside model_dir')
    if checkpoint_stage(src, selected.name) not in ('sft', 'vision'):
        raise ValueError('Complete pretraining and SFT first; align-safety requires an SFT checkpoint')
    rows = read_alignment_data(data_path)
    tokenizer_path = find_tokenizer_path(src)
    tokenizer = load_tokenizer(tokenizer_path)
    payload = torch.load(selected, map_location='cpu', weights_only=False)
    config = payload['config']
    del payload
    # Verify that every reviewed example fits and actually supervises assistant tokens.
    dataset = SFTDataset(tokenizer, data_path, block_size=config['block_size'])
    if len(dataset) != len(rows):
        raise ValueError('Some alignment examples are overlong or unsupervised; shorten them before training')
    counts = dict(Counter(row['category'] for row in rows))
    print(f'Policy: {POLICY}\nSource: {selected}\nUnchanged copy: {out / "original"}'
          f'\nAligned copy: {out / "aligned"}\nExamples: {counts}\nEpochs: {epochs}; learning rate: {learning_rate:g}')
    print('This is additional supervised training, not a guarantee of safety or factual accuracy.')
    if not yes:
        try:
            accepted = input('Continue with alignment? [y/N] ').strip().lower() in ('y', 'yes')
        except (EOFError, KeyboardInterrupt):
            accepted = False
        if not accepted:
            print('Cancelled. No model or output files changed.')
            return None

    out.mkdir(parents=True, exist_ok=False)
    original, aligned = out / 'original', out / 'aligned'
    manifest = {'status': 'preparing', 'policy': POLICY, 'source_checkpoint': str(selected),
                'source_sha256': sha256(selected), 'categories': counts,
                'epochs': epochs, 'learning_rate': learning_rate,
                'batch_size': batch_size, 'gradient_accumulation_steps': gradient_accumulation_steps,
                'dtype': dtype_name, 'training_prompt_keys': [prompt_key(r['messages']) for r in rows]}

    def save_status(status):
        manifest['status'] = status
        temporary = out / 'alignment.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        temporary.replace(out / 'alignment.json')

    try:
        save_status('preparing')
        for directory in (original, aligned):
            directory.mkdir()
            shutil.copy2(tokenizer_path, directory / 'tokenizer.json')
            (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        shutil.copy2(selected, original / selected.name)
        if sha256(original / selected.name) != manifest['source_sha256']:
            raise OSError('Original checkpoint copy verification failed')
        # Train the validated snapshot, even if the user's input file changes later.
        snapshot = out / 'alignment-data.jsonl'
        snapshot.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        manifest['data_sha256'] = sha256(snapshot)
        save_status('training')
        sft(str(original), str(snapshot), output_dir=str(aligned), checkpoint_name=selected.name,
            epochs=epochs, batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate, dtype_name=dtype_name, modelver='Geocentric Aligned',
            compile_mode='off', loss_chunk_size=256, eval_ratio=0., num_workers=0)
        metrics = json.loads((aligned / 'training_metrics.json').read_text())
        save_status(metrics.get('status', 'incomplete'))
    except KeyboardInterrupt:
        save_status('stopped')
        raise
    except Exception:
        save_status('failed')
        raise
    print(f'Alignment status: {manifest["status"]}. Original retained at {original}.')
    return manifest
