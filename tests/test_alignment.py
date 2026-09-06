import json
from pathlib import Path

import pytest
import torch

from geocentric.alignment import align_safety, read_alignment_data, sha256
from geocentric.behavior_eval import check_behavior, normalize_answer
from geocentric.checkpoint import save_checkpoint
from geocentric.model import GPTConfig, GeocentricGPT
from geocentric.tokenizer_train import train_byte_bpe_tokenizer


@pytest.fixture
def alignment_source(tmp_path, monkeypatch):
    import geocentric.train_sft as trainer
    monkeypatch.setattr(trainer, 'select_device', lambda: torch.device('cpu'))
    source = tmp_path / 'source'; source.mkdir()
    tok = train_byte_bpe_tokenizer(['Hello world. I cannot know. I can help.'] * 5,
                                   str(source / 'tokenizer.json'), vocab_size=300)
    model = GeocentricGPT(GPTConfig(vocab_size=tok.get_vocab_size(), block_size=256,
                                    n_layer=1, n_head=2, n_embd=32))
    # Both stages exist; alignment must load SFT, not the preferred pretraining file.
    save_checkpoint(model, source, 1, name='geocentric_pretrained_best.pt', extra={'stage':'pretrained'})
    with torch.no_grad():
        model.token_embedding.weight.add_(.01)
    save_checkpoint(model, source, 2, name='geocentric_sft_best.pt', extra={'stage':'sft'})
    data = tmp_path / 'align.jsonl'
    data.write_text(''.join(json.dumps({'category':c, 'messages':[
        {'role':'user','content':f'{c} question?'}, {'role':'assistant','content':'I can help.'}]})+'\n'
        for c in ('refusal','benign','uncertainty')))
    return source, data


def test_decline_leaves_no_output(alignment_source, tmp_path, monkeypatch):
    source, data = alignment_source
    monkeypatch.setattr('builtins.input', lambda _: 'no')
    assert align_safety(source, tmp_path/'new', data) is None
    assert not (tmp_path/'new').exists()


def test_stage_and_existing_output_guard(alignment_source, tmp_path):
    source, data = alignment_source
    with pytest.raises(ValueError, match='SFT'):
        align_safety(source, tmp_path/'new', data, checkpoint='geocentric_pretrained_best.pt', yes=True)
    with pytest.raises(ValueError, match='new output'):
        align_safety(source, source, data, yes=True)
    assert not (tmp_path/'new').exists()


def test_real_alignment_preserves_source_and_creates_trainable_pair(alignment_source, tmp_path):
    source, data = alignment_source
    hashes = {p.name:sha256(p) for p in source.iterdir() if p.is_file()}
    out=tmp_path/'pair'
    result=align_safety(source,out,data,yes=True,dtype_name='fp32',learning_rate=1e-3)
    assert result['status']=='complete'
    assert {p.name:sha256(p) for p in source.iterdir() if p.is_file()} == hashes
    assert sha256(out/'original/geocentric_sft_best.pt') == hashes['geocentric_sft_best.pt']
    baseline=torch.load(out/'original/geocentric_sft_best.pt',weights_only=False)
    changed=torch.load(next((out/'aligned').glob('*_sft.pt')),weights_only=False)
    assert changed['stage']=='sft'
    assert not torch.equal(baseline['model']['token_embedding.weight'],changed['model']['token_embedding.weight'])
    assert (out/'aligned/tokenizer.json').exists()
    # Exact overlap is caught before any inference model is loaded.
    evaluation=tmp_path/'eval.jsonl'
    evaluation.write_text(json.dumps({'category':'benign','prompt':'benign question?'})+'\n')
    with pytest.raises(ValueError,match='overlaps'):
        check_behavior(out/'original',evaluation,tmp_path/'report.json',compare_dir=out/'aligned')


def test_categories_required(tmp_path):
    data=tmp_path/'bad.jsonl'
    data.write_text(json.dumps({'category':'refusal','messages':[
        {'role':'user','content':'Q'}, {'role':'assistant','content':'A'}]})+'\n')
    with pytest.raises(ValueError,match='all three'):
        read_alignment_data(data)


def test_normalized_exact_match_is_not_substring_credit():
    assert normalize_answer(' Blue! ')=='blue'
    assert normalize_answer('It is blue or red.') != 'blue'
    assert normalize_answer('1.5') != normalize_answer('15')
    assert normalize_answer('C++') != normalize_answer('C')


def test_behavior_report_contains_real_responses(alignment_source,tmp_path,monkeypatch):
    import geocentric.device as device
    monkeypatch.setattr(device,'select_device',lambda:torch.device('cpu'))
    source,_=alignment_source
    data=tmp_path/'eval.jsonl'
    data.write_text(json.dumps({'category':'factual','prompt':'One plus one?','answers':['2']})+'\n')
    report=check_behavior(source,data,tmp_path/'report.json',max_new_tokens=3)
    result=report['models'][0]
    assert result['checkpoint']=='geocentric_sft_best.pt'
    assert isinstance(result['responses'][0]['response'],str)
    assert result['factual_exact_match'] in (0,1)
