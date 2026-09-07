"""Real tiny training runs: exercise data, optimizer, checkpoint and metrics together."""
import json

import pytest
import torch

from geocentric.epicycle import EpicycleConfig
from geocentric.checkpoint import save_checkpoint, load_checkpoint
from geocentric.model import GPTConfig, GeocentricGPT
from geocentric.tokenizer_train import train_byte_bpe_tokenizer


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "base"
    directory.mkdir()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("The earth turns while the moon orbits. A small model learns from text.\n" * 100)
    tokenizer = train_byte_bpe_tokenizer([corpus.read_text()], str(directory / "tokenizer.json"), vocab_size=300)
    return directory, corpus, tokenizer


@pytest.mark.parametrize("preset", ["speed", "capacity", "selective", "knowledge", "knowledge_selective"])
def test_pretrain_checkpoint_tokens_and_completed_resume(source, monkeypatch, preset):
    import geocentric.train_pretrain as train
    directory, corpus, tokenizer = source
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))
    epi = EpicycleConfig.preset('selective' if preset == 'knowledge_selective' else preset)
    if preset == 'knowledge_selective':
        epi.mneme = True
    epi.horizon_start = 8
    kwargs = dict(data_path=str(corpus), output_dir=str(directory), block_size=32,
                  n_layer=4, n_head=2, n_kv_head=1, n_embd=32, batch_size=2,
                  gradient_accumulation_steps=2, max_steps=3, num_workers=0,
                  dtype_name="fp32", compile_mode="off", eval_every=0, save_every=1,
                  log_every=20, val_fraction=0, epicycle=epi, loss_guard=False,
                  loss_chunk_size=7, resume_from="last")
    train.pretrain(**kwargs)
    path = directory / "geocentric_pretrained.pt"
    saved = torch.load(path, weights_only=False)
    assert saved["step"] == 3
    assert saved["loss"] is not None
    assert saved["tokens_seen"] == 3 * 2 * 2 * 32
    metrics = json.loads((directory / "training_metrics.json").read_text())
    assert metrics["tokens_seen"] == saved["tokens_seen"]
    train.pretrain(**{**kwargs, "gradient_checkpointing": True})
    resumed = torch.load(path, weights_only=False)
    assert resumed["config"]["gradient_checkpointing"] is True
    for key, tensor in saved["model"].items():
        assert torch.equal(tensor, resumed["model"][key])
    assert resumed["tokens_seen"] == saved["tokens_seen"]
    if preset in ("selective", "knowledge", "knowledge_selective"):
        train.pretrain(**{**kwargs, "max_steps": 4})
        continued = torch.load(path, weights_only=False)
        assert continued["step"] == 4
        assert continued["tokens_seen"] == 4 * 2 * 2 * 32
        assert any(not torch.equal(saved["model"][k], v) for k, v in continued["model"].items())


def test_sft_partial_accumulation_window_really_updates(source, monkeypatch):
    import geocentric.train_sft as train
    directory, corpus, tokenizer = source
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))
    model = GeocentricGPT(GPTConfig(vocab_size=tokenizer.get_vocab_size(), block_size=128,
                                    n_layer=2, n_head=2, n_embd=32))
    save_checkpoint(model, directory, 0, name="geocentric_pretrained.pt")
    data = directory / "sft.json"
    data.write_text(json.dumps([{"instruction": "What turns?", "output": "The earth."},
                                {"instruction": "What orbits?", "output": "The moon."}]))
    train.sft(str(directory), str(data), epochs=1, batch_size=1,
              gradient_accumulation_steps=8, num_workers=0, loss_guard=False,
              loss_chunk_size=7, dtype_name="fp32", compile_mode="off")
    saved = torch.load(directory / "geocentric_sft.pt", weights_only=False)
    assert saved["step"] == 1
    assert not torch.equal(saved["model"]["token_embedding.weight"], model.token_embedding.weight)


def test_vision_partial_window_really_updates(source, monkeypatch):
    from PIL import Image
    import geocentric.train_vision as train
    directory, corpus, tokenizer = source
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))
    model = GeocentricGPT(GPTConfig(vocab_size=tokenizer.get_vocab_size(), block_size=128,
                                    n_layer=2, n_head=2, n_embd=32))
    save_checkpoint(model, directory, 0, name="geocentric_pretrained.pt")
    Image.new("RGB", (16, 16), "red").save(directory / "red.png")
    data = directory / "vision.json"
    data.write_text(json.dumps([{"image": "red.png", "caption": "A red square."}]))
    train.train_vision(str(directory), str(data), epochs=1, batch_size=1,
                       gradient_accumulation_steps=8, num_workers=0, image_size=16,
                       patch_size=8, vision_layers=1, vision_width=64, vision_pool=1,
                       freeze_lm=True, loss_chunk_size=7, dtype_name="fp32")
    saved = torch.load(directory / "geocentric_vision.pt", weights_only=False)
    assert saved["step"] == 1
    assert any("exp_avg" in s for s in saved["optimizer"]["state"].values())
    assert json.loads(data.read_text())[0]["image"] == "red.png"


def test_source_losses_report_even_when_aggregate_window_is_too_short(source, monkeypatch):
    import geocentric.train_pretrain as train
    directory, corpus, tokenizer = source
    monkeypatch.setattr(train, 'select_device', lambda: torch.device('cpu'))
    tokens = len(tokenizer.encode(corpus.read_text()).ids) + 1
    train.pretrain(data_path=str(corpus), output_dir=str(directory), block_size=32,
                   n_layer=1, n_head=2, n_embd=32, batch_size=1,
                   gradient_accumulation_steps=1, max_steps=1, num_workers=0,
                   dtype_name='fp32', compile_mode='off', eval_every=1,
                   val_fraction=4.5/tokens, loss_guard=False)
    metrics=json.loads((directory/'training_metrics.json').read_text())
    assert metrics['source_eval_step']==1
    result=metrics['source_eval'][str(corpus)]
    assert result['loss'] > 0 and result['tokens']==3
    assert metrics['eval_loss'] is None


def test_sft_inherits_compact_optimizer_and_writes_startup_status(source,monkeypatch):
    import geocentric.train_sft as train
    directory,corpus,tokenizer=source
    monkeypatch.setattr(train,'select_device',lambda:torch.device('cpu'))
    model=GeocentricGPT(GPTConfig(vocab_size=tokenizer.get_vocab_size(),block_size=128,
                                 n_layer=1,n_head=2,n_embd=32))
    model._epicycle_state={'grown_to':1,'config':EpicycleConfig.preset('capacity').to_dict()}
    save_checkpoint(model,directory,82,name='geocentric_pretrained_best.pt')
    data=directory/'sft.json';data.write_text(json.dumps([{'instruction':'What turns?','output':'The earth.'}]))
    real=train.SFTDataset
    def preparing(*args,**kwargs):
        metrics=json.loads((directory/'training_metrics.json').read_text())
        assert metrics['status']=='preparing' and 'cache' in metrics['message']
        return real(*args,**kwargs)
    monkeypatch.setattr(train,'SFTDataset',preparing)
    train.sft(str(directory),str(data),epochs=1,dtype_name='fp32',loss_guard=False)
    saved=torch.load(directory/'geocentric_sft.pt',weights_only=False)
    assert saved['optimizer_type']=='RingAdamW'
    assert saved['sft_optimizer_config']['armillary_factored']
    metrics=json.loads((directory/'training_metrics.json').read_text())
    assert metrics['config']['batch_size']==1
    assert metrics['config']['loss_chunk_size']==256
    assert metrics['config']['compiled'] is False


def test_sft_separate_output_has_tokenizer_and_periodic_checkpoint(source,monkeypatch):
    import geocentric.train_sft as train
    directory,_,tokenizer=source
    monkeypatch.setattr(train,'select_device',lambda:torch.device('cpu'))
    model=GeocentricGPT(GPTConfig(vocab_size=tokenizer.get_vocab_size(),block_size=128,
                                 n_layer=1,n_head=2,n_embd=32))
    save_checkpoint(model,directory,82,name='geocentric_pretrained_best.pt')
    data=directory/'sft.json';data.write_text(json.dumps([
        {'instruction':'What turns?','output':'The earth.'},
        {'instruction':'What orbits?','output':'The moon.'}]))
    seen=[];real=train.save_checkpoint
    def record(model,out,step,**kwargs):
        seen.append(step)
        return real(model,out,step,**kwargs)
    monkeypatch.setattr(train,'save_checkpoint',record)
    output=directory.parent/'sft-output'
    train.sft(str(directory),str(data),output_dir=str(output),epochs=1,
              gradient_accumulation_steps=1,dtype_name='fp32',loss_guard=False,save_every=1)
    assert 1 in seen and 2 in seen
    assert (output/'tokenizer.json').read_bytes()==(directory/'tokenizer.json').read_bytes()


def test_sft_resume_restores_step_optimizer_tokens_and_progress(source, monkeypatch, capsys):
    import geocentric.train_sft as train
    directory, _, tokenizer = source
    monkeypatch.setattr(train, 'select_device', lambda: torch.device('cpu'))
    model=GeocentricGPT(GPTConfig(vocab_size=tokenizer.get_vocab_size(), block_size=128,
                                 n_layer=1, n_head=2, n_embd=32))
    save_checkpoint(model,directory,82,name='geocentric_pretrained_best.pt')
    data=directory/'sft.json';data.write_text(json.dumps([
        {'instruction':'One?','output':'First.'},
        {'instruction':'Two?','output':'Second.'},
        {'instruction':'Three?','output':'Third.'},
        {'instruction':'Four?','output':'Fourth.'}]))
    kwargs=dict(model_dir=str(directory),sft_data_path=str(data),batch_size=1,
                gradient_accumulation_steps=1,dtype_name='fp32',loss_guard=False,save_every=1)
    train.sft(**kwargs,epochs=1)
    first=torch.load(directory/'geocentric_sft.pt',weights_only=False)
    first_metrics=json.loads((directory/'training_metrics.json').read_text())
    assert first['step']==4 and first_metrics['step']==4
    train.sft(**kwargs,epochs=2)
    second=torch.load(directory/'geocentric_sft.pt',weights_only=False)
    metrics=json.loads((directory/'training_metrics.json').read_text())
    assert second['step']==8 and metrics['step']==8
    assert second['tokens_seen'] > first['tokens_seen']
    output=capsys.readouterr().out
    assert 'reused 4 cached' in output
    assert 'Resuming SFT at step 4 of 8' in output


def test_sft_interrupted_mid_accumulation_matches_uninterrupted(source, monkeypatch):
    import geocentric.train_sft as train
    directory, _, tokenizer = source
    monkeypatch.setattr(train, 'select_device', lambda: torch.device('cpu'))
    model = GeocentricGPT(GPTConfig(vocab_size=tokenizer.get_vocab_size(), block_size=128,
                                    n_layer=1, n_head=2, n_embd=32, dropout=.2))
    save_checkpoint(model, directory, 82, name='geocentric_pretrained_best.pt')
    data = directory / 'resume.json'
    data.write_text(json.dumps([{'instruction':f'Question {i}', 'output':f'Answer {i}.'}
                                for i in range(7)]))
    kwargs = dict(model_dir=str(directory), sft_data_path=str(data), epochs=2,
                  batch_size=1, gradient_accumulation_steps=2, dtype_name='fp32',
                  loss_guard=False, save_every=1, log_every=1)
    complete_dir, interrupted_dir = directory.parent/'complete', directory.parent/'interrupted'
    torch.manual_seed(777)
    train.sft(**kwargs, output_dir=str(complete_dir))
    real_forward = GeocentricGPT.forward
    calls = 0
    def interrupt(model, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt
        return real_forward(model, *args, **kwargs)
    monkeypatch.setattr(GeocentricGPT, 'forward', interrupt)
    torch.manual_seed(777)
    train.sft(**kwargs, output_dir=str(interrupted_dir))
    stopped = torch.load(interrupted_dir/'geocentric_sft.pt', weights_only=False)
    assert stopped['step'] == 1
    assert stopped['sft_resume_state']['next_batch'] == 2
    assert json.loads((interrupted_dir/'training_metrics.json').read_text())['step'] == 1
    monkeypatch.setattr(GeocentricGPT, 'forward', real_forward)
    train.sft(**kwargs, output_dir=str(interrupted_dir))
    expected = torch.load(complete_dir/'geocentric_sft.pt', weights_only=False)
    actual = torch.load(interrupted_dir/'geocentric_sft.pt', weights_only=False)
    assert expected['step'] == actual['step'] == 8
    assert expected['tokens_seen'] == actual['tokens_seen']
    for name, tensor in expected['model'].items():
        torch.testing.assert_close(actual['model'][name], tensor, rtol=0, atol=0)
    for key, state in expected['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(actual['optimizer']['state'][key][name], value, rtol=0, atol=0)
    # A completed run is a no-op on the weights when relaunched with the same budget.
    train.sft(**kwargs, output_dir=str(interrupted_dir))
    again = torch.load(interrupted_dir/'geocentric_sft.pt', weights_only=False)
    assert again['step'] == 8
    for name, tensor in actual['model'].items():
        torch.testing.assert_close(again['model'][name], tensor, rtol=0, atol=0)
@pytest.mark.parametrize('failure', ['forward', 'gradients'])
def test_sft_failure_status_and_no_phantom_steps(source, monkeypatch, failure):
    import geocentric.train_sft as train
    directory, _, tokenizer = source
    monkeypatch.setattr(train, 'select_device', lambda: torch.device('cpu'))
    model = GeocentricGPT(GPTConfig(vocab_size=tokenizer.get_vocab_size(), block_size=128,
                                    n_layer=1, n_head=2, n_embd=32))
    save_checkpoint(model, directory, 82, name='geocentric_pretrained_best.pt')
    data = directory/'bad-run.json'
    data.write_text(json.dumps([{'instruction': 'Question?', 'output': 'Answer.'}] * 3))
    if failure == 'forward':
        def fail(*args, **kwargs):
            raise RuntimeError('injected forward failure')
        monkeypatch.setattr(GeocentricGPT, 'forward', fail)
    else:
        monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', lambda *a, **k: torch.tensor(float('inf')))
    with pytest.raises(RuntimeError):
        train.sft(str(directory), str(data), epochs=1, dtype_name='fp32',
                  gradient_accumulation_steps=1, loss_guard=False)
    metrics = json.loads((directory/'training_metrics.json').read_text())
    assert metrics['status'] == 'failed' and metrics['step'] == 0
    assert not (directory/'geocentric_sft.pt').exists()


def test_sft_best_checkpoint_has_current_resume_metadata(source, monkeypatch):
    import geocentric.train_sft as train
    directory, _, tokenizer = source
    monkeypatch.setattr(train, 'select_device', lambda: torch.device('cpu'))
    model = GeocentricGPT(GPTConfig(vocab_size=tokenizer.get_vocab_size(), block_size=128,
                                    n_layer=1, n_head=2, n_embd=32))
    save_checkpoint(model, directory, 82, name='geocentric_pretrained_best.pt')
    data = directory/'eval-run.json'
    data.write_text(json.dumps([{'instruction': 'Question?', 'output': 'Answer.'}] * 60))
    train.sft(str(directory), str(data), epochs=1, dtype_name='fp32', eval_ratio=.2,
              gradient_accumulation_steps=1, loss_guard=False)
    best = torch.load(directory/'geocentric_sft_best.pt', weights_only=False)
    state = best['sft_resume_state']
    assert state['epoch'] == 2 and state['next_batch'] == 0
    assert state['best_eval'] == best['eval_loss']
def test_mneme_grounded_dataset_trains_and_evaluates_through_real_pipeline(source, monkeypatch):
    import geocentric.train_sft as train
    import geocentric.device as devices
    from geocentric.mneme import prepare_grounding_data, check_grounding
    directory, _, tokenizer = source
    monkeypatch.setattr(train, 'select_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(devices, 'select_device', lambda: torch.device('cpu'))
    model = GeocentricGPT(GPTConfig(vocab_size=tokenizer.get_vocab_size(), block_size=512,
                                    n_layer=1, n_head=2, n_embd=32))
    save_checkpoint(model, directory, 1, name='geocentric_pretrained.pt')
    cases = [dict(question='What turns?', context='The earth turns.', answer='earth'),
             dict(question='What turns?', context='No information about turning.', answerable=False)]
    source_path, curriculum = directory/'facts.jsonl', directory/'curriculum.jsonl'
    source_path.write_text('\n'.join(json.dumps(case) for case in cases))
    assert prepare_grounding_data(source_path, curriculum) == 3
    train.sft(str(directory), str(curriculum), epochs=1, gradient_accumulation_steps=1,
              dtype_name='fp32', loss_guard=False)
    saved = torch.load(directory/'geocentric_sft.pt', weights_only=False)
    assert saved['step'] == 3
    # Pipeline smoke test only: deliberately trained cases are not held-out evidence.
    report = check_grounding(directory, source_path, directory/'report.json', max_new_tokens=4)
    assert len(report['results']) == 2 and report['stage'] == 'sft'
    assert report['data_sha256']
