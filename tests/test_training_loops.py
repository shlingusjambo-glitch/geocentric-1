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


@pytest.mark.parametrize("preset", ["speed", "capacity", "selective"])
def test_pretrain_checkpoint_tokens_and_completed_resume(source, monkeypatch, preset):
    import geocentric.train_pretrain as train
    directory, corpus, tokenizer = source
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))
    epi = EpicycleConfig.preset(preset)
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
    if preset == "selective":
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
        assert metrics['status']=='preparing' and 'Tokenizing' in metrics['message']
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
