from contextlib import nullcontext
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geocentric.data import prepare_corpus
from geocentric.source_eval import evaluate_sources
from geocentric.tokenizer_train import train_byte_bpe_tokenizer


def test_source_ranges_reconstruct_each_file_without_overlap(tmp_path):
    corpus=tmp_path/'data';corpus.mkdir()
    (corpus/'html.txt').write_text('<p>Hello</p> ' * 80)
    (corpus/'language.txt').write_text('The moon circles the earth. ' * 80)
    tok=train_byte_bpe_tokenizer([p.read_text() for p in sorted(corpus.iterdir())],tmp_path/'tokenizer.json',vocab_size=300)
    out=tmp_path/'packed'
    prepare_corpus(tok,corpus,out,val_fraction=.2,progress=False)
    metas={split:json.loads((out/f'{split}.meta.json').read_text()) for split in ('train','val')}
    arrays={split:np.fromfile(out/f'{split}.bin',dtype=meta['dtype']) for split,meta in metas.items()}
    for split,meta in metas.items():
        assert sum(e['tokens'] for e in meta['sources']) == meta['tokens'] == len(arrays[split])
        assert meta['sources'][0]['start']==0
        assert meta['sources'][1]['start']==meta['sources'][0]['tokens']
    for train,val in zip(metas['train']['sources'],metas['val']['sources']):
        assert train['source']==val['source']
        original=tok.encode(Path(train['source']).read_text()).ids+[tok.token_to_id('<eos>')]
        restored=np.concatenate([arrays[s][entry['start']:entry['start']+entry['tokens']]
                                 for s,entry in [('train',train),('val',val)]])
        assert restored.tolist()==original
    prepare_corpus(tok,corpus,out,val_fraction=0,progress=False)
    assert not (out/'val.bin').exists()
    assert not (out/'val.meta.json').exists()


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.active_layers=1;self.config=SimpleNamespace(block_size=4);self.seen=[]
    def forward(self,x,labels,**kwargs):
        assert self.active_layers is None and not self.training
        assert torch.equal(x[:,1:],labels[:,:-1])
        assert torch.unique(x).numel()==1  # no window may cross the two source ranges
        self.seen.append(x.clone())
        return None,x.float().sum()


def test_bounded_source_eval_preserves_mode_and_depth(tmp_path):
    binary=tmp_path/'val.bin';np.array([2]*13+[7]*5,dtype=np.uint16).tofile(binary)
    meta=tmp_path/'val.meta.json';meta.write_text(json.dumps({'dtype':'uint16','sources':[
        {'source':'html','start':0,'tokens':13},{'source':'language','start':13,'tokens':5}]}))
    model=RecordingModel()
    result=evaluate_sources(model,binary,meta,torch.device('cpu'),nullcontext())
    assert model.training and model.active_layers==1
    assert result['html']['loss']==2 and result['language']['loss']==7
    assert result['html']['windows']==2 and result['language']['windows']==1
    assert len(model.seen)==3
    bad=json.loads(meta.read_text());bad['sources'][1]['tokens']=100;meta.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='outside'):
        evaluate_sources(model,binary,meta,torch.device('cpu'),nullcontext())
    assert model.training and model.active_layers==1


def test_legacy_metadata_and_invalid_budget(tmp_path):
    meta=tmp_path/'val.meta.json';meta.write_text('{"tokens":10,"dtype":"uint16"}')
    assert evaluate_sources(None,tmp_path/'unused',meta,None,nullcontext())=={}
    with pytest.raises(ValueError,match='positive'):
        evaluate_sources(None,tmp_path/'unused',meta,None,nullcontext(),0)


def test_watcher_displays_distinct_source_losses(tmp_path,monkeypatch):
    from scripts import watch_training
    monkeypatch.setattr(watch_training,'gpu_stats',lambda:'')
    (tmp_path/'training_metrics.json').write_text(json.dumps({'step':12,'status':'complete',
        'source_eval_step':12,'source_eval':{'html.txt':{'loss':.5,'tokens':256},
                                          'language.txt':{'loss':3.2,'tokens':512}}}))
    display=watch_training.render(tmp_path,[],[])
    assert 'html.txt' in display and '0.5000' in display
    assert 'language.txt' in display and '3.2000' in display


def test_watcher_keeps_saved_sft_percentage_and_resume_command(tmp_path, monkeypatch):
    from scripts import watch_training
    monkeypatch.setattr(watch_training, 'gpu_stats', lambda: '')
    monkeypatch.setattr(watch_training, 'observed_seconds_per_step', lambda step: None)
    (tmp_path/'training_metrics.json').write_text(json.dumps({
        'phase':'sft', 'step':2600, 'status':'stopped', 'tokens_seen':1234,
        'config':{'total_steps':172998, 'tokens_per_step':1024, 'params':120000000,
                  'n_layer':12, 'n_embd':768, 'block_size':1024,
                  'command':'geocentric sft --model_dir runs/geocentric-120m'},
    }))
    display=watch_training.render(tmp_path, [], [])
    assert '1.5%' in display
    assert 'step 2,600 / 172,998' in display
    assert 'geocentric sft --model_dir runs/geocentric-120m' in display
