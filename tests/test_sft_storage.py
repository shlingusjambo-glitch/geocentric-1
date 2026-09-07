import json
from pathlib import Path

import pytest

from geocentric.sft_storage import DiskExamples, iter_json_records
from geocentric.data import SFTDataset
from geocentric.tokenizer_train import train_byte_bpe_tokenizer


@pytest.mark.parametrize('suffix',['json','jsonl'])
def test_streamed_records_cross_read_boundary(tmp_path,suffix,monkeypatch):
    rows=[{'instruction':'hello'*14000,'output':'yes'}, {'instruction':'next','output':'fine'}]
    path=tmp_path/f'data.{suffix}'
    path.write_text(json.dumps(rows) if suffix=='json' else '\n'.join(json.dumps(r) for r in rows))
    monkeypatch.setattr(Path,'read_text',lambda *a,**k:pytest.fail('must not read entire file'))
    assert list(iter_json_records(path))==rows


@pytest.mark.parametrize('text',['[{"a":1},]', '[{"a":1}', '[{"a":1}] extra'])
def test_reject_malformed_array(tmp_path,text):
    path=tmp_path/'bad.json';path.write_text(text)
    with pytest.raises(ValueError):list(iter_json_records(path))


def test_disk_dataset_persists_and_reuses_matching_cache(tmp_path, capsys):
    tok=train_byte_bpe_tokenizer(['Question answer Hello world.']*5,tmp_path/'tokenizer.json',vocab_size=300)
    path=tmp_path/'sft.json';path.write_text(json.dumps([
        {'instruction':'Question','output':'Hello world.'},
        {'instruction':'Question','output':'answer'}]))
    dataset=SFTDataset(tok,path,128,cache_dir=tmp_path/'cache')
    assert isinstance(dataset.examples,DiskExamples)
    assert len(dataset)==2
    first=dataset[0]
    assert first['input_ids'].shape==first['labels'].shape
    assert (first['labels']==-100).any() and (first['labels']!=-100).any()
    cache=dataset.examples.path
    assert cache.exists()
    del dataset
    reused=SFTDataset(tok,path,128,cache_dir=tmp_path/'cache')
    assert reused.examples.reused
    assert reused[0]['input_ids'].shape == first['input_ids'].shape
    assert 'reused 2 cached' in capsys.readouterr().out
    assert cache.exists()


def test_cache_invalidates_when_data_changes(tmp_path):
    tok=train_byte_bpe_tokenizer(['Question answer.']*5,tmp_path/'tokenizer.json',vocab_size=300)
    path=tmp_path/'sft.json'
    path.write_text(json.dumps([{'instruction':'Question','output':'answer'}]))
    first=SFTDataset(tok,path,128,cache_dir=tmp_path/'cache')
    first_path=first.examples.path
    path.write_text(json.dumps([{'instruction':'Different','output':'answer'}]))
    second=SFTDataset(tok,path,128,cache_dir=tmp_path/'cache')
    assert not second.examples.reused
    assert second.examples.path != first_path
