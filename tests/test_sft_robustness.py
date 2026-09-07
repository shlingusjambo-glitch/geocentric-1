from contextlib import nullcontext
import threading
import json
from pathlib import Path

import pytest
import torch

from geocentric.train_sft import _prefetch, _normalize_gradients, evaluate, sft


def test_prefetch_order_and_producer_error():
    def source():
        yield 1
        yield 2
        raise ValueError('corrupt data')
    batches = _prefetch(source())
    assert next(batches) == 1
    assert next(batches) == 2
    with pytest.raises(ValueError, match='corrupt data'):
        next(batches)
    assert list(_prefetch([])) == []
    with pytest.raises(ValueError, match='positive'):
        list(_prefetch([], buffer=0))


def test_prefetch_cancel_full_queue_terminates_worker():
    filled = threading.Event()
    before = set(threading.enumerate())
    def source():
        for i in range(100):
            if i == 4:
                filled.set()
            yield i
    batches = _prefetch(source())
    assert next(batches) == 0
    assert next(batches) == 1
    assert filled.wait(timeout=2)
    batches.close()
    assert not [t for t in threading.enumerate() if t not in before and t.name == 'sft-prefetch']


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_grouped_gradient_normalization(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    parameters = [torch.nn.Parameter(torch.zeros(3, device=device, dtype=dtype))
                  for dtype in (torch.float32, torch.float64, torch.float32)]
    for p in parameters[:2]:
        p.grad = torch.tensor([1., -2., 3.], device=device, dtype=p.dtype)
    _normalize_gradients(parameters, .37)
    for p in parameters[:2]:
        expected = torch.tensor([1., -2., 3.], device=device, dtype=p.dtype) * .37
        torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
    assert parameters[-1].grad is None


class ValidationModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def forward(self, *args, **kwargs):
        return None, torch.tensor(self.value)


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('value,labels,error', [
    (float('nan'), [1, 2], 'Non-finite'),
    (0., [-100, -100], 'no supervised'),
    (4., [1, 2], None),
])
def test_validation_fails_honestly_and_restores_mode(training, value, labels, error):
    model = ValidationModel(value).train(training)
    batch = {'input_ids': torch.tensor([[1, 2]]), 'labels': torch.tensor([labels])}
    if error:
        with pytest.raises(RuntimeError, match=error):
            evaluate(model, [batch], torch.device('cpu'), nullcontext())
    else:
        assert evaluate(model, [batch], torch.device('cpu'), nullcontext()) == 2
    assert model.training == training


@pytest.mark.parametrize('kwargs', [{'log_every': 0}, {'eval_ratio': 1},
                                   {'learning_rate': float('nan')}, {'grad_clip': 0}])
def test_invalid_settings_rejected_before_loading(kwargs):
    with pytest.raises(ValueError):
        sft('missing', 'missing', **kwargs)


def test_metrics_failed_replace_preserves_previous_file(tmp_path, monkeypatch):
    from geocentric.training_metrics import initialize_training_metrics, update_training_metrics
    initialize_training_metrics(tmp_path, 'sft', {})
    path = tmp_path/'training_metrics.json'
    previous = path.read_bytes()
    def fail_replace(*args, **kwargs):
        assert path.read_bytes() == previous
        raise OSError('disk failure')
    monkeypatch.setattr(Path, 'replace', fail_replace)
    with pytest.raises(OSError, match='disk failure'):
        update_training_metrics(tmp_path, {'step': 123})
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('offsets,dropped', [(None, 0), (42, 0), ([0, 16], None)])
def test_malformed_cache_index_rebuilds(tmp_path, offsets, dropped):
    from geocentric.sft_storage import DiskExamples, CACHE_VERSION
    (tmp_path/'bad.bin').write_bytes(b'\x00' * 16)
    (tmp_path/'bad.index.json').write_text(json.dumps(
        {'version': CACHE_VERSION, 'offsets': offsets, 'dropped': dropped}))
    cache = DiskExamples(tmp_path, 'bad')
    assert not cache.reused
    cache.append({'input_ids': torch.arange(2), 'labels': torch.arange(2)})
    cache.finish()
    torch.testing.assert_close(cache[0]['input_ids'], torch.arange(2))
