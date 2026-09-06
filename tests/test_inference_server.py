import copy
import json
import threading
import urllib.request
import urllib.error

import pytest
import torch

from geocentric.model import GPTConfig, GeocentricGPT, KVCache
from geocentric.epicycle import RingAdamW, EpicycleConfig
from geocentric.generate import repeated_tail
from geocentric.sampling import sample_token
from geocentric.web_server import ChatEngine, ChatServer, network_urls


def tiny():
    return GeocentricGPT(GPTConfig(vocab_size=97, block_size=32, n_layer=2,
                                  n_head=2, n_kv_head=1, n_embd=32)).eval()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_preallocated_cache_matches_full_logits(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA hardware required')
    torch.manual_seed(42)
    model = tiny().to(device)
    ids = torch.randint(0, 97, (2, 15), device=device)
    with torch.no_grad():
        full, _ = model(ids)
        caches = [KVCache(max_length=32) for _ in model.blocks]
        model(ids[:, :5], caches=caches)
        pointers = [c.k.data_ptr() for c in caches]
        pieces = []
        for start, end in [(5, 8), (8, 9), (9, 15)]:
            logits, _ = model(ids[:, start:end], caches=caches, position_offset=start)
            pieces.append(logits)
            assert pointers == [c.k.data_ptr() for c in caches]
        torch.testing.assert_close(torch.cat(pieces, 1), full[:, [7, 8, 14]], atol=2e-6, rtol=2e-5)


def test_cache_capacity_and_autograd_transition():
    cache = KVCache(max_length=4)
    k = torch.randn(1, 1, 1, 8, requires_grad=True)
    with torch.no_grad():
        cache.append(k, k)
    cache.append(k, k)
    with torch.no_grad():
        actual, _ = cache.append(k, k)
        torch.testing.assert_close(actual, k.expand(1, 1, 3, 8))
        with pytest.raises(ValueError):
            cache.append(k.expand(1, 1, 2, 8), k.expand(1, 1, 2, 8))


def test_sampling_candidate_distribution(monkeypatch):
    captured = []
    def sample(p, n):
        captured.append(p)
        return p.argmax(-1, keepdim=True)
    monkeypatch.setattr(torch, 'multinomial', sample)
    logits = torch.tensor([[1., 4., 2., 3., -1.]])
    result = sample_token(logits, torch.tensor([[0]]), top_k=3, top_p=1,
                          min_p=0, temperature=1, repetition_penalty=1)
    assert result.item() == 1
    torch.testing.assert_close(captured[0], torch.tensor([[4., 3., 2.]]).softmax(-1))
    assert captured[0].numel() == 3


@pytest.mark.parametrize('partitioned', [False, True])
def test_ring_resume_across_all_rotations(partitioned):
    torch.manual_seed(11)
    p = torch.nn.Parameter(torch.randn(37, 11))
    opt = RingAdamW([p], factored=True, partitioned=partitioned, dwell=2)
    gradients = [torch.randn_like(p) for _ in range(15)]
    for grad in gradients[:5]:
        p.grad = grad.clone(); opt.step()
    q = torch.nn.Parameter(p.detach().clone())
    resumed = RingAdamW([q], factored=True, partitioned=partitioned, dwell=2)
    state = copy.deepcopy(opt.state_dict())
    if not partitioned:
        state['armillary'].pop('partitioned')  # actual legacy metadata
    resumed.load_state_dict(state)
    for grad in gradients[5:]:
        p.grad = grad.clone(); q.grad = grad.clone()
        opt.step(); resumed.step()
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    if partitioned:
        assert max(opt.ring_load) - min(opt.ring_load) <= 1
        assert opt.state[p]['m'].numel() <= (p.numel() + 3) // 4


def test_single_partition_matches_legacy_ring():
    p = torch.nn.Parameter(torch.randn(13, 7))
    q = torch.nn.Parameter(p.detach().clone())
    a = RingAdamW([p], rings=1, factored=True)
    b = RingAdamW([q], rings=1, factored=True, partitioned=True)
    for _ in range(4):
        p.grad = torch.randn_like(p); q.grad = p.grad.clone()
        a.step(); b.step()
        torch.testing.assert_close(p, q)
    assert not EpicycleConfig.preset('capacity').armillary_partitioned
    assert EpicycleConfig.preset('balanced').armillary_partitioned


def test_loop_guard_requires_sustained_cycle():
    assert repeated_tail([1, 2] * 6)
    assert repeated_tail([1] * 6)
    assert not repeated_tail([1, 2] * 5)
    assert not repeated_tail([1, 2] * 5 + [3])


@pytest.fixture
def server():
    engine = ChatEngine(tiny(), None)
    def stream(prompt, options, event, stats):
        yield 'Hello '
        yield 'world'
        stats.update(finish_reason='stop', generated_tokens=2)
    engine.stream = stream
    with ChatServer(('127.0.0.1', 0), engine) as instance:
        thread = threading.Thread(target=instance.serve_forever, daemon=True)
        thread.start()
        yield 'http://127.0.0.1:' + str(instance.server_port), engine
        instance.shutdown()
        thread.join()


def request(url, payload=None, origin=None):
    headers = {'Content-Type': 'application/json'}
    if origin:
        headers['Origin'] = origin
    return urllib.request.urlopen(urllib.request.Request(url,
        data=None if payload is None else json.dumps(payload).encode(), headers=headers))


def test_web_http_assets_stream_and_busy(server):
    url, engine = server
    for path in ['/', '/style.css', '/app.js', '/api/model', '/health']:
        with request(url + path) as response:
            assert response.status == 200
            assert response.read()
    payload = {'messages': [{'role': 'user', 'content': 'Hello'}], 'max_new_tokens': 5}
    with request(url + '/api/chat', payload) as response:
        events = [json.loads(line) for line in response]
    assert ''.join(e.get('text', '') for e in events) == 'Hello world'
    assert events[-1]['finish_reason'] == 'stop'
    engine.lock.acquire()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            request(url + '/api/chat', payload)
        assert exc.value.code == 429
    finally:
        engine.lock.release()
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(url + '/api/chat', payload, 'http://unrelated.invalid')
    assert exc.value.code == 403


@pytest.mark.parametrize('payload', [{}, {'messages': [{'role': [], 'content': 'x'}]},
    {'messages': [{'role': 'user', 'content': 'x'}], 'mode': []},
    {'messages': [{'role': 'user', 'content': 'x'}], 'temperature': float('nan')},
    {'messages': [{'role': 'user', 'content': 'x'}], 'max_new_tokens': 1.5}])
def test_malformed_requests_are_400(server, payload):
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(server[0] + '/api/chat', payload)
    assert exc.value.code == 400


def test_cancel_and_local_urls(server):
    url, engine = server
    event = threading.Event()
    engine.requests['test'] = event
    with request(url + '/api/cancel', {'request_id': 'test'}) as response:
        assert json.load(response)['cancelled']
    assert event.is_set()
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(url + '/api/cancel', {'request_id': []})
    assert exc.value.code == 400
    assert network_urls('127.0.0.1', 8123) == ('http://127.0.0.1:8123', [])


def test_requested_checkpoint_never_silently_falls_back(tmp_path):
    from geocentric.checkpoint import _find_checkpoint
    (tmp_path / 'other.pt').touch()
    with pytest.raises(FileNotFoundError):
        _find_checkpoint(tmp_path, 'missing.pt')


@pytest.mark.parametrize('cancel,unicode', [(False, False), (True, False), (False, True)])
def test_stream_context_cancellation_and_unicode(monkeypatch, cancel, unicode):
    from types import SimpleNamespace
    import geocentric.sampling as sampling
    from geocentric.generate import stream_text
    class Tokenizer:
        def encode(self, text): return SimpleNamespace(ids=list(range(1, 41)))
        def token_to_id(self, token): return 0
        def decode(self, tokens, **kwargs):
            if unicode:
                return '\ufffd' if len(tokens) == 1 else 'é' + 'x' * (len(tokens) - 2)
            return 'x' * len(tokens)
    model = tiny()
    monkeypatch.setattr(sampling, 'sample_token', lambda *a, **k: torch.tensor([[5]]))
    stats = {}; event = threading.Event()
    if cancel: event.set()
    output = ''.join(stream_text(model, Tokenizer(), 'prompt', max_new_tokens=100,
                                cancel_event=event, stats=stats, loop_guard=not unicode))
    assert stats['prompt_tokens'] == 1
    assert stats['truncated_prompt_tokens'] == 39
    if cancel:
        assert output == '' and stats['finish_reason'] == 'cancelled'
    elif unicode:
        assert output.startswith('é') and '\ufffd' not in output
        assert stats['generated_tokens'] == 31
    else:
        assert output == 'xxxxxx' and stats['finish_reason'] == 'repetition'


def test_tokenizer_search_still_uses_extra_directories(tmp_path):
    from geocentric.checkpoint import find_tokenizer_path
    model_dir = tmp_path / 'weights'; model_dir.mkdir()
    extra = tmp_path / 'assets'; extra.mkdir()
    (extra / 'tokenizer.json').write_text('{}')
    assert find_tokenizer_path(model_dir, extra_dirs=[extra]) == extra / 'tokenizer.json'
