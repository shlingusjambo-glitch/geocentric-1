"""Numerical and persistence contracts for the training efficiency paths."""
import copy
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

from geocentric.checkpoint import save_checkpoint, load_checkpoint, load_optimizer_state
from geocentric.epicycle import EpicycleConfig, EpicycleScheduler, RingAdamW, equant_loss
from geocentric.model import GPTConfig, GeocentricGPT, KVCache
from geocentric.streaming_loss import linear_cross_entropy


def tiny():
    return GeocentricGPT(GPTConfig(vocab_size=97, block_size=32, n_layer=4,
                                  n_head=2, n_kv_head=1, n_embd=32))


@pytest.mark.parametrize("reduction", ["mean", "sum", "none", "equant"])
@pytest.mark.parametrize("chunk", [1, 7, 100])
def test_chunked_loss_and_all_gradients_match_dense(reduction, chunk):
    torch.manual_seed(4)
    a = tiny()
    b = copy.deepcopy(a)
    b.loss_chunk_size = chunk
    ids = torch.randint(0, 97, (2, 19))
    labels = torch.randint(0, 97, ids.shape)
    labels[0, :8] = -100
    r = "none" if reduction == "equant" else reduction
    _, dense = a(ids, labels=labels, loss_reduction=r)
    logits, streamed = b(ids, labels=labels, loss_reduction=r, return_logits=False)
    assert logits is None
    torch.testing.assert_close(dense, streamed, atol=1e-5, rtol=1e-5)
    if reduction == "equant":
        dense, streamed = (equant_loss(v, labels) for v in (dense, streamed))
    dense.sum().backward()
    streamed.sum().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)


def test_entirely_masked_chunked_loss_is_differentiable_zero():
    h = torch.randn(11, 16, requires_grad=True)
    w = torch.randn(23, 16, requires_grad=True)
    loss = linear_cross_entropy(h, w, torch.full((11,), -100), 4)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(h.grad) == torch.count_nonzero(w.grad) == 0


def test_chunked_path_does_not_save_vocabulary_activations():
    saved_shapes = []
    h = torch.randn(41, 16, requires_grad=True)
    w = torch.randn(97, 16, requires_grad=True)
    def pack(t):
        saved_shapes.append(tuple(t.shape))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = linear_cross_entropy(h, w, torch.zeros(41, dtype=torch.long), 7)
    assert not any(len(s) == 2 and s[-1] == 97 for s in saved_shapes)
    loss.backward()


@pytest.mark.parametrize("width,ctx", [(32, 8), (31, 8), (31, 31)])
def test_horizon_fold_preserves_every_input_target_pair(width, ctx):
    ids = torch.arange(2 * width).reshape(2, width)
    labels = ids + 1
    sched = EpicycleScheduler(EpicycleConfig.preset("speed"), tiny(), 100, width)
    x, y = sched.prepare_batch(ids, labels, ctx)
    valid = y != -100
    assert x.size(1) == ctx
    assert torch.equal(x[valid], ids.reshape(-1))
    assert torch.equal(y[valid], labels.reshape(-1))


def test_horizon_reaches_non_power_of_two_context():
    sched = EpicycleScheduler(EpicycleConfig(enabled=True, horizon_start=8), tiny(), 100, 31)
    assert sched.context_length(50) == 31


def test_initial_stack_keeps_initialization_and_full_eval_matches_active():
    model = tiny().eval()
    weight = model.blocks[0].attn.proj.weight.detach().clone()
    EpicycleScheduler(EpicycleConfig.preset("speed"), model, 100, 32)
    assert torch.equal(model.blocks[0].attn.proj.weight, weight)
    ids = torch.randint(0, 97, (2, 13))
    active = model(ids)[0]
    model.active_layers = None
    torch.testing.assert_close(model(ids)[0], active)


@pytest.mark.parametrize("factored", [False, True])
def test_ring_resume_matches_uninterrupted_updates_and_storage(factored):
    torch.manual_seed(8)
    a = tiny()
    opt = RingAdamW(a.parameters(), rings=3, dwell=2, factored=factored)
    ids = torch.randint(0, 97, (2, 12))
    def update(model, optimizer):
        model(ids, labels=ids)[1].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    for _ in range(5):
        update(a, opt)
    b = copy.deepcopy(a)
    resumed = RingAdamW(b.parameters(), rings=3, dwell=2, factored=factored)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert opt.hot_ring == resumed.hot_ring
    assert opt.state_bytes() == resumed.state_bytes()
    for _ in range(6):
        update(a, opt)
        update(b, resumed)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert all(s["v"].dtype == torch.bfloat16 for s in resumed.state.values() if "v" in s)


def test_ring_frees_cold_momentum_even_without_gradient():
    params = [torch.nn.Parameter(torch.ones(8, 8)) for _ in range(2)]
    opt = RingAdamW(params, rings=2, dwell=1)
    for p in params:
        p.grad = torch.ones_like(p)
    opt.step()
    old = next(p for p in params if "m" in opt.state[p])
    old.grad = None
    opt.step()
    assert "m" not in opt.state[old]


def test_factored_state_is_smaller_and_fits_rank_one_second_moment():
    p = torch.nn.Parameter(torch.zeros(32, 64))
    grad = torch.arange(1, 33)[:, None] * torch.arange(1, 65)[None, :]
    opt = RingAdamW([p], rings=1, factored=True, weight_decay=0, lr=0.01)
    p.grad = grad.float()
    opt.step()
    # A separable squared gradient is represented exactly by row/column moments.
    torch.testing.assert_close(p, torch.full_like(p, -0.01), atol=1e-7, rtol=1e-5)
    assert opt.state_bytes() == 4 * (32 + 64 + p.numel())


def test_deferent_checkpoint_resume_does_not_erase_learning(tmp_path):
    model = tiny()
    config = EpicycleConfig.preset("speed")
    sched = EpicycleScheduler(config, model, 100, 32)
    sched.apply(25)
    opt = RingAdamW(model.parameters(), rings=2, dwell=3)
    ids = torch.randint(0, 97, (2, 12))
    for _ in range(3):
        model(ids, labels=ids)[1].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    model._training_tokens = 72
    save_checkpoint(model, tmp_path, 25, optimizer=opt)
    loaded = load_checkpoint(tmp_path, torch.device("cpu"))
    restored_opt = RingAdamW(loaded.parameters(), rings=2, dwell=3)
    assert load_optimizer_state(tmp_path, "model.pt", restored_opt) == 25
    resumed = EpicycleScheduler(config, loaded, 100, 32, start_step=25)
    torch.testing.assert_close(model(ids)[0], loaded(ids)[0], rtol=0, atol=0)
    assert loaded._training_tokens == 72
    sched.apply(50)
    resumed.apply(50)
    torch.testing.assert_close(model(ids)[0], loaded(ids)[0], rtol=0, atol=0)


def test_multitoken_cached_continuation_matches_full_forward():
    model = tiny().eval()
    ids = torch.randint(0, 97, (2, 17))
    caches = [KVCache() for _ in model.blocks]
    full = model(ids)[0][:, -1:]
    model(ids[:, :10], caches=caches)
    cached = model(ids[:, 10:], caches=caches, position_offset=10)[0]
    torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)


def test_evaluation_is_weighted_by_supervised_tokens():
    from geocentric.train_pretrain import evaluate
    model = tiny()
    batches = []
    total, count = 0, 0
    for width in [3, 17]:
        ids = torch.randint(0, 97, (1, width))
        labels = torch.randint(0, 97, ids.shape)
        labels[0, 0] = -100
        total += model(ids, labels=labels, loss_reduction="sum")[1].item()
        count += width - 1
        batches.append(dict(input_ids=ids, labels=labels))
    assert evaluate(model, batches, torch.device("cpu"), nullcontext()) == pytest.approx(total / count)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Silicon")
def test_mps_bf16_chunked_backward_matches_dense():
    torch.manual_seed(31)
    a = tiny().to("mps")
    b = copy.deepcopy(a)
    b.loss_chunk_size = 7
    ids = torch.randint(0, 97, (2, 17), device="mps")
    with torch.autocast("mps", dtype=torch.bfloat16):
        dense = a(ids, labels=ids)[1]
        chunked = b(ids, labels=ids, return_logits=False)[1]
    dense.backward()
    chunked.backward()
    torch.testing.assert_close(dense, chunked, atol=2e-3, rtol=2e-3)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=5e-3, rtol=5e-2)
