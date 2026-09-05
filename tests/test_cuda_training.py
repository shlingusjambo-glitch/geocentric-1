"""CUDA integration tests: skipped explicitly when no NVIDIA GPU is accessible.

Run scripts/validate_cuda.sh on the target machine; it fails before pytest when CUDA
is absent, so a green all-skipped run can never be mistaken for GPU validation.
"""
import copy

import pytest
import torch

from geocentric.device import resolve_dtype, supports_bf16
from geocentric.epicycle import EpicycleConfig, EpicycleScheduler, RingAdamW
from geocentric.model import GPTConfig, GeocentricGPT
from geocentric.trainer import build_optimizer, find_batch_size


def test_turing_auto_dtype_uses_fp16_even_when_bf16_is_emulated(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation: including_emulation)
    assert not supports_bf16(torch.device("cuda"))
    assert resolve_dtype(torch.device("cuda")) == torch.float16


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def make_model():
    model = GeocentricGPT(GPTConfig(vocab_size=256, block_size=64, n_layer=4,
                                   n_head=4, n_kv_head=2, n_embd=64)).cuda()
    model.loss_chunk_size = 17
    return model


@cuda
@pytest.mark.parametrize("kind", ["adamw", "ring", "factored"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_cuda_fp16_growth_fold_scaling_and_resume(kind, checkpointing):
    torch.manual_seed(7)
    model = make_model()
    for block in model.blocks:
        block.gradient_checkpointing = checkpointing
    scheduler = EpicycleScheduler(EpicycleConfig(enabled=True, horizon_start=16), model, 10, 64)
    optimizer = (build_optimizer(model, 1e-3, quiet=True) if kind == "adamw" else
                 RingAdamW(model.parameters(), lr=1e-3, rings=2, dwell=2, factored=kind == "factored"))
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    x = torch.randint(0, 256, (2, 64), device="cuda")
    y = torch.roll(x, -1, 1)
    initial = model.token_embedding.weight.detach().clone()
    for step in range(8):
        _, ctx = scheduler.apply(step)
        ids, labels = scheduler.prepare_batch(x, y, ctx)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(ids, labels=labels, return_logits=False)[1]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        scaler.step(optimizer)
        scaler.update()
        if step == 3:
            saved = copy.deepcopy(optimizer.state_dict())
            optimizer.load_state_dict(saved)
            scaler.load_state_dict(copy.deepcopy(scaler.state_dict()))
    assert not torch.equal(initial, model.token_embedding.weight)
    assert all(p.dtype == torch.float32 for p in model.parameters())


@cuda
def test_cuda_chunked_loss_matches_dense_fp16():
    a = make_model()
    b = copy.deepcopy(a)
    ids = torch.randint(0, 256, (2, 33), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        dense = a(ids, labels=ids)[1]
        chunked = b(ids, labels=ids, return_logits=False)[1]
    dense.backward()
    chunked.backward()
    torch.testing.assert_close(dense, chunked, atol=1e-3, rtol=1e-3)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-3, rtol=2e-2)


@cuda
def test_cuda_batch_probe_preserves_weights_rng_and_uses_requested_optimizer():
    model = make_model()
    before = {k: p.clone() for k, p in model.state_dict().items()}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    called = []
    def factory():
        called.append(True)
        return RingAdamW(model.parameters(), rings=2, factored=True)
    batch = find_batch_size(model, 32, torch.device("cuda"), torch.float16,
                            max_batch=3, probe_steps=2, optimizer_factory=factory)
    assert batch == 3 and called
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    for name, p in model.state_dict().items():
        assert torch.equal(before[name], p)


@cuda
def test_cuda_compile_chunked_backward():
    model = make_model()
    compiled = torch.compile(model)
    ids = torch.randint(0, 256, (2, 32), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = compiled(ids, labels=ids, return_logits=False)[1]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
