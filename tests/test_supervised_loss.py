import copy

import pytest
import torch

from geocentric.model import GeocentricGPT, GPTConfig
from geocentric.streaming_loss import linear_cross_entropy, supervised_cross_entropy


@pytest.mark.parametrize('reduction', ['sum', 'mean'])
@pytest.mark.parametrize('masked', [0, 17, 24])
def test_supervised_projection_matches_loss_and_gradients(reduction, masked):
    torch.manual_seed(17)
    h = torch.randn(2, 12, 16, requires_grad=True)
    w = torch.randn(37, 16, requires_grad=True)
    other_h = h.detach().clone().requires_grad_()
    other_w = w.detach().clone().requires_grad_()
    labels = torch.randint(0, 37, (2, 12))
    labels.view(-1)[:masked] = -100
    a = linear_cross_entropy(h, w, labels, 7, reduction)
    b = supervised_cross_entropy(other_h, other_w, labels, 7, reduction)
    a.backward(); b.backward()
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(h.grad, other_h.grad, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(w.grad, other_w.grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('device,dtype', [('cpu', torch.bfloat16), ('mps', torch.bfloat16), ('cuda', torch.float16)])
def test_checkpointed_tied_model_supervised_gradients(device, dtype):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(7)
    a = GeocentricGPT(GPTConfig(vocab_size=97, block_size=32, n_layer=2,
                                n_head=2, n_embd=32, gradient_checkpointing=True)).to(device)
    a.loss_chunk_size = 8
    b = copy.deepcopy(a); b.loss_supervised_only = True
    x = torch.randint(0, 97, (2, 32), device=device)
    labels = x.clone(); labels[:, :20] = -100
    with torch.autocast(device, dtype=dtype):
        la = a(x, labels=labels, return_logits=False, loss_reduction='sum')[1]
        lb = b(x, labels=labels, return_logits=False, loss_reduction='sum')[1]
    (la / 24).backward(); (lb / 24).backward()
    torch.testing.assert_close(la, lb, atol=.1, rtol=.002)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=.008, rtol=.04)
