import copy
import pytest
import torch
from geocentric.streaming_loss import linear_cross_entropy, sparse_replay_cross_entropy
from geocentric.epicycle import equant_loss, EpicycleConfig
from geocentric.model import GeocentricGPT, GPTConfig


@pytest.mark.parametrize('chunk',[1,7,64])
@pytest.mark.parametrize('mode',['equant','weighted','masked'])
def test_loss_and_first_order_gradients_match_checkpointed_ce(chunk,mode):
    torch.manual_seed(7)
    h=torch.randn(2,19,16,requires_grad=True)
    w=torch.randn(53,16,requires_grad=True)
    h2=h.detach().clone().requires_grad_(); w2=w.detach().clone().requires_grad_()
    y=torch.randint(0,53,(2,19));y[0,:3]=-100
    if mode=='masked': y.fill_(-100)
    a=linear_cross_entropy(h,w,y,chunk,'none')
    b=sparse_replay_cross_entropy(h2,w2,y,chunk)
    torch.testing.assert_close(a,b)
    if mode=='equant':
        a=equant_loss(a,y);b=equant_loss(b,y)
    else:
        scale=torch.zeros(38);scale[::3]=torch.linspace(-2,2,len(scale[::3]))
        a=(a*scale).sum();b=(b*scale).sum()
    a.backward();b.backward()
    torch.testing.assert_close(h.grad,h2.grad,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(w.grad,w2.grad,atol=2e-6,rtol=2e-5)


def test_only_nonzero_unmasked_rows_are_replayed(monkeypatch):
    import geocentric.streaming_loss as module
    real=module.F.linear; rows=[]
    def measured(h,w,*args,**kwargs):
        if torch.is_grad_enabled():rows.append(len(h))
        return real(h,w,*args,**kwargs)
    monkeypatch.setattr(module.F,'linear',measured)
    h=torch.randn(31,8,requires_grad=True);w=torch.randn(29,8,requires_grad=True)
    y=torch.arange(31)%29;y[2]=-100
    loss=sparse_replay_cross_entropy(h,w,y,8)
    (loss[:5].sum()*128).backward()
    assert sum(rows)==4


@pytest.mark.parametrize('device,dtype',[('cpu',torch.bfloat16),('mps',torch.bfloat16),('cuda',torch.float16)])
def test_autocast_tied_model_gradients(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS required')
    torch.manual_seed(11)
    a=GeocentricGPT(GPTConfig(vocab_size=97,block_size=32,n_layer=2,n_head=2,n_embd=32)).to(device)
    b=copy.deepcopy(a);a.loss_chunk_size=b.loss_chunk_size=8;b.loss_sparse_replay=True
    x=torch.randint(0,97,(2,32),device=device);y=torch.roll(x,-1,1)
    with torch.autocast(device,dtype=dtype):
        la=a(x,labels=y,return_logits=False,loss_reduction='none')[1]
        lb=b(x,labels=y,return_logits=False,loss_reduction='none')[1]
        aa=equant_loss(la,y)*128;bb=equant_loss(lb,y)*128
    torch.testing.assert_close(la,lb)
    aa.backward();bb.backward()
    for p,q in zip(a.parameters(),b.parameters()):
        assert torch.isfinite(q.grad).all()
        torch.testing.assert_close(p.grad,q.grad,rtol=.04,atol=.10)


def test_preset_is_opt_in_and_old_config_loads():
    assert not EpicycleConfig.from_dict({'enabled':True,'equant':True}).equant_sparse_replay
    assert not EpicycleConfig.preset('quality').equant_sparse_replay
    assert EpicycleConfig.preset('selective').equant_sparse_replay


def test_compile_path_uses_checkpointed_fallback(monkeypatch):
    import geocentric.streaming_loss as module
    monkeypatch.setattr(torch.compiler, 'is_compiling', lambda: True)
    h=torch.randn(5,8,requires_grad=True);w=torch.randn(13,8,requires_grad=True)
    labels=torch.arange(5)
    expected=linear_cross_entropy(h,w,labels,3,'none')
    # Confirm routing directly without depending on an installed compiler backend.
    seen=[]
    real=module.linear_cross_entropy
    def record(*args,**kwargs):
        seen.append(True)
        return real(*args,**kwargs)
    monkeypatch.setattr(module,'linear_cross_entropy',record)
    actual=sparse_replay_cross_entropy(h,w,labels,3)
    torch.testing.assert_close(actual,expected)
    assert seen==[True]
    actual.sum().backward()
    assert torch.isfinite(h.grad).all() and torch.isfinite(w.grad).all()
