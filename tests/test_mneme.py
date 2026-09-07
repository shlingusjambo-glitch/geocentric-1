import json
import copy

import pytest
import torch

from geocentric.epicycle import EpicycleConfig, mneme_loss
from geocentric.mneme import ABSTENTION, evidence_pairs, prepare_grounding_data, score_grounded_response
from geocentric.model import GeocentricGPT, GPTConfig


def test_mneme_balances_rare_targets_and_ignores_masked_rows():
    labels = torch.tensor([1, 1, 1, 1, 2, -100])
    losses = torch.ones(6, requires_grad=True)
    loss = mneme_loss(losses, labels, 3)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(1.))
    assert losses.grad[4] > losses.grad[0] > 0
    assert losses.grad[-1] == 0


@pytest.mark.parametrize('device', ['cpu', 'mps', 'cuda'])
def test_mneme_weighted_reference_gradient(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    labels = torch.tensor([1] * 25 + [2, 3, -100], device=device)
    losses = torch.linspace(.1, 2., 28, device=device, requires_grad=True)
    actual = mneme_loss(losses, labels, 5)
    weights = torch.tensor([1.] * 25 + [4., 4., 0.], device=device)
    expected = .5 * (losses[:-1].mean() + (losses * weights).sum() / weights.sum())
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual, losses, retain_graph=True)[0]
    b = torch.autograd.grad(expected, losses)[0]
    torch.testing.assert_close(a, b)


def test_mneme_uniform_and_all_masked():
    losses = torch.randn(4, requires_grad=True)
    torch.testing.assert_close(mneme_loss(losses, torch.arange(4), 4), losses.mean())
    zero = mneme_loss(losses, torch.full((4,), -100), 4)
    zero.backward()
    assert zero.item() == 0 and losses.grad.count_nonzero() == 0
    assert EpicycleConfig.from_dict({}).mneme is False
    assert EpicycleConfig.preset('knowledge').mneme
    assert EpicycleConfig(mneme=True).equant


def test_mneme_equant_preserves_outlier_trim_and_easy_token_signal():
    losses = torch.arange(100, dtype=torch.float32, requires_grad=True)
    labels = torch.arange(100) % 7
    loss = mneme_loss(losses, labels, 7, equant_keep=.65, equant_trim=.02)
    loss.backward()
    assert losses.grad[-2:].count_nonzero() == 0
    assert (losses.grad[:-2] > 0).all()


@pytest.mark.parametrize('device', ['cpu', 'mps', 'cuda'])
def test_mneme_equant_chunked_tied_model_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    torch.manual_seed(97)
    a = GeocentricGPT(GPTConfig(vocab_size=41, block_size=32, n_layer=2,
                                n_head=2, n_embd=32, gradient_checkpointing=True)).to(device)
    b = copy.deepcopy(a); b.loss_chunk_size = 7
    x = torch.randint(0, 41, (2, 32), device=device)
    labels = x.roll(-1, 1); labels[:, :4] = -100
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    with torch.autocast(device, dtype=dtype, enabled=device != 'cpu'):
        la = mneme_loss(a(x, labels=labels, return_logits=False, loss_reduction='none')[1], labels, 41, .65)
        lb = mneme_loss(b(x, labels=labels, return_logits=False, loss_reduction='none')[1], labels, 41, .65)
    la.backward(); lb.backward()
    torch.testing.assert_close(la, lb, rtol=.003, atol=.003)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=.05, atol=.008)


def test_grounding_pairs_and_atomic_rejection(tmp_path):
    record = {'question': 'What is the capital of France?', 'answer': 'Paris',
              'context': 'Paris is the capital of France.'}
    pairs = list(evidence_pairs(record))
    assert pairs[0]['messages'][-1]['content'] == 'Paris'
    assert pairs[1]['messages'][-1]['content'] == ABSTENTION
    assert 'Paris' not in pairs[1]['messages'][1]['content']
    source, output = tmp_path/'facts.jsonl', tmp_path/'ground.jsonl'
    source.write_text(json.dumps(record) + '\n')
    assert prepare_grounding_data(source, output) == 2
    assert len(output.read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        prepare_grounding_data(source, output)
    source.write_text(json.dumps(record) + '\n' + json.dumps({**record, 'answer': 'London'}))
    with pytest.raises(ValueError, match='verbatim'):
        prepare_grounding_data(source, tmp_path/'bad.jsonl')
    assert not (tmp_path/'bad.jsonl').exists()
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('context', ['', 'The document discusses rivers, not capitals.',
                                     'The value is 12. Another entry says the value is 19.'])
def test_annotated_unanswerable_contexts_and_strict_scoring(context):
    record = {'question': 'What is the value?', 'context': context, 'answerable': False}
    pair, = evidence_pairs(record)
    assert pair['messages'][-1]['content'] == ABSTENTION
    assert score_grounded_response(ABSTENTION, record)['exact_match']
    assert not score_grounded_response(ABSTENTION + ' But the value is 42.', record)['exact_match']
    with pytest.raises(ValueError, match='omit'):
        list(evidence_pairs({**record, 'answer': '12'}))


def test_strict_evidence_never_invents_and_rejects_conflict():
    from geocentric.mneme import EvidenceRegistry
    record = {'question': 'What is the value?', 'context': 'The value is 12.', 'answer': '12'}
    registry = EvidenceRegistry([record])
    assert registry.answer(' WHAT is the value? ')['answer'] == '12'
    assert registry.answer('What is another value?')['status'] == 'abstained'
    conflict = EvidenceRegistry([record, {**record, 'answer': '19', 'context': 'The value is 19.'}])
    assert conflict.answer(record['question'])['status'] == 'abstained'
    unknown = EvidenceRegistry([record, {'question': record['question'], 'context': '', 'answerable': False}])
    assert unknown.answer(record['question'])['status'] == 'abstained'
