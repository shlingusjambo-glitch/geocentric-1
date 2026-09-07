import pickle

import pytest
import torch

from geocentric.data import pad_collate
from geocentric.model import GeocentricGPT, GPTConfig, KVCache
from geocentric.sft_storage import DiskExamples


@pytest.mark.parametrize('batch_size', [1, 4])
def test_collation_preserves_tokens_and_mask(batch_size):
    batch = [{'input_ids': torch.arange(i + 2), 'labels': torch.arange(i + 2) - 100}
             for i in range(batch_size)]
    result = pad_collate(batch, 99)
    for row, item in enumerate(batch):
        n = item['input_ids'].numel()
        for key in item:
            torch.testing.assert_close(result[key][row, :n], item[key])
        assert (result['input_ids'][row, n:] == 99).all()
        assert (result['labels'][row, n:] == -100).all()


def test_mapped_cache_worker_pickle_and_independent_samples(tmp_path):
    cache = DiskExamples(tmp_path, 'test')
    example = {'input_ids': torch.arange(7), 'labels': torch.arange(7) - 100}
    cache.append(example); cache.finish()
    sample = cache[0]
    sample['input_ids'].zero_()
    torch.testing.assert_close(cache[0]['input_ids'], example['input_ids'])
    restored = pickle.loads(pickle.dumps(cache))
    assert restored._mapping is None
    torch.testing.assert_close(restored[-1]['labels'], example['labels'])
    loader = torch.utils.data.DataLoader(restored, batch_size=1, num_workers=1,
                                         multiprocessing_context='spawn')
    torch.testing.assert_close(next(iter(loader))['input_ids'][0], example['input_ids'])


@torch.no_grad()
@pytest.mark.parametrize('limit', [0, 1, 8, 40])
def test_buffered_generation_matches_reference(limit):
    torch.manual_seed(92)
    model = GeocentricGPT(GPTConfig(vocab_size=37, block_size=24, n_layer=2,
                                    n_head=2, n_embd=32)).eval()
    prompt = torch.randint(0, 37, (2, 5))
    original = prompt.clone()
    actual = model.generate(prompt, limit, temperature=0, repetition_penalty=1)
    reference = prompt.clone()
    caches = [KVCache(max_length=24) for _ in model.blocks]
    cur, offset = prompt, 0
    for _ in range(min(limit, 24 - prompt.size(1))):
        logits, _ = model(cur, caches=caches, position_offset=offset)
        offset += cur.size(1)
        cur = logits[:, -1].argmax(-1, keepdim=True)
        reference = torch.cat((reference, cur), dim=1)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(prompt, original)
