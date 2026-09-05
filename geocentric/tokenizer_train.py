from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List

from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

# Role tags are real vocabulary entries so a turn boundary costs one token instead
# of several, and so the model can never confuse them with user-typed text.
SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|eot|>",
    # Reserved now so a text-only tokenizer can be used for vision training later
    # without growing the embedding matrix. One vocabulary slot is a cheap option.
    "<|image|>",
]

# 8192 was far too small. At that size the tokenizer emits sub-word rubble, so the
# model spends most of its capacity relearning spelling and its effective context
# in characters is a fraction of its block size.
DEFAULT_VOCAB_SIZE = 32000


def batch_iterator(texts: Iterable[str], batch_size: int = 1000) -> Iterator[List[str]]:
    batch: List[str] = []
    for text in texts:
        text = text.strip()
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def train_byte_bpe_tokenizer(
    texts: Iterable[str],
    output_path: str | Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer from scratch and save tokenizer.json."""
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[AddedToken(t, special=True, normalized=False) for t in SPECIAL_TOKENS],
        show_progress=True,
    )

    batches = batch_iterator(texts)
    try:
        first_batch = next(batches)
    except StopIteration as exc:
        raise ValueError(
            "No training text was found for tokenizer training. Check --data_path."
        ) from exc

    def chained() -> Iterator[List[str]]:
        yield first_batch
        yield from batches

    tokenizer.train_from_iterator(chained(), trainer=trainer)
    tokenizer.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in SPECIAL_TOKENS])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def token_id(tokenizer: Tokenizer, token: str) -> int:
    tid = tokenizer.token_to_id(token)
    if tid is None:
        raise KeyError(
            f"Tokenizer does not contain required token: {token}. "
            "It was trained by an older Geocentric release — retrain it with `geocentric train-tokenizer`."
        )
    return int(tid)
