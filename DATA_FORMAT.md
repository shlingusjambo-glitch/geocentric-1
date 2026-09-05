# Data formats

## Pretraining

Point `--data_path` at a file or a folder. Folders are searched recursively for
`.txt`, `.md`, `.jsonl`, `.json`, and `.csv`.

### Plain text (`.txt`, `.md`)

Each file is treated as **one continuous document** and streamed in chunks. Training
windows are cut from the concatenated token stream and may span document
boundaries — that is what teaches long-range coherence.

If your file genuinely holds many separate documents, pass a separator so an `<eos>`
is placed at each real boundary:

```bash
geocentric pretrain --data_path corpus.txt --doc_sep $'\n\n\n'
```

Do not set `--doc_sep` to a single newline. That reproduces the 2.1 bug where every
line became its own end-of-sequence document.

### Structured records

One record is one document.

```jsonl
{"text": "A full document of prose."}
{"instruction": "Summarize photosynthesis.", "output": "Plants convert light..."}
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}]}
```

CSV with an `instruction`/`input`/`output`/`response`/`text`/`messages` header is
read by column; a headerless CSV joins each row's cells with newlines.

## Instruction tuning

`--sft_data_path` takes `.jsonl` or `.json`. Three shapes are accepted:

```jsonl
{"instruction": "What is 2+2?", "input": "", "output": "4."}
{"system": "Be terse.", "instruction": "Capital of Japan?", "output": "Tokyo."}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

Rules:

- Only assistant turns contribute to the loss; system and user turns are masked.
- The `<|eot|>` closing each assistant turn **is** supervised — that is what teaches
  the model to stop.
- Conversations longer than the context are dropped, not truncated. Truncation
  removes the stop token and trains run-on output. Override with `--keep_overlong`.
- Records missing an instruction or a response are skipped.

## Generated artifacts

`prepare` (and the first `pretrain`) writes into `<output_dir>/corpus/`:

| File | Contents |
|---|---|
| `train.bin` | Flat `uint16`/`uint32` token ids, memory-mapped during training |
| `train.meta.json` | Token count, dtype, vocab size |
| `val.bin` | Contiguous tail of the stream held out for evaluation |

The validation split is the **end** of the token stream, not random windows, so
documents cannot leak between train and eval. Delete the folder or pass
`--reprepare` to re-tokenize after changing the corpus or tokenizer.
