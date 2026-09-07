# SFT reliability and performance audit

This pass retains the bounded data-prefetch queue, checkpoint formats, training
objective and RTX 2060 memory safeguards. It does not increase batch size or remove
activation checkpointing to inflate GPU utilization.

## Fixed failure cases

1. Prefetch producers could block forever on a full queue after the consumer
   stopped. Cancellation now interrupts queue puts; SFT explicitly closes the
   iterator on completion, errors, loss-guard stops and interruption. Shutdown
   waits at most one second for the worker, because external file I/O can block
   independently of the queue. Prefetch stays bounded and ordered.
2. Runtime failures could leave watcher status at "running". They now record
   "failed" and retain the original exception even if recording status also fails.
3. Non-finite gradient updates advanced the optimizer-step counter and trained-token
   count despite skipping the update. Rejected updates no longer advance those
   counters or the learning-rate schedule.
4. An entire epoch with no successful optimizer updates could be saved as completed.
   It now fails explicitly, without replacing the checkpoint with a claimed success.
5. Invalid validation batches could be silently ignored, allowing a biased result
   or a zero-loss "best" checkpoint. Non-finite validation now fails; an empty or
   fully masked validation set also fails rather than returning zero.
6. Validation always switched the model back to training mode and failed to restore
   mode on exceptions. It now restores the caller's original mode in a finally block.
7. Best-checkpoint resume metadata could contain the previous best score and a stale
   data position. It is refreshed before saving the new best checkpoint.
8. Setting eval_ratio=0 still reserved a validation example in datasets over 50
   examples. Zero now disables the validation split as requested.
9. log_every=0 crashed after training started. Invalid logging intervals, evaluation
   ratios, non-finite/nonpositive learning rates and clipping thresholds now fail
   before checkpoint loading or training.
10. Wrong-type fields in a cache index could crash cache loading. Invalid offset
    lists and dropped-example counts now trigger a cache rebuild.
11. A non-object metrics JSON could crash SFT startup. SFT treats it as invalid
    prior metrics and initializes usable metrics.
12. Metrics were overwritten in place, exposing partial JSON to the watcher or
    leaving it after an interrupted write. They now use a temporary file and atomic
    replacement; failed writes leave the previous document intact. This is atomic
    visibility, not a guarantee against storage-device failure or power loss.

## Performance changes

- CPU/CUDA gradients are grouped by device and dtype for foreach normalization,
  replacing a separate Python-dispatched multiplication per parameter. Other
  backends and sparse gradients retain the scalar fallback.
- Backward is enqueued before reading the loss on the host. This removes the host
  barrier between forward and backward; non-finite losses still discard the whole
  accumulated gradient window before any optimizer step.
- No extra synchronization is introduced for measuring GPU utilization. Sustained
  input and supervised-token throughput are more useful than targeting a particular
  utilization percentage. Logging, checkpoint I/O and variable conversation lengths
  can make utilization fluctuate.

There is no measured RTX 2060 speedup for this pass: CUDA is unavailable on the local
development machine. The user's roughly 7,000 tokens/s is the current hardware
baseline, not a result attributed to these changes. Compare the same run, checkpoint,
batch settings and logging/save intervals over a sustained interval after updating.

## Validation

296 Python tests passed; 15 hardware/optional tests were skipped. Tests cover a full
prefetch queue followed by cancellation, producer errors and ordering, mixed-dtype
gradient normalization, failed metric replacement, malformed cache indexes, invalid
validation, failure status, rejected updates and best-checkpoint resume metadata.
Existing interrupted-versus-uninterrupted training tests still match weights and
optimizer state exactly. CUDA normalization has a hardware-gated regression test.
Only tiny synthetic tests were run; no long training job was started.
