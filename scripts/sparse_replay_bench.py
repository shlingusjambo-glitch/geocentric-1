"""Matched full-model EQUANT updates; synthetic tokens measure systems, not quality."""
import argparse, copy, json, statistics, time
from pathlib import Path
import torch
from geocentric.model import GeocentricGPT,GPTConfig
from geocentric.epicycle import equant_loss
from geocentric.device import select_device,resolve_dtype


def sync(d):
    if d.type=='cuda':torch.cuda.synchronize()
    elif d.type=='mps':torch.mps.synchronize()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='auto');p.add_argument('--steps',type=int,default=12)
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--output',default='runs/sparse-replay.json')
    args=p.parse_args()
    if args.steps < 1 or args.warmup < 0:
        p.error('steps must be positive and warmup nonnegative')
    d=select_device() if args.device=='auto' else torch.device(args.device)
    dtype=resolve_dtype(d) if d.type!='cpu' else torch.float32
    torch.manual_seed(2026)
    config=GPTConfig(vocab_size=16000,block_size=256,n_layer=4,n_head=4,n_kv_head=2,n_embd=256)
    original=GeocentricGPT(config).to(d);original.loss_chunk_size=256
    models={'checkpointed':original,'sparse_replay':copy.deepcopy(original)}
    models['sparse_replay'].loss_sparse_replay=True
    optimizers={k:torch.optim.AdamW(m.parameters(),lr=1e-4) for k,m in models.items()}
    scalers={k:torch.amp.GradScaler('cuda',enabled=d.type=='cuda' and dtype==torch.float16) for k in models}
    timings={k:[] for k in models};losses={k:[] for k in models}
    for step in range(args.steps+args.warmup):
        x=torch.randint(0,16000,(4,256),device=d);y=torch.roll(x,-1,1)
        order=list(models) if step%2==0 else list(reversed(models))
        for name in order:
            model,opt,scaler=models[name],optimizers[name],scalers[name]
            opt.zero_grad(set_to_none=True);sync(d);started=time.perf_counter()
            with torch.autocast(d.type,dtype=dtype,enabled=dtype!=torch.float32):
                per=model(x,labels=y,return_logits=False,loss_reduction='none')[1]
                loss=equant_loss(per,y)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();sync(d)
            elapsed=time.perf_counter()-started
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite benchmark loss')
            if step>=args.warmup:timings[name].append(elapsed);losses[name].append(float(loss.detach()))
    rates={k:1024/statistics.median(v) for k,v in timings.items()}
    payload=dict(device=str(d),dtype=str(dtype),torch=torch.__version__,config=vars(config),
                 steps=args.steps,warmup=args.warmup,seconds=timings,losses=losses,
                 tokens_per_second=rates,improvement_percent=100*(rates['sparse_replay']/rates['checkpointed']-1),
                 caveat='Synthetic full-model updates. Alternating order, identical initialization and batches. No capability claim; CUDA needs separate measurement.')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in {'seconds','losses','config'}},indent=2))

if __name__=='__main__':main()
