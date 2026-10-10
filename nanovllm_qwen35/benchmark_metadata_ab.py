"""Interleaved unprofiled A/B experiment for batch-local metadata reuse only."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from types import MethodType
from unittest.mock import patch
import torch
from nanovllm import SamplingParams
from nanovllm.qwen35 import Qwen35LLM
from benchmark_gpu_baseline import run_case, command


def main():
    output = Path('metadata_reuse_ab_20261010.json')
    telemetry = output.with_suffix('.gpu.csv')
    if output.exists() or telemetry.exists():
        raise RuntimeError('Refusing to overwrite previous results')
    source = Path('../archive/metadata-reuse-20261010/pool_backend_before.py').read_text()
    namespace = {'__name__':'qwen35_adapter.metadata_ab_old', '__package__':'qwen35_adapter'}
    exec(compile(source, '<old-pool-backend>', 'exec'), namespace)
    old_forward = namespace['PooledCPUBackend'].forward
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    report = {'status':'running', 'description':'same engine, L64, N32, B1/2; 2 warmups per variant; 6 interleaved pairs AB/BA',
        'before_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'after_sha256':hashlib.sha256(Path('qwen35_adapter/pool_backend.py').read_bytes()).hexdigest(),
        'gpu_before':command(['nvidia-smi','-i','2','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv']), 'cases':[]}
    engine = None
    with telemetry.open('x') as monitor_file:
        monitor = subprocess.Popen(['nvidia-smi','-i','2','--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,temperature.gpu','--format=csv','--loop=1'],stdout=monitor_file)
        try:
            engine = Qwen35LLM('/home/ubuntu/huggingface/Qwen3.5-4B',max_model_len=512,max_num_seqs=2,block_size=256,greedy=True)
            forwards = {'A':MethodType(old_forward, engine.backend), 'B':engine.backend.forward}
            seed = engine.tokenizer('Explain how a key value cache accelerates language model inference.',add_special_tokens=False)['input_ids']
            prompts = [(seed*10)[i:i+64] for i in range(2)]
            for batch in (1,2):
                for variant in ('A','B','B','A'):
                    with patch.object(engine.backend,'forward',forwards[variant]):
                        run_case(engine,prompts[:batch],32,torch,SamplingParams)
                case = {'batch':batch,'runs':[]}
                for pair in range(6):
                    pair_runs = {}
                    for variant in (('A','B') if pair%2==0 else ('B','A')):
                        with patch.object(engine.backend,'forward',forwards[variant]):
                            run = run_case(engine,prompts[:batch],32,torch,SamplingParams)
                        case['runs'].append({'pair':pair,'variant':variant,**run})
                        pair_runs[variant] = run
                    assert [r['output_ids'] for r in pair_runs['A']['requests']] == [r['output_ids'] for r in pair_runs['B']['requests']]
                report['cases'].append(case)
                output.write_text(json.dumps(report,indent=2))
                for variant in ('A','B'):
                    rows = [r for r in case['runs'] if r['variant']==variant]
                    print(f'B{batch} {variant}: output {statistics.median(r["output_tokens_per_s"] for r in rows):.3f} tok/s',flush=True)
            report['status']='complete'
        except BaseException as exc:
            report['status']='failed'; report['error']=repr(exc)
            raise
        finally:
            monitor.terminate(); monitor.wait(timeout=10)
            report['gpu_after_measurement']=command(['nvidia-smi','-i','2','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv'])
            output.write_text(json.dumps(report,indent=2))
            if engine is not None and not engine.runner.poisoned and not engine.states.poisoned and engine.scheduler._pending is None:
                engine.close()


if __name__=='__main__':
    main()
