# Triton Paged Decode 接入实验（2026-10-15）

## 结论

已增加可选 `decode_backend='triton_paged'`，默认仍为 `dense`。现有 `flash_paged` 保留。
通过本次合成输入与真实模型的探索性正确性门槛，但不是逐位等价或模型质量认证。
整步收益较小且并非所有配置获益，当前手写实现不替代 FlashAttention。

## 数据流与教学重点

1. `Qwen35Engine` 选择后端，`RunnerInputBuilder` 在CPU验证块归属、有效长度与slot。
2. `PooledCPUBackend.forward`（GPU派生类复用）每批上传一次元数据；Prefill仍走原路径。
3. Full层取得本层KV池、当前请求块表；Q/K Norm及RoPE仍在原来的位置执行。
4. `index_copy_` 在当前stream写入本轮K/V，随后调用 `paged_decode_trusted`。
5. Triton直接读取分页KV，输出 `[1,Hq,D]`；继续原有gate、o_proj与残差路径。
6. GDN、采样、Scheduler结算不变。没有重新初始化历史状态，没有再次写历史KV。

接入版核心位于 `nanovllm_qwen35/qwen35_adapter/triton_paged.py`。教学文件保留独立安全包装器；接入版只有不触发同步的shape/layout检查，值与所有权检查依赖Runner，不接受外部未经验证的metadata。
两份kernel使用相同算法；本次BF16合成配置两者输出逐元素一致。后续修改任一实现都应重新对拍，不能假设它们自动保持一致。

重要限制：模型仍然在每层内部逐请求调用。B=4真实引擎不是一次grid=(4,16)，而是四次grid=(1,16)；独立算子B=4才是一次批量调用。没有实现全层批量化、split-K、GQA跨Q头复用或引擎CUDA Graph。

## 实验条件与口径

- NVIDIA L40物理GPU 7，开始/结束均约292 MiB、利用率0%；其他卡运行其他任务，共享主机CPU仍可能带来抖动。未做全过程设备独占审计。
- Torch 2.5.1+cu124，Triton 3.1.0；Python环境nanovllm。单卡BF16 Qwen3.5-4B，FP32 softmax累积；TF32关闭，CPU线程2。
- 基础Git HEAD：`7b733033fd833238c7d8f4ba4dd8bd13a684c7ac`，加本次未提交接入改动。
- 模型最大长度512、最大请求4、页256；LM Head固定row；不启用chunked prefill。
- 独立算子：同一份已量化QKV，Hq16/Hkv4/D256，非连续页、无效KV填NaN。预热10次，5轮，每轮50次，交替顺序。
- 模型计时：每请求固定输出8 token、ignore_eos；各配置/后端预热2轮，再6轮轮换顺序。测量不含模型加载、tokenization、编译、诊断快照或profiler。
- TTFT从入队前到首token结算；TPOT=(最后token时间-首token时间)/7；总输出吞吐=B×8/总时间。不是在线服务SLA。
- 峰值显存是Torch allocated，包含权重、池、临时张量，不是KV逻辑字节；allocator保留在同一进程。

## 正确性：先隔离，再真实模型

合成用例B=1/4、L=128/1024、tile=8/16/32，接入Triton、教学Triton与CPU FP64参考（舍入至BF16）对拍，阈值atol=.008/rtol=.01。接入版与教学版逐元素一致；Flash/dense也通过参考阈值。KV逐字节未修改。

真实模型使用两请求，L=19/257，每次完整Prefill加3步Decode；各后端强制喂相同后续token，避免生成轨迹变化混淆。
捕获所有层输出、GDN conv/recurrent、逻辑顺序KV与logits；预先设相对L2误差3%作为探索性门槛。实际Triton对dense最坏值：

| 项目 | L=19相对L2 | L=257相对L2 |
|---|---:|---:|
| 层输出 | 2.3633% | 1.5143% |
| logits | 1.0674% | 1.0720% |
| 逻辑KV | 0.3605% | 0.0934% |
| conv | 1.3595% | 1.0009% |
| recurrent | 0.8588% | 1.0433% |

两种长度的所有检查步logits argmax一致。首次非逐位一致点均为第一次Decode的layer_idx=3（首个Full层）。Prefill路径未变。
代码层面的精度差异：dense会把QK结果与softmax概率存回BF16；Triton使用FP32 scores、在线softmax与PV累积后才存回BF16。这能解释差异的可能来源，但本次未逐项消融归因。后续GDN状态差异可以来自上游hidden变化，不能仅凭状态不相等判定slot串用。
真实自由生成计时中，四种配置、三个后端的8-token输出均一致；有限短样本不保证长序列或任务质量。

## 独立算子：分清提交间隙与设备执行

原始eager CUDA Event区间包含host供给不足时的GPU空隙，不能称为纯kernel时间。例如B1/L128：dense约130.0μs、Flash29.4μs、Triton16约34.1μs。

为隔离这项影响，另做测量专用CUDA Graph：捕获连续16次独立算子调用、预热后重放，5轮取中位数，再除16。下表是摊销GPU区间，仍包含节点/图执行开销，不是Nsight单kernel硬件计数；只用于算子实验，不代表引擎支持Graph。

| B | L | dense μs | Flash μs | Triton tile8 μs | tile16 μs | tile32 μs |
|---|---|---:|---:|---:|---:|---:|
| 1 | 128 | 22.00 | 10.11 | 27.90 | 16.95 | 13.50 |
| 4 | 128 | 32.77 | 10.24 | 27.78 | 16.83 | 13.38 |
| 1 | 1024 | 45.18 | 12.41 | 208.57 | 120.83 | 93.11 |
| 4 | 1024 | 139.39 | 19.01 | 208.51 | 120.96 | 93.25 |

长上下文下串行遍历历史的手写版明显落后Flash；tile增大在这些样本有效，但不能推出越大越快。实际接入保持tile16，未利用测量结果偷偷改默认配置。

## 真实引擎：整步TPOT（ms，中位数）

| B | prompt L | dense | Flash | Triton16 | Triton相对dense降幅 | 配对获胜次数 |
|---|---|---:|---:|---:|---:|---:|
| 1 | 128 | 33.757 | 32.594 | 33.042 | 2.12% | 6/6 |
| 4 | 128 | 130.286 | 126.541 | 131.064 | -0.60% | 3/6 |
| 1 | 256 | 33.987 | 32.407 | 33.020 | 2.84% | 6/6 |
| 4 | 256 | 130.701 | 125.793 | 127.694 | 2.30% | 5/6 |

Triton TPOT范围分别为32.941–35.815、127.317–150.113、32.957–33.114、126.526–141.616ms。有明显运行波动，不能把小中位数差异表述为普遍稳定提升。

| B | L | TTFT dense/Flash/Triton ms | 总输出吞吐 dense/Flash/Triton tok/s | 峰值allocated GiB（均相同） |
|---|---|---|---|---:|
| 1 | 128 | 126.51 / 126.04 / 125.97 | 22.06 / 22.58 / 22.38 | 8.139 |
| 4 | 128 | 502.12 / 500.25 / 543.24 | 22.65 / 22.90 / 21.90 | 8.142 |
| 1 | 256 | 136.06 / 135.85 / 136.51 | 21.31 / 22.07 / 21.77 | 8.167 |
| 4 | 256 | 537.63 / 537.80 / 538.56 | 22.03 / 22.56 / 22.15 | 8.172 |

Prefill并未改变，因此不能将TTFT差异归因于Decode kernel。整段生命周期峰值也可能由Prefill决定；没有测出峰值显存降低，不等于内部临时张量完全相同。

## 下一课建议

先学习并实现split-K的两个分片合并公式，再做长历史并行化；保留当前kernel作正确性参考。
原因是当前每个request/head只有一个program顺序遍历全部历史，B=1只有16个program，缺乏足够并行性。寄存器占用、实际带宽等仍需Nsight验证，不能仅凭耗时断言硬件瓶颈。
另一条独立方向是把Full层投影与attention移到请求循环外，才能让真实引擎利用批量grid；不和split-K同时改，便于归因。

## 使用与复现

```python
from nanovllm.qwen35 import Qwen35LLM
llm = Qwen35LLM(model_path, decode_backend='triton_paged')
```

在主目录运行（先重新核查空闲GPU）：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/ubuntu/enter/envs/nanovllm/bin/python -u week5/triton_paged_decode/integration_experiment.py

CUDA_VISIBLE_DEVICES=7 TRITON_MICRO_ONLY=1 TRITON_DEVICE_TIMING=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/enter/envs/nanovllm/bin/python -u week5/triton_paged_decode/integration_experiment.py
```

结果原始小日志保留在本目录 `integration_20261015.jsonl` 和 `device_timing_20261015.jsonl`。没有trace或权重副本；临时Triton JIT缓存退出自动清理。
