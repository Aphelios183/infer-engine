# 小课5：不混批的Chunked Prefill首版

2026-10-11。代码已接入并在GPU6 L40验证。默认仍是完整Prefill；分块通过参数显式开启。**同分块参考对齐通过，但整段与分块的数值等价尚未建立，不能宣称质量无损。**

## 1. 开启方式

在 `/home/ubuntu/infer-engine/nanovllm_qwen35` 下：

```python
from nanovllm.qwen35 import Qwen35LLM
from nanovllm import SamplingParams

engine = Qwen35LLM(
    '/home/ubuntu/huggingface/Qwen3.5-4B',
    max_model_len=512,
    max_num_seqs=2,
    prefill_chunk_size=64,       # 每请求每个Prefill步骤的上限
    max_num_batched_tokens=128,  # 整批输入token预算，不是每请求预算
)
try:
    prompt = engine.tokenizer.apply_chat_template(
        [{'role': 'user', 'content': '请解释KV缓存和分页管理的关系。'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    print(engine.generate([prompt], SamplingParams(max_tokens=32)))
finally:
    engine.close()
```

先检查空闲卡再设置CUDA_VISIBLE_DEVICES，不要固定认为GPU6一直空闲。`prefill_chunk_size=None`保持旧的整段路线。输入加生成预算仍不得超过max_model_len。

## 2. 调度与资源策略

- 一批要么Prefill，要么Decode，不混合两阶段。这里只与nano的“不混批”思路一致，不声称调度优先级或实现细节完全相同。
- 保留现有Decode优先；若没有选出Decode，则先继续已准入的部分Prefill，再准入新请求。长生成任务可能延迟其他Prefill，没有新增公平性保证或抢占。
- 每个chunk长度取“剩余prompt、单请求chunk上限、本批剩余预算”的最小值。
- 首次准入预留整个prompt的KV块，分配并初始化state_slot；后续chunk既不重新分配状态槽，也不清零conv/recurrent。
- 预留容量不等于有效历史。即使block_table已覆盖整个prompt，context_lens仍只到本轮实际处理后的end。
- 每轮只允许一个未结算batch；失败后不能重跑已更新的GDN状态，仍需确认GPU停止访问后整体回收失败批次。

## 3. 必须掌握的三个索引与计数

`start = num_cached_tokens`，`end = start + input_count`；本轮输入为token_ids[start:end]，位置为range(start,end)。

`last_indices`索引packed hidden的token行；`sample_rows`索引原批次的请求行。只有处理完prompt的请求或Decode请求才出现在sample_rows中。

例：批次[B,A]，B处理3个但还没完成，A处理3个并完成：

```python
cu_seqlens_q = [0, 3, 6]
last_indices = [5]
sample_rows = (1,)
```

结算时两者都增加3个已处理token，只有A追加采样token。整批sample_rows为空时，forward照常更新状态，跳过LM Head和sampler，仍执行GPU同步、postprocess计数提交。

`StepResult.request_ids`包含所有执行请求，`token_ids`仅对应sample_rows，外部消费者必须这样关联：

```python
for row, token_id in zip(result.sample_rows, result.token_ids):
    request_id = result.request_ids[row]
```

不能直接zip所有request_ids与token_ids。基线脚本已同步更新此计数方式。

## 4. 源码阅读顺序

代码根目录 `/home/ubuntu/infer-engine/nanovllm_qwen35`：

1. `qwen35_adapter/scheduler.py`：ScheduledBatch.sample_rows，schedule的Decode/Prefill分支，postprocess的计数与采样分离。
2. `qwen35_adapter/admission.py`：scheduled_tokens限制本轮预算，BlockManager仍预留完整prompt。
3. `qwen35_adapter/runner_inputs.py`：输入切片、部分context_lens、完整容量块表，以及last_indices/sample_rows校验。
4. `qwen35_adapter/gpu_engine.py`：新增参数；无采样时跳过LM Head，GPU同步不跳过。CPU教学runner同步实现相同协议。
5. `qwen35_adapter/pool_backend.py`和`layer_compute.py`：已有past-aware KV/GDN路径，本次无需重写；前次positions/slots复用仍保留，历史addresses跨层复用仍未实现。

## 5. 实测证据

`verify_chunked_prefill.py`是单个可执行验证入口，不恢复先前删除的tests目录。

- 控制流程10类检查通过：部分采样映射与温度对应、不混批、整批无采样、完整prompt预留、状态保留、pending取消保护、部分取消后复用、错误采样回收、零预算和全局预算截断。
- GPU扩展组38步：chunk上限1/3/4/16/64，长度3/5/6/7/9/17/33/65/128，包含异长双请求、请求重排、跨块、尾部单token、全部/部分/无采样。与**安装环境HF按相同输入分块执行**相比，采样logits及各步有效KV、conv/recurrent精确一致；生成token一致。
- 无采样chunk的forward后失败、部分Prefill取消、槽位复用、最终chunk EOS均通过。
- 原默认完整Prefill的27步真实4B验证仍精确通过，保留原生成结果；没有自动开启分块。

### 整段与分块数值：明确尚未通过的部分

同分块精确对齐不能推出与整段Prefill相等。本次还比较了最终prompt位置的logits和全层缓存：

| 长度 / chunk上限 | logits最大绝对误差 | 相对L2 | 首token argmax |
| --- | --- | --- | --- |
| 7 / 3 | 0.171875 | 1.31% | 相同 |
| 33 / 16 | 0.453125 | 4.53% | 相同 |
| 65 / 64 | 2.111328 | 17.43% | 相同 |
| 128 / 64 | 0.468750 | 2.45% | 相同 |

缓存也不完全相等，例如128/64的跨层KV最大绝对差约12.125、conv约14、recurrent约0.22168。未通过放宽阈值将其标为等价。分段形状、chunk/recurrent计算路径和持久状态精度均可能影响结果，但具体误差来源仍需进一步定位，不能仅以“BF16舍入”解释全部差异。

本次检查的prompt末端argmax相同，不代表完整生成、概率分布或任务质量相同。因此该功能保持显式选择的实验性能力，不用于宣称无损加速。没有做本轮性能benchmark，也不承诺吞吐提升。

## 6. 复现与产物

适配根目录执行：

```bash
/home/ubuntu/enter/envs/nanovllm/bin/python verify_chunked_prefill.py --cpu-only --output chunked_control_new.json
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/ubuntu/enter/envs/nanovllm/bin/python verify_chunked_prefill.py --output chunked_gpu_new.json
```

输出文件必须未存在。最终证据为 `chunked_control_extended_20261011.json`、`chunked_gpu_extended_20261011.json` 和 `chunked_default_regression_20261011.json`；它们是小型验证记录，无大trace。早期控制/34步运行结果也保留以区分实验批次。

## 7. 学生下一步

概念问答已通过，代码由助手实现，不自动记为学生独立实现。先阅读schedule与postprocess：当总预算5、chunk上限4、两个prompt长度9和5时，三轮Prefill输入数为(4,1)、(4,1)、(1,3)。请逐轮写出sample_rows、两个请求的num_cached_tokens，以及最后一轮生成token与原批次行号的对应关系。
