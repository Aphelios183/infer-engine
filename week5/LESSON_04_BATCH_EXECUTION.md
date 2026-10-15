# Week5 Lesson4：从请求循环到批量执行

2026-10-14。入口：[Week5计划](PLAN.md)。主线始终为Qwen3.5-4B适配，不另起玩具推理引擎。本次仅建立课程，不实施批量化、不运行GPU。

## 1. 为什么现在学这一课

分页Decode减少了历史gather，但四组端到端计时有升有降。学生提出“大batch可能更快，显存节省后可容纳更多请求”的假设；保留为待验证问题，而不是结论。

必须分清：

- 调度batch：本轮选择了多少个请求。
- packed输入：把本轮token放进一张张量，边界由元数据区分。
- 计算batch：一次算子调用处理了多少行/请求。
- 可驻留容量：资源池能容纳多少请求，不等于吞吐。

当前有前两者，但很多计算仍逐请求调用；增大max_num_seqs可能主要增加循环次数。分页改动未缩小持久KV/state池；整轮峰值实测也未下降，不能写成“显存少很多”。逐请求临时张量一般不能按batch简单乘算。

## 2. 第一处源码：逐请求循环在哪里

根目录：`/home/ubuntu/infer-engine/nanovllm_qwen35`。

按顺序读：

1. `qwen35_adapter/runner_inputs.py::RunnerInputBuilder.prepare`：packed输入、last_indices和sample_rows如何关联。
2. `qwen35_adapter/pool_backend.py::PooledCPUBackend.forward`：外层遍历模型层，内层遍历request_ids，逐请求调用decoder_layer，再cat输出。
3. `qwen35_adapter/gpu_engine.py::GPUExecutionRunner.run`：只选择需要采样的hidden行，送入compute_logits。
4. `qwen35_adapter/gpu_engine.py::PooledGPUBackend.compute_logits`：列表推导逐行调用父类投影，最后torch.cat。
5. `qwen35_adapter/pool_backend.py::compute_logits`：F.linear执行hidden到词表的映射。

教学任务：指出“张量有B行”与“一次GEMM算B行”之间差了哪一段循环。不要因类名PooledCPUBackend就误认为GPU路径在CPU算。

## 3. 第一候选：LM Head批量化

当前概念代码：

```python
logits = torch.cat([linear(row[None], weight) for row in last_hidden], dim=0)
```

候选形式：

```python
logits = linear(last_hidden, weight)
```

这里仅展示思路，不是已批准直接替换的补丁。

Qwen3.5-4B hidden_size=2560，vocab_size=248320。令S为本轮实际需要采样的请求数：

- last_hidden为[S,2560]。
- 共享输出权重为[248320,2560]。
- logits为[S,248320]。
- S不一定等于调度请求数B。部分Prefill请求没有完成prompt时不采样；S=0时跳过LM Head和sampler，但forward及计数结算仍执行。

选择LM Head作为首个候选，是因为不直接修改KV/GDN状态读写，容易隔离；不预设它是最大热点。批量化仍须保持输出行与sample_rows的映射，不能让A的token追加给B。

## 4. 为什么不能直接认为数值完全一样

旧代码刻意逐行投影，是为了保持已有BF16参考计算形状。把多个小GEMM变成一个较大GEMM，数学表达式不变，但后端可能更换计算/归约路径；浮点输出不保证逐位相等。

验证顺序：

1. 使用完全相同的last_hidden和权重，仅比较两种LM Head。
2. 检查形状、dtype、有限值，记录最大绝对误差、相对L2、top1是否相同及top1/top2间隔。
3. 如有差异，对有限样例引入关闭TF32的FP32投影参考，区分旧新两条路径的偏差；不把旧实现视为绝对真值。
4. 再接入短端到端生成，包含部分采样、零采样、请求换序和结束回收。候选尚未验证时不删除逐行参考开关。
5. 用统一参考输入做单点对照，再看候选自身生成轨迹；不能在已更新GDN状态上重复forward做A/B。

不得因token相同就称无损，也不得直接沿用分页实验2%的探索门槛作为本课质量标准。

## 5. 后续Full Attention批量化的接口预习

同一层、B个单token Decode请求可准备Q[B,1,16,256]、block_tables[B,M]、context_lens[B]；当前层K/V池仍共享同一存储，实际访问由每行块表隔离。新KV分别按每个请求slot写入。

但现有decoder_layer按请求封装全部计算，不能只把kernel的batch参数改成B；还要组织投影、RoPE、gate和输出行。GDN则需state_slots与边界元数据，不能把不同请求的token串成一条递推序列。

本课先理解和评估LM Head，再决定是否进入Full层批量化，不同时改GDN、Graph或调度。

## 6. 用实验验证大batch与显存假设

正确性通过后，固定上下文和输出长度，对B=1/2/4/8比较逐行与批量候选。事先估算并确认空闲卡；实际可用容量不足则缩小矩阵，不测试到影响他人任务。

三类测量分开：

- 算子微基准：相同hidden/权重，预热后GPU计时，比较LM Head自身；不得冒充引擎吞吐。
- 无profiler引擎交错A/B：比较TPOT、输出token/s和波动，保持其他开关不变。
- 独立profile：核实调用次数及CPU提交/kernel变化。

显存：Prefill完成后同步、重置峰值，再观察Decode；记录基准allocated、峰值allocated和reserved，另保留全轮峰值。不能把reserved差异直接当有效KV容量。扩大max_num_seqs也会增加预分配池，必须注明配置变化。

## 7. vLLM与InferLab对照任务

核实本地vLLM仓库根和commit后，从GPU ModelRunner追踪hidden行选择、compute_logits和采样映射；比较其批次组织与我们的逐请求路径。未核查前不引用固定行号或声称最新文档就是本地行为。

InferLab对照线性层/执行接口如何表达矩阵的行数、stride及缓冲所有权；只读，不同步开展完整C++实现。

## 8. 验收与第一题

验收：能定位两层循环，手推S与B，解释数学等价/浮点差异；设计公平的算子与端到端测试；能指出状态资源和临时内存的区别。助手实现完成不等于学生独立完成。

第一题：本轮3个请求都在Decode，每个请求选出一个hidden，hidden_size=2560、vocab_size=248320。

1. last_hidden的shape是多少？
2. 当前逐行LM Head调用几次F.linear，每次输入shape是多少？
3. 改为一次批量F.linear后，输出shape是多少？

回答后再追问部分Prefill采样行；此刻不要求写完整优化代码。
