# Lesson 02：从 logits 到自己控制的生成循环

当前进入 Lesson 2A：先学会取 logits、选 token 和维护循环状态。Lesson 2B 才接真实模型 forward。
工作目录：`/home/ubuntu/infer-engine`；继续使用 nanovllm 环境。此文的教学张量实验只用 CPU，不下载文件、不加载权重。

## 1. 接住上一课：第17个位置怎么写

假设 batch=1、输入17个 token，模型返回所有输入位置的分数：

```python
# logits: [1, 17, V]
last_logits = logits[:, -1, :]            # [1, V]
next_ids = last_logits.argmax(dim=-1, keepdim=True)  # [1, 1]
```

索引从0开始，第17个位置是索引16；`-1` 表示最后一个位置。
三个维度分别是 batch、输入位置、词表候选。不能沿序列维取 argmax。

logits 是候选 token 的未归一化分数，不是 token ID，也不是 embedding。
贪心选择只需 argmax，不必先 softmax；softmax 不改变同一组有限分数的大小顺序。
这里的“最后位置”成立于当前 batch=1、无 padding 的范围；有 padding 后要取最后有效位置。
部分真实模型接口只返回最后位置 logits，后续需检查实际输出，不能假定始终返回所有位置。

## 2. 先运行一个不需要模型的实验

在主目录终端执行：

```bash
CUDA_VISIBLE_DEVICES='' /home/ubuntu/enter/envs/nanovllm/bin/python -B - <<'PY'
import torch

# 教学词表只有5个候选；它们不是 Qwen 的真实 token 编号。
logits = torch.zeros(1, 17, 5)
logits[0, -1] = torch.tensor([0.2, -0.5, 3.0, 1.0, 2.0])

last_logits = logits[:, -1, :]
next_ids = last_logits.argmax(dim=-1, keepdim=True)

print("完整 logits:", tuple(logits.shape))
print("最后位置:", tuple(last_logits.shape))
print("下一个输入:", tuple(next_ids.shape))
print("选中的教学 token ID:", next_ids.item())

assert tuple(last_logits.shape) == (1, 5)
assert tuple(next_ids.shape) == (1, 1)
assert next_ids.item() == 2
PY
```

先预测再运行：分数最大是3.0，但选出的 token ID 是2，不是3。
`keepdim=True` 保留长度为1的维度，方便下一步作为 `[B,1]` 输入。
实验只证明形状与选择逻辑；不证明真实模型生成、缓存或数值对拍通过。

## 3. Prefill 与 Decode 的时序

| 动作 | 本次输入 | 模型已处理的 token 数 | 刚选出的新 token |
| --- | --- | --- | --- |
| Prefill | 原始17个 token | 17 | A |
| 第一次 Decode | 仅A，并传历史状态 | 18 | B |
| 第二次 Decode | 仅B，并传历史状态 | 19 | C |

每一行的最后一列都还没被下一次 forward 处理。
对于 Full Attention，“已处理数量”对应当前历史 K/V 长度；Linear Attention 更新 recurrent/conv state，不能拿同一套普通 KV 张量布局解释它。

你已经回答正确：选出A不代表A写入缓存；输入A后才写入，并预测B。

## 4. 自己的生成循环要负责什么

本课复用模型的 forward，但不调用现成 generate() 代替主循环。
将来按下面的职责分工实现：

1. 格式化消息并得到 input_ids；模板只应用一次。
2. Prefill，取得最后位置 logits 和更新后的 cache。
3. 从 logits 选一个新 token，记录到输出。
4. 先检查新 token 是否是有效 EOS；再检查新输出数量是否达到上限。
5. 若停止，立即返回，不为最后选出的 token 多跑一次 forward。
6. 否则仅输入刚选出的 token 和 cache，运行 Decode，并更新 cache。
7. 重复第3步。

EOS ID 应从实际模型/生成配置读取并统一成集合，不把旧课常量硬编码进循环。
若同一步既命中 EOS 又到长度上限，本课约定停止原因优先记 eos。
`max_new_tokens=0` 应直接返回空输出，不进行 Prefill。

## 5. 三个变量，不要混成一个 length

- `prompt_length`：原始输入 token 数，固定不变。
- `generated_ids`：新选出的 token 列表；本课包含终止 EOS，可见文本解码时可隐藏特殊标记。
- `processed_tokens`：已经经过模型 forward 的 token 数。

当前单请求循环：若至少产生一个新 token，且选中最后一个后立即停止：

```text
processed_tokens = prompt_length + len(generated_ids) - 1
```

这是当前循环约定，不是所有推理框架必须遵守的通用存储规则。不要为凑等式去做无用 forward。

## 6. 将来接真实 Qwen3.5 时的边界

以下工作仍待实现/验证，本课不宣称已有可运行生成器：

- 读取锁定来源与版本的模型资产，验证权重与配置。
- 检查当前 Transformers 的真实 forward/cache/position/mask 接口。
- 用 eval + inference_mode，先单请求、无 padding、短输出。
- 对 Decode 传入正确的历史状态与位置；不能只写 input_ids 就假设位置永远正确。
- 每个新请求创建独立状态，不复用上一请求的 cache。
- 先验证一次 Prefill 和一次 Decode，再连接循环；禁止通过升级依赖掩盖未定位的错误。
- 运行前重新检查空闲卡；本课CPU实验不需要选择 GPU。

下载成功不等于模型可运行；参考模型生成不等于 nano-vLLM 原生适配完成。

## 7. 你的练习

先完成第1–2节的CPU实验，再回答：

1. `argmax(dim=-1)` 选的是位置编号还是词表 token ID？
2. 去掉 keepdim=True 后，next_ids 的形状是什么？
3. 输入17个 token，预算3个新 token，依次选出 A、B、C（均非EOS）：需要几次 Prefill、几次 Decode？停止时 processed_tokens 是多少？
4. 如果第一次 Decode 后选出 EOS，generated_ids 有几个 token，processed_tokens 是多少？

不需要一次学完。现在先回答第3题，再进入真实 forward 的准备。

## 8. 本课验收

- [ ] 独立写出最后位置 logits 的切片与贪心选 token。
- [ ] 能推演 EOS 和长度上限，理解最后选出的 token 可能尚未写入缓存。
- [ ] 实际实现并验证参考 forward（未完成）。
- [ ] 手写循环、记录停止原因，完成至少三条短提示词（未完成）。
- [ ] 后续逐步 logits 对拍通过前，不声称缓存数值正确性已验收。

## 9. 完整书写：现在阅读这两个文件

本节新增完整可运行的单请求控制代码；原第6、8节的“未完成”描述属于编写本课初稿时的状态，最终实测进度以本节及运行报告为准。

先读 [minimal_generate.py](minimal_generate.py) 中的 run_greedy，再读 [reference_runtime.py](reference_runtime.py) 中的 forward_last。

你的伪代码与真实实现对应关系：

| 你的写法 | 本课实际实现 |
| --- | --- |
| model.prefill(prompt_ids) | 第一次 step，cache=None、processed_tokens=0 |
| model.decode(next_ids, cache) | 后续 step，输入[1,1]，传入历史cache和已处理位置 |
| logits[:, -1, :] | forward_last内部完成，返回[1,V] |
| select_token(...) | argmax(dim=-1, keepdim=True)，返回[1,1] |
| next_token_id | next_ids.item()，整数用于输出和EOS判断 |
| cache.length | 混合cache的get_seq_length与processed_tokens；不要假定每层都有普通KV |

循环用 for range(max_new_tokens) 表达最多选出多少个新token。
第一次迭代包含Prefill，所以生成N个token只需最多N-1次Decode。
预算为0时run_greedy不调用forward；CLI仍会准备/加载模型，因此无GPU的零预算学习应运行CPU单元测试。

### CPU测试（无需模型、无需GPU）

```bash
cd /home/ubuntu/infer-engine
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m unittest week3.test_reference_runtime week3.test_minimal_generate -v
```

替身测试控制EOS出现时机，是为验证循环，不冒充真实模型结果。
覆盖：0/1/2预算、首token EOS、输入含EOS、异常后新请求、非法输入和非有限分数。

### 真实生成（先检查当时的GPU，不复用历史空闲结论）

```bash
cd /home/ubuntu/infer-engine
nvidia-smi
read -r -p '本次可用的物理 GPU 编号: ' LESSON_GPU
CUDA_VISIBLE_DEVICES="$LESSON_GPU" HF_HUB_OFFLINE=1 /home/ubuntu/enter/envs/nanovllm/bin/python -B -m week3.minimal_generate --model /home/ubuntu/huggingface/Qwen3.5-4B --device cuda:0 --prompt '用一句话解释 KV Cache。' --max-new-tokens 16 --output week3/results/my-first-generation.json
```

输出文件已存在会拒绝覆盖；换一个新文件名即可。
本次加载完整ConditionalGeneration checkpoint（含未使用的视觉权重），只接受纯文本。
保持现有环境，Full Attention使用eager，线性注意力保留环境现有回退路径；不提供性能优化声明。
加载器仅接受本课固定的ModelScope Git/LFS版本并离线校验，不是通用模型下载器。

### 你要看懂的trace

每步记录phase、input_ids、position_start、position_end_exclusive、processed_tokens、token_id、generated_tokens、stop_reason。
最后一步停止时，最后选出的token没有再经过forward。
结果只包含Python/JSON值，不把cache或GPU张量放进报告。

读完后，自己把预算改成1和2：先预测trace行数，再运行新文件名的实验。
注意：当前只完成生成控制链；下一课仍需验证混合状态与全重算/缓存logits对拍。
