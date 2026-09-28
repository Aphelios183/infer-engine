# Lesson 2：真实 Prefill + Decode 验证

日期：2026-09-28。范围：单请求两次 forward 冒烟验证，不是完整生成验收、数值对拍或性能测试。

## 运行条件

- 模型：/home/ubuntu/huggingface/Qwen3.5-4B。
- 来源：ModelScope Qwen/Qwen3.5-4B；revision ed182e32090db791077e12e0f58d22f3daafa173。
- Python：/home/ubuntu/enter/envs/nanovllm/bin/python。
- Transformers 5.8.0，torch 2.5.1+cu124。
- GPU物理编号5，CUDA_VISIBLE_DEVICES=5 后进程内 cuda:0；运行前利用率0%，约289 MiB占用，非独占卡。
- Qwen3_5ForConditionalGeneration，BF16，attn_implementation=eager，eval + inference_mode。
- 本地离线加载，未使用 generate()，未改依赖。
- 缺少线性注意力加速依赖，Transformers 明确回退 torch 实现；本次不报告性能。

## 实测结果

提示词：用一句话解释 KV Cache。关闭thinking，模板输入17个token。

| 阶段 | 输入形状 | logits形状 | 贪心选出 | Full Attention KV长度 |
| --- | --- | --- | --- | --- |
| Prefill | (1,17) | (1,17,248320) | 79852，KV | 17 |
| Decode 1 | (1,1) | (1,1,248320) | 18887，前导空格加Cache | 18 |

通过断言：logits形状符合配置、两次输出均为有限值、缓存长度17→18；第一次选出的token非EOS，才执行Decode。
Prefill position_ids为0..16，mask长度17；Decode position_ids为17，mask长度18。
选取使用 logits[:, -1, :].argmax(dim=-1, keepdim=True)。

## 实际混合状态

缓存对象为 DynamicCache，共32层。

- Full Attention位于原始索引3、7、11、15、19、23、27、31，每层K和V均为BF16：
  - Prefill：(1,4,17,256)
  - Decode：(1,4,18,256)
- 其余24层的conv_states：(1,8192,4)，recurrent_states：(1,32,128,128)。
- 本次实际conv/recurrent状态均观察为BF16；不能把配置中声明的float32当成实测dtype。
- Linear状态形状两次相同；本次只记录形状/dtype，未验证其数值更新、请求隔离或完整内存占用。

## 结论与限制

只输入刚选出的一个token并携带历史状态，真实调用成功；第二个新token尚未写入缓存。
“KV Cache”只是两次选择得到的开头，不是一份完整回答。

未完成：全重算/缓存logits对拍、EOS与长度边界的真实循环测试、多提示词验收、独立reference_runtime模块和原生nano-vLLM适配。
因此原计划Tasks 3–6不能因本次诊断运行被整体标记完成。

原始输出：[forward_smoke_20260928.log](results/forward_smoke_20260928.log)。
下一步：自己补齐控制循环的停止条件与每步输入，而不是直接调用generate()。
