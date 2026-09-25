# Week 3 第一课：模型接收的不是文字，而是 token IDs

今天只做 CPU 实验：读取本地 Qwen3-0.6B 配置和 tokenizer。先不用 GPU、不加载权重，也不修改 nano-vLLM 源码或 Python 环境。

## 1. 我们到底要自己实现什么？

交付物仍是自己的 infer-engine。第一阶段可以复用 Transformers 的模型 forward、tokenizer 和已有算子，但必须自己写 Prefill/Decode 的控制流程、下一 token 选择、EOS/长度停止、缓存对象传递和请求状态。

仅调用现成 generate() 不算完成这项交付。第三周后续会让你在 minimal_generate.py 中实现自己的单请求循环；它是引擎的第一块，不宣称已经自研模型算子、分页缓存或完整服务。nano-vLLM 是参照与学习对象。

## 2. 今天的输入链路

```text
用户文本 → chat template → 格式化文本 → tokenizer → token IDs
```

- chat template：标明谁在说话、消息在哪里结束、模型应该从哪里开始回答。
- tokenizer：按已有词表和编码规则，把文本转换成整数 ID 序列。
- token ID：词表索引，不是向量、概率或已经计算好的 K/V。
- embedding：模型稍后把 ID 查表变成向量；今天还没有走到这一步。

“一个字等于一个 token”不成立。分词可能把片段合并，也可能把某个字符拆成多个 token；标点、空格、角色标记同样会影响输入长度。

## 3. 运行观察脚本

```bash
cd /home/ubuntu/infer-engine
conda activate nanovllm
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python week3/inspect_model.py
```

脚本只读取本地文件，local_files_only=True，不下载模型，不加载 model.safetensors。输出中的 KV 大小是理论计算值，不是测到的 GPU 分配量。

再换一句话观察长度，不要只背默认答案：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python week3/inspect_model.py --prompt 'Hello, KV Cache!'
```

脚本会比较两种路线：模板直接 tokenize，与先得到模板文本再 encode(add_special_tokens=False)。输入 IDs 必须相同。这是为了防止在预处理阶段重复添加标记。

## 4. 真实模型配置与教学模型的区别

本地配置：28 层、hidden_size=1024、16 个 Q 头、8 个 KV 头、head_dim=128，模型配置 vocab_size=151936。

GQA 每 2 个 Q 头共享一组 K/V。注意 1024/16=64，但配置明确 head_dim=128：不能照搬第一周的 C/H 假设。Q 投影总宽度为 16×128=2048，K/V 各为 8×128=1024，Attention 后经输出投影回到 hidden_size=1024。

BF16 或 FP16 KV 每元素 2 字节。单请求 4096 个位置的全层 KV 理论占用为：

```text
2 × 28 × 1 × 4096 × 8 × 128 × 2
= 469762048 字节 = 448 MiB
```

这不是模型权重大小，也不是启动引擎后的总显存。正式生成时还要确认实际 KV dtype 和布局。

## 5. 模板中的特殊 token

本地 tokenizer 的 EOS 是 <|im_end|>，ID 为 151645。默认短提示词格式化后，既有用户消息结束标记，也有 assistant 开始标记；关闭 thinking 时，当前模板还会在 assistant 前缀里放置空的 <think>...</think> 标记。

这些字符只是实际模板格式，不说明本次已经生成了思考过程。关闭 thinking 的实际行为要以本地模板输出为准。

一个易错点：输入里已经有用户消息的 EOS/结束标记，并不意味着生成循环应该立即退出。后面实现停止条件时，判断的是新生成的 token 与生成上限，而不是“整段输入曾经出现过 EOS”。

另一个区别：本地 tokenizer.vocab_size=151643，len(tokenizer)=151669，而模型 config.vocab_size=151936。它们分别报告基础词表、含新增 token 的 tokenizer 大小和模型配置的词表输出宽度；不要强行当成同一个数。为什么模型额外保留这些位置，需要进一步核查具体模型设计，今天不凭差值下结论。

## 6. 连接下一课：从 ID 到下一个 ID

以 batch=1、输入长度 L 为例，参考完整模型通常返回 [1,L,V] 的 logits，V 使用模型输出词表维度。这里的 logits 是候选 token 的未归一化得分，不是概率。

取最后有效输入位置的 logits，选择下一个 token。对于这个模型，V 预期为配置中的 151936，实际 forward 时还要验证。把新选出的 token 送入下一次 forward，才会计算并写入它的 K/V。

今天还没有加载模型，这些是下一课要检查的预期，不是今天已测量的 forward 结果。

## 7. 你的第一组问题

先运行默认脚本，再回答：

1. 原始用户文本有几个 token？应用模板后有几个？为什么模型真正输入的长度变了？
2. 输出里看到 ID=151645，它是用户消息中的标记，为什么不能据此直接停止生成？
3. 今天得到的 input IDs 是向量吗？哪个步骤会把它们变成向量？

先回答这三题，不急着一次学习全部模型层。接下来我们才加载权重，观察 logits，并把生成循环写进自己的仓库。
