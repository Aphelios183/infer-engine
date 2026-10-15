# Lesson5：手写一个只读分页 Decode kernel

2026-10-15。此文件讲解独立教学kernel，不宣称比FlashAttention快。后续新增可选引擎接入与性能实验，见 [接入实验报告](INTEGRATION_RESULTS.md)；默认后端及row/batch开关不变。

## 先看文件

- paged_decode.py：Python包装器 + 分6步注释的Triton kernel。
- verify.py：小型练习对拍，FP64 CPU稠密参考；不保存trace/权重，临时JIT缓存退出自动清理。

## 契约

Q为[B,Hq,D]，每请求只有一个新token；K/V为当前层连续池[Npages,page_size,Hkv,D]，块表[B,M]、有效长度[B]为int32 CUDA张量。当前token KV已在池中，Q/K已完成RoPE；返回[B,Hq,D]。不包括KV写入、GDN、gate/o_proj、Prefill、backward。

有效上下文必须非空，覆盖页必须合法，有效Q/K/V应为有限值。包装器检查形状、dtype、设备、布局与元数据，CPU读元数据会同步；不是无同步生产路径，不支持直接Graph捕获。FP16可调用但此次先验证FP32/BF16。不支持任意stride。自己编写的kernel允许toy page_size=4，不能把这一点套用到FlashAttention接口。

## 逐段读代码：先预测，再读实现

1. `@triton.jit`：声明将编译为GPU执行的函数。Python负责验证/分配/启动，kernel内用tl操作，不调用torch。`tl.constexpr`参数在编译时确定，改变值可能产生新编译版本。
2. `_paged_decode[(B,Hq)](...)`是启动网格；`tl.program_id(0/1)`得到请求号和Q头号。一个program不是一个线程；`num_warps=4`表示线程协作规模，不是4个program。
3. `tl.arange(0,BLOCK_D)`创建索引向量。D=80时BLOCK_D=128，加载/写出mask保护剩余列；constexpr让编译器知道这些块张量的形状。
4. tokens[BLOCK_T]通过块表变为slots。计算tile与存储页独立，tile可能跨页；块表加载本身也必须mask，否则还没加载KV就可能越界。
5. 广播：slots[:,None]与dims[None,:]形成[BLOCK_T,BLOCK_D]地址。pointer+offset按元素寻址，不能再乘dtype字节数。masked tl.load的other=0只是安全加载，scores仍要mask为-inf。
6. 两次归约：QK用axis=1留下token维；PV用axis=0留下特征维。m/l标量，acc向量；新最大值出现时分子分母一起缩放，最终acc/l。
7. 只遍历start<length，避免第一块全无效造成-inf-(-inf)=NaN；长度0由包装器拒绝，不默默除零。
8. `tl.store`写独占[request,q_head,:]，不需要原子操作。多个Q头可只读共享KV头，但不能共享输出区域。

本例使用FP32乘法/归约/在线状态，最后按输出dtype存储，未使用tl.dot或Tensor Core优化。顺序遍历长历史、不同Q头重复读KV、有限program数和寄存器压力都可能限制性能；tl张量也不保证永远只驻留寄存器。先保证正确，不自动调参或宣称高性能。

## 四个练习（一次做一题）

1. B=2,Hq=16：grid多少个program？program_id=(1,4)读哪个请求、哪个KV头（Hkv=4）？
2. BLOCK_T=16,D=80：K块的编译形状是什么？哪些列必须mask？scores与acc分别是什么shape？
3. 删除score mask但保留load mask，会影响分子还是分母？用尾块反例解释。
4. 请求换序时，为什么只换Q会失败？自己指出tables/lengths应如何换序，池是否也必须搬移？

不要同时更改索引、tile和精度。每次只做一个练习改动，再运行对拍。助手实现不算学生独立手写。

## 验证与运行

先检查空闲卡；从本目录执行：

```bash
CUDA_VISIBLE_DEVICES=<空闲编号> PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/enter/envs/nanovllm/bin/python verify.py
```

固定seed42；页4/256，D64/80/256，GQA比例不同，长度1/6/9/17/255/256/257/300，tile8/16/32。无效KV填NaN，帮助发现误读；检查换序输出、KV逐字节不变、零/越界长度与非法页拒绝。CPU FP64参考读取相同已量化输入；比较时舍入回输出dtype。阈值FP32 atol/rtol=2e-5，BF16 atol=.008/rtol=.01，属教学用例而非模型质量门槛。

官方入门：[向量加法、program与启动网格](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)，[masked load语义](https://triton-lang.org/main/python-api/generated/triton.language.load.html)。在线文档可能新于服务器版本；以本次安装环境编译结果为准。

## 本次结果

2026-10-15在GPU 7（NVIDIA L40，运行前292 MiB/0%利用率）验证，Torch 2.5.1+cu124、Triton 3.1.0。

- 24组配置全部通过：2种dtype × 4组形状/长度 × 3种tile。
- 相对FP64参考舍入至输出dtype后的最大绝对误差：FP32为1.1920929e-7，BF16为3.0517578e-5。
- 请求换序输出逐元素一致；K/V池逐字节不变；NaN填充未污染输出；非法长度/页被拒绝。
- 临时JIT缓存退出自动清理，不保留trace或权重副本。

这是固定seed的合成输入对拍，不是任意输入精度保证。没有接入真实模型，也没有性能结论。
