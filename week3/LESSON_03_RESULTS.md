# Lesson 3：真实混合缓存观察报告

日期：2026-09-28。范围：状态元数据、选定层的更新和长度重置观察；不是logits正确性或性能验收。

## 工程验证

10项新增CPU测试先失败、后通过；课程回归发现71项，69通过、2项真实tokenizer测试默认跳过。
CPU覆盖共享存储、带偏移view、实际dtype、空张量与未知状态区别、层数/长度错误和只读数值检查。
snapshot_cache只返回JSON元数据，不clone张量；实验代码另行clone少量状态用于数值对比。

## 真实运行

模型：Qwen3.5-4B，ModelScope revision ed182e32090db791077e12e0f58d22f3daafa173。
物理GPU5，UUID GPU-093b8773-f073-4c18-87c7-2afa616fd9c2，运行前289 MiB、利用率0%，非独占。
BF16，eager Full Attention，Linear Attention为环境原有torch回退；未改依赖。
每个请求生成4个token，即1次Prefill、3次Decode，均以length停止。

| 请求 | 输入token | 各次forward后的已处理长度 | 首次缓存MiB | 最后缓存MiB |
| --- | --- | --- | --- | --- |
| A（较长问题） | 35 | 35 → 36 → 37 → 38 | 26.59375 | 26.6875 |
| B（你好） | 13 | 13 → 14 → 15 → 16 | 25.90625 | 26.0 |

这里的缓存MiB是全部已观察状态张量的逻辑字节。本次与去重后的storage字节恰好相等，不代表两种口径永远相等。

### Full Attention

检查全部8层，K/V的历史维度均等于上表已处理长度。
每增加一个token，8层K/V合计增加32768字节（32 KiB）。
对原始第3层额外复制更新前K/V，逐次检查历史前缀，数值完全保留。
只对第3层做了历史数值比较，不能声称8层历史数值均已对拍。

### Linear Attention

全部24层：
- conv_states形状为(1,8192,4)。
- recurrent_states形状为(1,32,128,128)。
- 本次实际dtype均为BF16；合计逻辑状态字节25.5 MiB。

抽查原始第0层，A请求3次Decode的recurrent最大绝对变化：
2.634765625、2.125、1.583984375。
conv最大绝对变化：
30.78125、30.78125、31.71875。

这些数值说明选定样本发生更新，不是误差指标，也不是“正确性容差”。
其他23层只核对元数据，未逐元素比较更新。

## 把模型缓存、诊断副本和进程显存分开

- 实际加载模型参数字节：9078531072（包含未使用的视觉权重）。
- 加载阶段peak allocated：9180272128字节。
- A最后一步模型缓存：27983872字节；额外保留的诊断clone：1269760字节。
- B最后一步模型缓存：27262976字节；额外诊断clone：1179648字节。
- A带诊断的request peak allocated：9238007808字节。
- B带诊断的request peak allocated：9236123648字节。
- 两请求结束、诊断引用释放后，allocated均为9188792320，reserved均为9284091904字节。

allocator数字包含模型、工作空间和诊断开销，不能拿它们当模型缓存大小。
clone、float()差异计算也会改变峰值；此实验不作为性能/原始峰值显存基线。
请求结束reserved仍高不等于泄漏。整个进程退出后GPU5回到原有289 MiB占用。

## 生命周期边界

B请求首步位置为0，缓存长度13，没有延续A的38个位置。
这只是长度/入口状态观察，不足以证明完全没有数值污染。
“B单独运行 vs A之后B”、异常恢复的正式数值比较留到Lesson 4。

## 原始证据与复现

- [完整JSON](results/qwen35-cache-observation.json)
- [实验源码归档](../archive/development/lesson3-observation-experiment.py)
- [CPU单元测试](test_cache_inspection.py)
- [只读观察器](cache_inspection.py)

源码是本次一次性诊断脚本。复跑前检查GPU，显式设置CUDA_VISIBLE_DEVICES及PYTHONPATH=.，并将脚本destination改为新报告文件名；已有结果拒绝覆盖。不要直接照搬物理GPU5的历史状态。

## 审阅与未完成项

独立只读审阅未发现Critical/Important问题。
已知小限制：两个独立的零字节storage可能共用storage_index；字节总和仍为0，但不能用该索引证明空张量共享存储。
测试已检查值不变和JSON输出；张量身份/引用存活的专门回归尚未补充。
不要把本课结果写成数值对拍通过、内存泄漏已排除或原生引擎适配完成。
