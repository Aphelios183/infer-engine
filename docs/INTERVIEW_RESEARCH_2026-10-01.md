# AI Infra面经与岗位调研：2026-10-01

## 范围与可信度

面向推理框架/推理性能实习，不是Agent后端、训练算法或资深算子专家的全量要求。优先查近期公开面经，再用官方岗位描述校验方向。检索到相关9月面经，但不能保证覆盖全网最新帖子。

面经是匿名用户的自述或整理，不能独立确认面试真实发生、录用结果与普遍性。部分页面仅显示月日，不能据此断言面试年份；下表保留原页面日期口径。没有足够独立样本估计频率，不使用“某题必考”“覆盖率90%”等说法。校招、实习、社招不能混为同一门槛。

## 纳入的样本

| 编号 | 来源与日期口径 | 可支持的观察 | 限制 |
| --- | --- | --- | --- |
| S1 | [阿里云0914一面](https://www.nowcoder.com/feed/main/detail/40f786c658f44730876703db3b1504ac)，页面09-14，标题0914 | 推理性能方向涉及算子位置、硬件生态、FlashAttention、项目深挖、AI Coding与手写题 | 单篇自述；不是统一实习题库，年份未独立确认 |
| S2 | [阿里AI Infra二面原帖](https://www.nowcoder.com/feed/main/detail/2b15e6dfeeeb491f97fdeeaa2c12e438?sourceSSR=enterprise)，[平台列表可见正文](https://www.nowcoder.com/enterprise/134/interview)，09-18 | 提到TP/Graph故障定位、Decode同步、集合通信及PD分离 | 原帖正文抓取受限，使用同平台列表；同文重复条目去重，非两个独立样本 |
| S3 | [阿里AI Infra一面原帖](https://www.nowcoder.com/feed/main/detail/6568c432f3d84f3d89ee663a277de6dc?sourceSSR=enterprise)，[平台列表](https://www.nowcoder.com/enterprise/134/interview)，09-17 | 涉及rollout路由、并行方式、KV传输与推理框架 | 内容偏RL/训推交叉；与S2可能相关，不作为独立趋势统计 |
| J1 | [百度AI Infra实习生J104435](https://talent.baidu.com/jobs/detail/INTERN/6ba68d4d-889c-48a1-bed7-d4112c43014b)，官方2026-08-13 | 框架工程、编程算法、性能分析、CUDA及计算通信效率有明确要求 | 职责同时覆盖训练与推理；不要求两个月全部实现 |
| J2 | [百度推理基座校招J101235](https://talent.baidu.com/jobs/detail/GRADUATE/7753cdb4-70b0-462b-a634-ee53818c7c9a)，官方2026-07-21 | 推理引擎、缓存、调度、注意力与矩阵计算是相关能力方向 | 校招不是实习最低线；不照搬全部工具清单 |
| J3 | [百度视觉模型推理实习J98660](https://talent.baidu.com/jobs/detail/INTERN/88ef5f2d-93ca-445b-9b39-90b7e2f4880e)，官方2026-07-21 | Linux、编程和数据结构基础；另有在读身份及到岗要求 | CV/多模态岗位，不等于本项目直接匹配；资格条件应逐岗核查 |

## 排除与降权

- [快手2026校招整理帖](https://www.nowcoder.com/discuss/904326953174323200?sourceSSR=dynamic)把模型量化问题答成金融量化策略，存在明显内容污染；不将其中答案作为技术依据。
- [字节Agent Infra面经](https://www.nowcoder.com/discuss/930155293864914944?sourceSSR=post)自述2026-09-17，但偏Agent Runtime和后端；不因此给推理引擎路线增加整套MySQL/Agent框架课程。
- [字节实习题目汇编](https://caomaolufei.github.io/AIInfraGuide/interview/字节跳动-ai-infra-实习-一二三面/)标注2026-04-17，但属于二次整理；仅作为线索，不认作近期一手样本。
- 搜索出现的招聘聚合站、薪资文章、商业课程和他人简历，不作录用标准或项目收益证据。没有取得可信正文的公司不编造面试结论。

## 技术答案不从面经抄

- 框架：以服务器nano-vLLM与正式vLLM源码为准；实际路径见 [Week4对照卡](../week4/FRAMEWORK_COMPARISON.md)。正式vLLM本地版本尚未确认，不等同文档v0.18.2。
- Attention IO：[FlashAttention原论文](https://arxiv.org/abs/2205.14135)，用于理解分块与HBM访问，不要求当前独立重写完整kernel。
- 通信：[NCCL官方collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)，先分清AllReduce、AllGather、ReduceScatter语义；再学实现与性能，不能混淆API和互连硬件。

## 对当前项目的推断，而非企业官方结论

最合适主攻方向是推理框架/模型适配，性能分析作为第二支点。nano-vLLM–Qwen3.5提供真实混合模型的状态与执行问题；InferLab承担C++资源管理、调度和局部性实验。两者应形成一条证据链，而不是两个都只跑通README的项目。

目前不能仅凭课程问答判断达到实习录用水平。需要补齐独立源码修改、真实模型正确性、至少一次有证据的性能分析、C++与算法现场编程；计划见 [面试耦合计划](../INTERVIEW_PLAN.md)。
