# Qwen3.5 × nano-vLLM：课程即适配主线

日期：2026-09-30。用户要求：后续课程必须围绕nano-vLLM的Qwen3.5改进开展，而不是学完框架后另起模型项目。

## 实现边界

以下是待执行路线，不是已完成能力。参考nano-vLLM只读；首次实现时在 `/home/ubuntu/infer-engine` 内固定适配副本、来源指纹与导入路径，保留用户注释，明确上游已有代码和自己的改动。具体目录在首次实作时确认并记录。

主模型Qwen3.5-4B；起点为单卡、TP=1、eager、纯文本、单请求。先把错误功能显式拒绝，再扩展。原始实现已有普通KV分页、前缀复用和有限chunked prefill，这些不算我们新增能力。

## 逐课增量与测试

| 课程 | nano-vLLM改进落点 | 必须留下的验证 | 正式vLLM对照 |
| --- | --- | --- | --- |
| Lesson2 调度 | scheduler/block_manager准入契约：KV块与linear state slot联合检查；Qwen3.5功能边界检查 | CPU预算trace；状态slot耗尽不分KV；KV分配失败回滚slot；被拒绝请求不推进计数 | V1 Scheduler.schedule预算及混合状态约束 |
| Lesson3 Runner | 执行批次携带request_id、token范围、状态slot；异常路径清理context | [A,B]→[B,A]映射；token slot与state slot不同；forward抛错后上下文干净 | Worker/ModelRunner输入准备与执行边界 |
| Lesson4 模型 | text_config、模型选择、权重映射、Full/Linear层映射及状态池接口 | 32层类型核对；8层Full KV映射；权重缺失/多余检查；状态形状、dtype、初始化检查 | Qwen3.5模型和混合缓存管理 |
| Lesson5 生命周期 | 状态池接入准入、执行、结束与失败路径；真实单请求控制链 | 两请求CPU隔离、重复释放/陈旧句柄、slot再用清理；真实短生成并记录状态轨迹 | 输出结算、释放及在途资源生命周期 |
| Lesson6 集成 | 独立完成一项改动；梳理配置→调度→状态→runner→模型→回收 | 回归测试、限制清单、来源说明；与InferLab接口映射 | 解释简化范围，不声称复刻生产vLLM |

每课顺序：读现有代码 → 预测失败 → 写最小测试 → 小范围修改 → 回归 → 学生解释。上表测试名和接口是验收目标，不代表已有可运行测试。Lesson2先学习调度并形成准入契约；状态池实体在Lesson4建立，再在Lesson5接通，不能拿mock通过替代真实模型证明。

## 第一版必须守住的约束

1. Full Attention原始层[3,7,11,15,19,23,27,31]映射紧凑KV层0..7；另外24层是linear状态，不分配普通逐token KV。
2. 请求获得有效KV资源和linear状态所有权后才能执行。分配中途失败必须回滚本次新增资源，不释放别的请求的资源。
3. 只命中Full Attention前缀不能跳过整个混合模型前缀：还需要正确边界的conv/recurrent状态。首版关闭或拒绝该路径，不能只清零linear状态后沿用KV命中计数。
4. 首版不支持未经验证的chunked prefill、抢占恢复、CUDA Graph；检查发生在状态突变前。完整prompt超过单轮预算时明确拒绝或要求调整配置，不能永久留在waiting死等。
5. 无资源时正常等待；必须保持资源计数不变并能在资源释放后继续。永久不可执行与暂时资源不足要区分。
6. 并发、共享前缀、抢占、Graph分别是后续功能，不因单请求跑通自动成立。

## 两个里程碑不能混淆

- M1：nano-vLLM控制链＋完整Transformers参考forward桥接。用于检查请求/停止/状态生命周期，不声称nano原生PagedAttention已经消费Qwen3.5缓存。
- M2：原生适配模型消费引擎提供的布局与状态，完成模型加载、Full/Linear执行、正确性对拍和生命周期验证。不能只改模型名或import。

Week4优先框架掌握、CPU状态契约与最小接入，不能保证本周完整M2。后续围绕同一适配副本推进：原生模型正确性 → 多请求与分块状态一致性 → profiling与性能改进。量化、多模态、投机解码暂不加入主线。

Week3真实模型数值结果仍为diagnostic；不因课程推进自动转为通过。真实性能结论必须在正确性与资源隔离通过后另行测量。
