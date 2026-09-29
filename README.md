# infer-engine 学习入口

当前主线：Qwen3.5-4B。工作目录：`/home/ubuntu/infer-engine`。

- [总路线图](ROADMAP.md)
- [现在开始：Week 4 逐课计划](week4/PLAN.md)
- [Week 4 Lesson 1：真实请求链路](week4/LESSON_01.md)
- [nano-vLLM 与正式 vLLM 对照](week4/FRAMEWORK_COMPARISON.md)
- [Week 4 结课验收](week4/ASSESSMENT.md)
- [Week 3 逐课记录](week3/PLAN.md)
- [运行环境与验证边界](week3/ENVIRONMENT.md)

第一、二周代码和性能数据保留。Week 3 已完成真实 Qwen3.5 tokenizer、参考生成、混合状态观察及数值/隔离诊断；正式数值容差未批准，不能声称验收通过。按用户选择进入 Week 4 框架学习与渐进适配，不把桥接或诊断冒充原生引擎适配完成。

## 归档说明

- `archive/old-models/QWEN3_06B.md`：旧模型教学记录，数值与命令不作为当前课程答案。
- `archive/development/superpowers/`：已批准的设计与详细工程计划，初学者无需先通读。
- `archive/development/execution-records/`：迁移前的开发日志、测试证据与任务进度；保留历史路径以便追溯，不应照抄历史工作区命令。

当前执行 week4/PLAN.md，archive 下保存第一阶段设计与历史计划；不要按旧任务状态重做已经交付的课程。本轮不改外部 nano-vLLM/InferLab 源码，也不改现有环境。
