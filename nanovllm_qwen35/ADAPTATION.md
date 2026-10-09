# Qwen3.5适配副本：Lesson4起点

目录：`/home/ubuntu/infer-engine/nanovllm_qwen35`。当前已通过显式`nanovllm.qwen35.Qwen35LLM`入口接通单卡eager纯文本GPU生成；原`nanovllm.LLM`的Qwen3.5 guard仍保留，不能把新模型误送旧Qwen3 Runner。首版使用参考torch chunk、稠密历史gather与逐请求计算，不是高性能生产引擎。

## 来源与贡献

2026-10-07从 `/home/ubuntu/t1/nano-vllm-main/nano-vllm-main` 复制25个源码/说明/入口文件，保留用户中文注释。不复制wheel、egg-info、pycache、模型权重或Git元数据，不创建嵌套Git仓库。

`SOURCE_MANIFEST.json`记录复制前各文件SHA256，来源是用户本地快照，不能声称某个已核实的上游commit。来源pyproject声明MIT并指向GeeeekExplorer/nano-vllm，但本地没有LICENSE文件；未自行编造许可证。对外分发前需从匹配来源补齐并核对许可证。

- `nanovllm/`：参考引擎副本；config.py增加Qwen3.5未实现路径guard，__init__.py按需加载LLM以支持真实资源模块的CPU测试。
- `qwen35_adapter/contracts.py`：学生提交的build_layer_maps保留算法，注释改为原始层号；助手补配置校验和guard。
- `qwen35_adapter/__main__.py`：仅读取config.json，验证导入来源，不加载torch或权重。
- `tests/test_contracts.py`：助手编写的CPU测试，正常映射断言来自学生；不代表学生独立完成全套测试。
- 原README、example.py、bench.py保留为来源资料；其中示例不代表Qwen3.5已适配，不要直接用它们跑新模型。

## 固定运行入口

```bash
cd /home/ubuntu/infer-engine/nanovllm_qwen35
/home/ubuntu/enter/envs/nanovllm/bin/python -m unittest discover -s tests -p 'test_*.py' -v
/home/ubuntu/enter/envs/nanovllm/bin/python -m qwen35_adapter --config /home/ubuntu/huggingface/Qwen3.5-4B/config.json
```

不要pip install -e，不升级环境，不下载第二份权重。当前从副本根目录运行，以此目录优先解析包；CLI打印并核查nanovllm导入位置，避免误用已安装包。直接从别处执行example.py仍不属于已验证入口。

## 不能越过的边界

普通KV分配仍是上游实现；StateManager已完成CPU同步原型，见[接口与测试](STATE_MANAGER.md)，并通过AdmissionController接入[HybridScheduler](SCHEDULER.md)。这是独立CPU控制链，尚未接入LLMEngine/ModelRunner。Qwen3.5原生模型、权重映射、linear算子接入均未实现。Qwen3.5配置会在AutoConfig和Runner启动前被明确拒绝，不能误送Qwen3ForCausalLM。

首个将来可运行版本限定TP=1、eager、纯文本、单请求；前缀复用、分块Prefill、抢占恢复与Graph需独立正确性验收。当前整体执行拒绝并不等于这些细粒度功能开关已经实现。

课程：[Lesson4](../week4/LESSON_04.md)。当前诊断结果：[后端核查](BACKEND_AUDIT.md)。输入准备说明：[Runner输入准备](RUNNER_INPUTS.md)；剩余适配清单见[ADMISSION](ADMISSION.md)。以下保留各阶段边界记录，最新进展见文末。

后续增量：[CPU执行接口](EXECUTION.md)已用合成后端连接forward/logits/采样/结算，新增11项测试；仍未接入真实Qwen3.5模型与原LLMEngine，不支持异步GPU后端。

最新：[纯文本权重映射计划](BACKEND_AUDIT.md)已检查426项文本权重，未实际加载模型。

后续更新：同一报告第7节已完成参数存储骨架和严格CPU加载器，真实426项文本权重逐字节验证通过；head共享embedding。没有Transformer计算实现，不支持forward，不解除Qwen3.5执行guard。

再更新：第8节已有独立CPU单请求Full/GDN及Decoder层计算，12项新增对拍测试通过。ParameterNode整体forward仍拒绝；层函数尚未接Runner分页池、完整模型或GPU，不解除引擎guard。

最新进度：第9节PooledCPUBackend已将上述层计算接入现有CPU分页KV和linear状态池，并接CPUExecutionRunner，9项新增测试、全套113项通过。批次内逐请求计算；完整4B参考对拍、原LLMEngine和GPU执行仍未验收。上面的未接池描述为历史阶段记录。

2026-10-08 GPU诊断：第10节已在物理GPU2的L40上运行真实4B、19-token Prefill与两步Decode，逐层比较原始HF chunk参考和受控recurrent参考。原始参考logits仍有差异；recurrent消融三步全层/最终logits均为0差异。仅诊断，不是原始参考M1验收，也未接GPU调度/分页池。

按用户要求移除纯CPU测试文档：本目录VERIFY.md、week4/test/README.md与RESULTS.md。备份在`/home/ubuntu/infer-engine/archive/cpu-test-docs-2026-10-08/cpu-test-docs.tar`。保留测试代码、课程和接口说明，不删除历史GPU实验报告。

最新：BACKEND_AUDIT第11节记录GPU引擎接线、27个精确参考对拍步骤、资源池逐项一致与实际文本生成。运行说明同节；测试代码共118项通过。之前“不能生成”及未接GPU的段落属于阶段历史，当前只开放显式单卡eager入口，不开放旧LLM的TP/Graph等路径。

文档清理（2026-10-08）：移除适配副本中的README_CN.md和REQUIREMENTS_CN.md，因旧Qwen3说明/需求易与当前适配混淆；英文来源README和SOURCE_MANIFEST保留。删除前tar备份并用tar对比确认一致，恢复来源为`/home/ubuntu/infer-engine/archive/doc-cleanup-2026-10-08/old-qwen3-docs.tar`，只含这两个文件。原始nano来源目录未改，SOURCE_MANIFEST仍表示最初复制快照，不代表当前副本文件集合。
