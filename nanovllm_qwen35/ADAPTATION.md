# Qwen3.5适配副本：Lesson4起点

目录：`/home/ubuntu/infer-engine/nanovllm_qwen35`。当前只完成配置契约、层映射和明确拒绝未实现执行路径；不能生成Qwen3.5文本。

## 来源与贡献

2026-10-07从 `/home/ubuntu/t1/nano-vllm-main/nano-vllm-main` 复制25个源码/说明/入口文件，保留用户中文注释。不复制wheel、egg-info、pycache、模型权重或Git元数据，不创建嵌套Git仓库。

`SOURCE_MANIFEST.json`记录复制前各文件SHA256，来源是用户本地快照，不能声称某个已核实的上游commit。来源pyproject声明MIT并指向GeeeekExplorer/nano-vllm，但本地没有LICENSE文件；未自行编造许可证。对外分发前需从匹配来源补齐并核对许可证。

- `nanovllm/`：参考引擎副本；当前仅在config.py增加Qwen3.5未实现路径guard。
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

普通KV分配仍是上游实现；StateManager已完成独立CPU同步原型，见[接口与测试](STATE_MANAGER.md)，尚未接入Scheduler/Runner。Qwen3.5原生模型、权重映射、linear算子接入均未实现。Qwen3.5配置会在AutoConfig和Runner启动前被明确拒绝，不能误送Qwen3ForCausalLM。

首个将来可运行版本限定TP=1、eager、纯文本、单请求；前缀复用、分块Prefill、抢占恢复与Graph需独立正确性验收。当前整体执行拒绝并不等于这些细粒度功能开关已经实现。

课程：[Lesson4](../week4/LESSON_04.md)。测试结果：[VERIFY](VERIFY.md)。
