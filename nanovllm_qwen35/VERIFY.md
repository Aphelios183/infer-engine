# Lesson4验证记录

## 后续增量：StateManager CPU同步原型

在同一环境执行全套 `python -m unittest discover -s tests -p 'test_*.py' -v`：24个测试通过（0.043s），其中12个配置测试、12个StateManager测试。覆盖slot0、池满、重复请求、未知查询、重复释放、跨层清零、邻居隔离、batch重排、初始化中途失败回滚、dtype保留、存储别名与非法池检查。使用小尺寸CPU张量，不分配真实32层模型或GPU状态。

初始化失败测试验证所有权与空闲集合恢复；无主slot的字节可能已部分清零，重试会重新初始化，不能将此称为模型历史快照回滚。CUDA及其他非CPU设备被明确拒绝。未测试或实现GPU在途事件、多线程、陈旧generation句柄、KV联合准入和真实模型forward。

以下为此前配置契约阶段的独立记录，仍保留原结果。

日期：2026-10-07。环境：现有 `/home/ubuntu/enter/envs/nanovllm/bin/python`。仅CPU，不加载模型权重、不分配GPU缓存。

## 实际执行

在适配副本根目录运行：

```text
python -m unittest discover -s tests -p 'test_*.py' -v
Ran 12 tests in 0.003s
OK
```

真实配置CLI：

```bash
python -m qwen35_adapter --config /home/ubuntu/huggingface/Qwen3.5-4B/config.json
```

输出确认：32层、8个Full层、24个Linear层；Full原始层为[3,7,11,15,19,23,27,31]；EOS声明248044；候选每层每请求conv形状[8192,4]、recurrent形状[32,128,128]。另外执行了两张映射互斥、完整覆盖0..31和预期索引相同的断言，全部通过。

配置SHA256：`ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c06221670`。

导入来源检查指向 `/home/ubuntu/infer-engine/nanovllm_qwen35/nanovllm/__init__.py`，只检查find_spec，不执行该包的GPU相关导入。

## 来源与安全边界验证

- 25个来源文件最初逐字节复制；再次检查原目录25个文件SHA256未改变。
- 副本中相对于来源仅nanovllm/config.py改变；新adapter/tests/docs另列，不冒充上游。
- Config入口测试用假的AutoConfig捕获越界调用，确认Qwen3.5在调用AutoConfig前抛NotImplementedError；没有启动Runner、NCCL或CUDA。
- 原模型源码文件nanovllm/models/qwen3.py已排除父目录models/忽略规则，防止遗漏源码。

## 不能推出的结论

不能推出模型数值通过、GPU状态dtype确认、Qwen3.5原生执行完成、性能收益或学生独立实作通过。原有Qwen3 GPU路径本次未回归；其源码仅增加配置类型guard，不声称完全验证兼容性。
