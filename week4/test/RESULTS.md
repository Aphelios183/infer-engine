# 实测记录：2026-10-01

位置：`/home/ubuntu/infer-engine/week4/test`。使用现有nanovllm环境的Python标准库；未启动GPU、未修改环境、未修改nano-vLLM源码。

## 正常回归

```text
python -m unittest discover -s week4/test -p 'test_*.py' -v
Ran 12 tests in 0.002s
OK
```

## 故障注入与恢复

可在主目录复现：

```bash
/home/ubuntu/enter/envs/nanovllm/bin/python week4/test/check_mutations.py
```

实际输出摘要：

```text
EXPECTED FAILURE detected: _reset_slot
EXPECTED FAILURE detected: observe_batch
Ran 12 tests in 0.001s
OK
Original implementation restored; all 12 tests pass.
```

1. 临时跳过状态清零，复用测试捕获A残留进入C。
2. 临时按物理slot排序输出，重排测试捕获请求顺序与状态错配。
3. 两次都是预期的断言失败，不是import错误；mock上下文退出后恢复，再跑12个正常测试全部通过。

## 结论

通过的是独立CPU教学模型的资源与状态规则；测试确实可以抓到上述两种人为错误。耗时仅为测试日志，不是推理性能指标。

尚未验证：真实nano-vLLM接入、Qwen3.5逐层张量、GPU forward、并发/异步、完整混合前缀复用、模型正确性与性能。助手编写并运行测试，不等于学生已独立完成实作；建议学生阅读两个失败用例并解释其断言后再进行个人验收。
