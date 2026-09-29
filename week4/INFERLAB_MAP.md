# InferLab：本周只建立框架映射

实际阅读目录：/home/ubuntu/infra-main。本轮查看README、对应目录与分页/调度头文件，未构建或跑测试；不是已验证完整项目。

## 三层职责

README将项目分为原生C++/可选CUDA机制层、Python实验编排层、结果产物层。
学习时先找机制接口，再看一个测试，不从全部benchmark或治理脚本入手。

| 已学习概念 | InferLab 阅读入口 | 本周要回答 |
| --- | --- | --- |
| 请求ID与块映射 | native_core/include/inferlab/paged_kv_cache.hpp | RequestBlocks、BlockLocation分别表示什么？ |
| 块分配与前缀引用 | native_core/src/paged_kv_cache.cpp；native_core/tests/paged_kv_cache_test.cpp | 接口和测试怎样表达分配、共享、释放？ |
| 请求生命周期 | native_core/include/inferlab/continuous_batch_scheduler.hpp | InferenceRequest与RequestRuntime为什么分开？ |
| 调度 | native_core/src/continuous_batch_scheduler.cpp；native_core/tests/continuous_batch_scheduler_test.cpp | 一步调度推进了什么？ |
| Decoder调用边界 | native_core/include/inferlab/decoder_kv.hpp；native_core/src/decoder_kv.cpp；native_core/tests/decoder_kv_test.cpp | 模型计算与缓存状态在哪相接？具体能力需读实现核实 |

头文件的RequestLifecycleMetrics包含queue_wait_steps、time_to_first_token_steps等字段：它们是步数口径，不能直接当毫秒TTFT，更不能把机制模拟数据包装为L40实测吞吐。

本周输出一页 week4/notes/inferlab_mapping.md：画出nano-vLLM职责与这些接口的对应关系，明确“名称类似不代表实现相同”。
暂不承诺完整C++重写、不运行全部仓库构建、不改动InferLab；若后续需要运行，另选一个小测试目标并先检查构建说明。
