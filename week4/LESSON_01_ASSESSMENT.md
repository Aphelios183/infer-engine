# Lesson 1 概念验收记录

日期：2026-09-30。结论：根据本次对话，概念验收通过，进入Lesson2；不等于独立实现、真实模型或整周验收通过。

已掌握并纠正：

- add_request → waiting → schedule → ModelRunner → postprocess 的职责。
- prompt=[11,22,33]，生成44、55：Prefill后tokens=4、cached=3；Decode结束释放前tokens=5、cached=4。
- deallocate后tokens仍为[11,22,33,44,55]，num_tokens=5，cached=0，block_table=[]；不是全部清空。
- 最后选出的55因达到生成上限，不再送入forward；不写缓存的原因不是前缀复用。
- 逻辑位置、块内偏移、物理块和token slot；分配容量不等于有效历史。
- batch重排时状态按request映射；重置位置不能清除conv/recurrent历史。
- 正式V1以每请求计算进度与预算组织调度；waiting/running不等价于模型执行阶段。
- grammar约束不是正则化；非阻塞提交不自动证明实测CPU/GPU重叠。

仍待实作：学生独立写出源码trace、运行CPU契约测试、完成并解释一项小修改。这些保留到逐课练习及结课验收，不标记完成。
