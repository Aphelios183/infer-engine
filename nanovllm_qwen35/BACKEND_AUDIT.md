# 真实Qwen3.5后端核查

日期2026-10-08；只读本机模型配置、权重索引/文件头、已安装参考源码，并运行小型CPU缓存类探针。不加载模型权重、不运行完整forward、不使用GPU、不修改环境或执行guard。

## 可复现入口

```bash
cd /home/ubuntu/infer-engine/nanovllm_qwen35
/home/ubuntu/enter/envs/nanovllm/bin/python -m qwen35_adapter.audit_backend --model-dir /home/ubuntu/huggingface/Qwen3.5-4B --probe-cache
```

脚本只读safetensors文件头，不读张量payload；这不是权重完整性或数值加载验收。源码SHA随命令输出，防止升级环境后误用本报告。

环境：torch 2.5.1+cu124、transformers 5.8.0、safetensors 0.7.0。`fla`和`causal_conv1d`模块均未找到，相应distribution未安装；本机参考代码有torch fallback。本报告不是上游最新版说明，也未证明当前环境可直接跑优化GDN内核。

## 1. checkpoint不是原nano可以直接遍历加载的文件

实际738个张量：文本426、vision297、MTP15；690个BF16、48个F32。文本前缀为`model.language_model.`。纯文本首版需要显式允许忽略vision/MTP，同时严格检查文本键缺失/重复/意外键，不能全局strict=False掩盖错误。

没有独立lm_head权重，配置tie_word_embeddings=true；embedding权重[248320,2560]，输出头需要正确共享。原`nanovllm/utils/loader.py`按名称直接get_parameter并对子串做packed替换，不能原样加载这个checkpoint。

## 2. 关键形状与语义

下表来自真实文件头，Linear层取原始0、Full层取原始3；Linear权重形状均按[out,in]：

| 模块 | 实际形状 | 适配要求 |
| --- | --- | --- |
| Full q_proj | [8192,2560] | 16×256×2，按每个head拆Q/gate；不是32个Q头，也不是全局前半Q后半gate |
| Full k_proj/v_proj | 各[1024,2560] | 4个KV头，每头256 |
| Full o_proj | [2560,4096] | attention输出先乘sigmoid(gate)，再输出投影 |
| GDN in_proj_qkv | [8192,2560] | Q2048、K2048、V4096，不等长三分 |
| GDN in_proj_z | [4096,2560] | gated norm所需z |
| GDN in_proj_a/b | 各[32,2560] | 时间衰减/更新门控，不是普通QKV |
| GDN conv1d | [8192,1,4] | depthwise causal conv |
| GDN out_proj | [2560,4096] | recurrent输出回hidden维度 |

另有关键差异：

- Full Q/K norm及普通层RMSNorm使用`1 + weight`，而原nano直接乘weight；复制参数但不改公式会算错。GDN的RMSNormGated又是直接乘weight后乘SiLU(z)，不能给所有norm一律加1。
- 部分RoPE：head_dim=256，partial_rotary_factor=0.25，即rotary_dim=64；原nano Qwen3使用全head旋转。纯文本仍须遵循参考position与旋转布局；本次不启用多模态M-RoPE。
- 文件头中GDN A_log与norm.weight为F32，dt_bias为BF16。checkpoint dtype、模型加载后参数dtype、算子累加dtype和持久state dtype是四件事，不能全局转换后一概宣称对齐。

## 3. GDN持久状态dtype：已查明一条转换路径

参考fallback会把递推主要输入升至FP32，返回FP32 recurrent state。但已安装`LinearAttentionLayer.lazy_initialization`先从conv输入记录dtype，再以该dtype分配recurrent存储；update_recurrent_state执行copy_，可能转换dtype。

实际小型CPU探针结果：conv BF16、输入recurrent FP32、存储recurrent BF16；值1.0010000467写入后为1.0。这说明为什么“配置写float32”仍不足以证明持久状态是FP32，也为Week3观察提供了可验证解释。

边界：只验证本机缓存类这条路径，未再次运行真实模型。不能因此决定原生pool一律BF16或FP32。下一次真实单请求探针必须记录prefill/decode各层实际state、算子返回值和最终存储，并按选定参考路径批准误差容限。

## 4. Runner接口不直接等于HF接口

我们的输入为packed一维token、cu_seqlens、分页地址和state_slots；参考TextModel接收[B,T]、position_ids及Cache，Full层用cache.update，GDN用原始layer_idx访问cache.layers。

- M1参考桥接：先单请求，使用独立HF cache建立参考生成/数值基线。必须明确这不消费我们的分页KV池，不能宣称原生适配完成。
- M2原生后端：消费Full紧凑层映射、block_tables/slot_mapping，以及Linear紧凑层映射/state_slots；每层持久状态必须确实写回池。不能把池按slot高级索引得到的副本当作原地更新视图。
- GPU执行需要新的同步/事件与失败隔离契约，不能将CPUExecutionRunner的cpu_sync标签换成GPU就回收资源。

## 5. 源码定位（本机版本）

参考文件根：`/home/ubuntu/enter/envs/nanovllm/lib/python3.11/site-packages/transformers/`。

- `models/qwen3_5/modeling_qwen3_5.py:358` GDN；424 forward；455单token有状态conv；509递推分支；522 chunk分支；535 recurrent写回。
- 同文件631 Full Attention；669按head拆Q/gate；700输出门控；722普通RMSNorm；175 Gated RMSNorm；124 RoPE维度。
- 同文件1238 TextModel.forward，1255 DynamicCache创建，1283逐层执行；1701文本CausalLM，1766 lm_head。
- `cache_utils.py:768` state延迟分配，823 recurrent写回，1099按层查询历史存在。
- nano对照：`nanovllm/models/qwen3.py`、`nanovllm/layers/layernorm.py`、`nanovllm/utils/loader.py`。

## 下一交付与验收

先做纯文本权重加载计划及形状/覆盖检查（不加载GPU），明确norm、Q/gate、RoPE差异；再实现单请求参考执行与真实state观测，最后接原生Full/GDN与池写回。优化与恢复策略统一记在[待办](OPTIMIZATION_BACKLOG.md)。

课堂题：Full层q_proj输出8192维，为什么不是32个Q头？回答时说清Q/gate拆分、Q头数和K/V头数。

## 6. 纯文本权重映射与覆盖检查（2026-10-08已执行）

实现`qwen35_adapter/weight_plan.py`，测试`tests/test_weight_plan.py`。不另建重复说明文档。

```bash
cd /home/ubuntu/infer-engine/nanovllm_qwen35
/home/ubuntu/enter/envs/nanovllm/bin/python -m qwen35_adapter.weight_plan --model-dir /home/ubuntu/huggingface/Qwen3.5-4B
# 添加 --full 可查看全部逐项映射与明确排除的权重键。
```

实际结果：426/426文本权重匹配，文本中BF16=378、F32=48；明确排除vision297、MTP15。没有遗漏或额外文本键，形状全部符合本机配置。`lm_head.weight`登记为`model.embed_tokens.weight`的共享别名，不是第427份需要从文件加载的独立权重。

映射规则将`model.language_model.`精确前缀改为`model.`，内部投影暂时保持分开，不使用原nano的qkv/gate_up融合加载器。目标名称是拟定原生后端的参数契约，尚无已实例化模型可做named_parameters覆盖对拍。

每项保留source、target、shape、checkpoint dtype、shard和semantics。Full q_proj保留每头Q/gate交错顺序；普通RMSNorm登记one_plus_weight，GDN gated norm登记direct_weight。这些是执行语义标签，本次没有实际转换参数或执行norm。

文件检查还验证：重复JSON键、跨分片重复tensor、索引与文件头一一对应、byte range与shape/dtype匹配、重叠/空洞/尾部未登记payload和总字节数。读取仅限头部和文件大小，不读或校验权重payload内容，因此不等于文件内容校验和或真实加载成功。

9项新增单测覆盖正常映射/别名/输入不变、缺文本权重、未知命名/额外文本/意外独立head、明确排除视觉/MTP、Q/gate错误形状、norm/layout标签、非法dtype、不支持的untied配置、重复JSON键。文件头合法路径另由真实checkpoint命令验证；本次没有声称所有文件损坏分支都已有单测。

只支持本课dense、无attention bias、有output gate、SiLU、共享embedding的模型契约。支持BF16/F16/F32文件头并保留原dtype，不决定GPU运行dtype，不支持量化checkpoint。下一步是实际模型参数清单与加载器实现、别名存储验证，再做真实forward。

## 7. 参数结构与实际CPU加载（2026-10-08已完成）

第6节是此前映射计划阶段。本阶段实现`qwen35_adapter/parameters.py`和`weight_loader.py`，测试`tests/test_weight_loader.py`。

`Qwen35TextParameters`是注册到nn.Module的参数存储骨架，不是可执行Transformer。model.layers为ModuleList，每个Full/Linear层有对应投影、norm、MLP参数节点；embedding、最终norm和lm_head在顶层约定位置。首次构造全部使用meta设备，不随机初始化、不分配真实权重内存。forward始终明确拒绝，loaded=True也不代表能推理。

加载步骤：

1. 严格检查config、索引、文件头与文本映射计划。
2. 建立meta骨架，对照实际named_parameters，必须与计划目标集合相同。
3. 按分片仅取文本tensor，检查shape/dtype，clone为CPU独立存储，替换对应Parameter；保留checkpoint混合dtype和原始布局，不做packed融合、全局cast或norm参数改写。
4. 重新绑定lm_head.weight与model.embed_tokens.weight为同一Parameter。替换embedding后，旧meta别名不会自动跟随，所以必须重新绑定。
5. 检查426个目标均已加载，无meta残留、无意外参数、无梯度；别名身份与存储均一致，最后才标loaded=True并返回模型与报告。

加载器只创建新对象，不接受正在执行的模型做原地热加载。发生错误不会返回部分加载模型，但不承诺异常traceback保留期间所有CPU内存立即释放。checkpoint应在加载期间保持不变。语义标签用于后续forward实现，不能把保留Q/gate字节布局等同于已经实现Attention。

```bash
cd /home/ubuntu/infer-engine/nanovllm_qwen35
/home/ubuntu/enter/envs/nanovllm/bin/python -m qwen35_adapter.weight_loader --model-dir /home/ubuntu/huggingface/Qwen3.5-4B --verify-values
```

本次检查主机内存余量后实际执行完整CPU加载：loaded_count=426，verified_tensors=426；BF16参数378、FP32参数48；唯一参数字节8,411,510,272。视觉297、MTP15明确排除；head_is_embedding_parameter=true、head_storage_matches=true。CLI进程执行完毕已退出，没有留下常驻模型，不占用GPU。

verify-values逐个对已加载参数与对应文件tensor做全量字节比较，包括NaN位模式；它证明拷贝与映射结果一致，不证明有限性、计算语义、数值容差或推理正确。唯一参数字节不是进程峰值RSS；加载期间另有文件映射、临时张量和运行时开销。

7项新增测试：meta结构/forward保护、全部值和dtype及共享别名、修改参数不影响文件/重新加载值不变、缺键提前拒绝、错误shape、部分加载异常不标记完成、模型注册目标额外键拒绝。使用小型双分片checkpoint；真实4B加载另以以上命令验证。全套92项通过（0.206s）。

下一步是有实际计算的文本模型层：先按参考公式实现norm、Q/gate拆分与partial RoPE，单层对拍后接Full/GDN缓存。原LLMEngine和GPU执行保护不变；不能把本参数骨架直接当成CPUBackend.forward实现。

## 8. 真实层计算与缓存续算（2026-10-08已完成）

实现`qwen35_adapter/layer_compute.py`；测试`tests/test_layer_compute.py`。计算文件不导入Transformers，独立实现PyTorch公式；测试对照本机Transformers 5.8.0的Qwen3.5参考层（源码定位与版本见前文）。这是CPU eager、单请求[1,T,H]实现，不是Triton/FLA优化后端。

阅读顺序：

1. `rms_norm`：FP32计算均方及归一化，乘1+weight后转回输入dtype。`gated_rms_norm`直接乘weight，再乘SiLU(z)，按参考实现保留转换顺序。
2. `split_q_gate`：reshape到[...,heads,2*head_dim]再拆，不能把整个投影全局二分。
3. `partial_rope`：支持default纯文本position，[B,H,T,D]中仅前rotary_dim旋转，尾部保持原值；不支持动态RoPE或不同时间/空间坐标的多模态输入。
4. `full_attention`：投影→Q/K norm→RoPE→历史K/V拼接→GQA→带历史偏移的因果mask→softmax→Attention输出门控→o_proj。返回FullState，不修改传入旧状态。
5. `gated_delta_net`：QKV投影→depthwise causal conv+SiLU→Q/K L2归一化/头扩展→FP32逐token gated delta递推→gated norm→out_proj。返回conv/recurrent和处理长度，不修改传入旧状态。Prefill使用慢速串行递推，不是FLA chunk kernel。
6. `decoder_layer`：从已注册参数节点读取权重，连接输入norm、Full/GDN混合器、残差、post-attention norm、SiLU gate/up/down MLP和第二次残差。

用法示意（参数来自已加载的存储骨架）：

```python
output, new_state = decoder_layer(
    hidden_states, positions, parameters.model.layers[layer_idx],
    parameters.config['text_config'], layer_idx, state=old_state,
    state_dtype=torch.float32,
)
```

调用需要CPU单请求隐藏状态，以及与历史长度连续的位置。FullState存储稠密K/V；GDNState存储conv和recurrent。它们不是现有分页KV/StateManager池的视图，不消费slot_mapping/state_slots。当前没有把它们接入CPUExecutionRunner或原LLMEngine；下一阶段必须设计显式读池/写回，不能将状态副本更新误认为池已改变。

### 测试证据

12项新增测试通过，全套104项通过（0.182s）。FP32层对拍/续算检查中记录的最大绝对差为8.568167686462402e-08，验收阈值atol=2e-5、rtol=2e-4；这是小尺寸随机权重的测试阈值，不批准完整4B/BF16模型的数值容差。

- 两种norm与partial RoPE：FP32/BF16输入对参考结果精确一致；RoPE未旋转尾部精确不变。
- Q/gate：手工索引排列和错误宽度拒绝。
- Full：参考Prefill→两步Decode输出与K/V一致；缓存续算对完整前缀、旧cache不变、位置错配拒绝。
- GDN：参考Prefill→Decode输出及conv/recurrent一致；65-token跨参考chunk边界；分段续算对完整前缀、旧状态不变；1/2/3-token短prompt、显式状态dtype与错配拒绝。
- 两种Decoder层：将参考随机权重复制到我们注册的参数骨架，比较完整norm/mixer/residual/MLP计算；多请求输入明确拒绝，不默默串联。

### dtype与部署边界

GDN默认持久recurrent为FP32；可以显式选择BF16，用于后续研究参考cache舍入。BF16策略在每次调用结束转存，调用切分可能影响舍入，不能由FP32续算通过推出BF16分块等价。本次没有跑完整BF16 GDN层对拍，只检查显式state存储选择；也没有运行真实4B权重的全模型forward。

参考环境缺少FLA/causal-conv1d，测试走torch fallback，未安装依赖。保留Qwen3.5引擎guard、ParameterNode.forward拒绝和CPU同步执行限制。层函数可算不等于完整模型能生成，更不等于分页、多请求、GPU或性能优化已完成。

## 9. 层计算接入真实CPU资源池（2026-10-08）

实现`qwen35_adapter/pool_backend.py::PooledCPUBackend`，测试`tests/test_pool_backend.py`。本节更新第8节的“未接池”状态：已连接真实HybridScheduler/AdmissionController/BlockManager/StateManager和CPUExecutionRunner，但仍不是原LLMEngine或GPU后端。

### 数据路径

调用方提供CPU连续KV张量，布局`[2, Full层数, 物理块数, block_size, KV头数, head_dim]`，参数必须完全加载。构造时验证KV/conv/recurrent的形状、dtype、设备及不共享底层存储；不按全部原始层分普通KV。

Full：原始layer_id → full_map紧凑索引 → 本层K/V池 → 根据请求block_table逐个逻辑位置生成物理地址 → gather历史K/V → 真实层计算 → 按slot_mapping仅写入本轮新增K/V。无效容量槽不参与读取，历史KV不被重写。

GDN：原始layer_id → linear_map紧凑索引 → 本轮请求state_slot → 读取对应conv/recurrent切片 → 真实层计算 → 对原池的整数索引视图执行copy_写回。不能对高级索引副本做更新后就认为已写回池。

每层按cu_seqlens_q分开处理各请求，然后按原batch顺序拼回hidden_states。首次Prefill从空历史算起；Decode保留各请求历史，不每轮初始化。成功forward本身不推进调度计数；CPUExecutionRunner在采样成功后统一调用postprocess。

```python
backend = PooledCPUBackend(scheduler, loaded_parameters, kv_pool)
runner = CPUExecutionRunner(scheduler, backend, sampler)
batch = scheduler.schedule()
if batch is not None:
    result = runner.run(batch)  # 已包括采样和结算，不再重复postprocess
```

backend.forward先重新构造当前pending batch的合法输入，逐字段验证传入元数据，防止改错slot/过期输入；同一batch开始执行后禁止重复forward。资源仍由管理器拥有，后端只读写张量，不分配块、不回收请求、不替换state映射。

### 实际测试

新增9项，全套113项通过（0.319s）。使用4层小模型（Linear、Full、Linear、Full），随机FP32参数，真实管理器、小型真实张量池；不是4B权重运行。

- Prefill：每请求输出及各层KV、conv/recurrent与独立稠密完整前缀计算对照；同时验证原始层号与紧凑索引不混用。
- Decode：请求重排后slot仍归原请求，跨物理块扩容后正确读历史、写新位置，旧KV字节不变，未使用槽保留哨兵值。
- 隔离：只让A执行时，仍在运行的B的KV/conv/recurrent完全不变；空闲state slot不被改写。
- 防错：篡改state_slots或把A写入地址指向B时，在任何池写入前拒绝；同一批次第二次forward拒绝；错误KV池形状拒绝。
- 控制链：真实计算后合成EOS采样，由CPUExecutionRunner结算并回收，未把合成EOS冒充模型生成。
- 失败：第一层GDN已写池、下一层抛错时整批失败回收；脏字节仍可存在，再分配确实复用脏slot0并验证两类状态全清零。失败回收不是张量回滚。

### 限制与下一步

仅串行CPU执行，批次内逐请求循环；Full Attention先从分页池gather出稠密历史，GDN使用顺序递推。支持当前完整Prefill/每请求单token Decode，不启用prefix、chunk调度、抢占或Graph。不能宣称实现GPU PagedAttention内核或性能优化。

默认测试状态为FP32。完整4B、混合BF16、参考模型端到端logits和真实文本生成未在本次验证。ParameterNode仍只是参数存储；后端通过独立层函数执行，不调用ParameterNode.forward。下一步对固定短输入进行真实权重的端到端对拍，并独立设计GPU池与同步生命周期，不能将cpu_sync标记直接改成GPU执行。

## 10. M1/M2衔接：单卡真实4B逐层GPU诊断

2026-10-08完成。代码`qwen35_adapter/gpu_reference.py`；`layer_compute.py`扩展为同设备CPU/CUDA数学计算。CPUExecutionRunner、StateManager和PooledCPUBackend仍保持CPU限制，未绕过GPU资源生命周期保护。

物理GPU2，NVIDIA L40 46068MiB；开始时利用率0、占用约295MiB，有各卡共有进程的基础占用，因此不是独占性能实验。诊断只限一张卡，未停止其他进程。参考与自实现同时驻留，peak allocated为17,134,337,024字节；含诊断开销，不能当成单引擎最低显存需求。命令结束后进程退出。

### 参考与实验设置

安装的Transformers 5.8.0，Qwen3_5ForCausalLM + eager attention + torch GDN fallback；使用严格文本加载器得到的426项相同权重，保留checkpoint混合BF16/FP32。用meta构造参考类、严格assign参数并重建非持久RoPE buffer，不等同于所有from_pretrained默认dtype策略。双方持久recurrent显式对齐本机参考BF16，内部递推FP32；禁用TF32。

固定问题“请用一句话说明KV缓存的作用。”，套模板一次后19 tokens，运行Prefill和2次Decode。每步以参考选出的token共同作为下一步输入（teacher forcing），不让生成分支差异污染对拍。不是完整答案生成或质量评测。

记录每层输出的累积误差，另把参考的该层输入送入自实现来观察局部误差。局部路径Decode仍使用自己的历史cache，故不等于每一步都重置到相同参考缓存。逐层KV、conv/recurrent也记录误差。临时报警标准atol=0.05、rtol=0.01仅用于诊断分流，不是获批数值验收标准；不因argmax相同宣称通过。

### 原始chunk参考结果

| 阶段 | logits最大绝对差 | logits相对L2 | 两路选出token |
| --- | --- | --- | --- |
| Prefill | 0.16015625 | 0.0157319 | 79852 / 79852 |
| Decode1 | 0.25 | 0.0149887 | 220 / 220 |
| Decode2 | 0.1875 | 0.0153390 | 111734 / 111734 |

三个token解码为“KV 缓存”。首个非零差异在原始第0层GDN Prefill，局部最大差0.0009765625。相同输入下8个Full层在本例三步的输出误差均为0；不能因此把它们接收前层误差后的累积偏差归因于Full公式错误。后层会传播和放大差异。

原始结果保存`gpu_reference_results.json`，包含32层×3阶段的输出与状态指标。所有最终logits有限，但未满足当前临时close判定，M1不标通过。

### 受控消融：只更换参考的GDN Prefill计算路径

参考层的chunk_gated_delta_rule临时换为同参考源码的torch_recurrent_gated_delta_rule；只影响本进程参考实例，不修改安装包或模型文件。自实现代码、权重、dtype、输入和其他模型结构不变。

结果：Prefill、Decode1、Decode2的每层累积输出、局部输出与最终logits最大误差均为0。结果保存`gpu_recurrent_ablation.json`，明确标注reference_prefill=recurrent_ablation。

这一对照支持本例差异来自chunk与逐token递推的数值路径，不是简单的层权重映射或Full计算错误。不能用修改过的参考替代原始参考宣布M1通过，也不能从一个短prompt推广到长上下文/所有输入。

### 复现

```bash
cd /home/ubuntu/infer-engine/nanovllm_qwen35
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/ubuntu/enter/envs/nanovllm/bin/python -m qwen35_adapter.gpu_reference --model-dir /home/ubuntu/huggingface/Qwen3.5-4B --output gpu_reference_results.json
# 仅作定位，不作为原始参考验收：
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/ubuntu/enter/envs/nanovllm/bin/python -m qwen35_adapter.gpu_reference --model-dir /home/ubuntu/huggingface/Qwen3.5-4B --output gpu_recurrent_ablation.json --recurrent-reference
```

首次试跑在构造token张量时遇到当前tokenizer返回类型差异，未进入forward；已改为模板渲染字符串后tokenize(add_special_tokens=False)，避免重复特殊token。

下一步：对齐原始chunk Prefill实现或制定经过更多输入/长度验证的数值标准，再将GPU层执行接入GPU池和同步生命周期。本次已建立真实单卡参考路线和首个误差来源定位，没有声称GPU推理框架全部完成。纯CPU测试.md已归档删除，测试代码保留用于回归。
