# Qwen3.8 DFlash2：带缓存短块 prefill attention 优化

日期：2026-10-12。实现 checkpoint：`5d90b2e`；基线：`1c56042`。
当前状态：实现、CUDA回归、三模式4K实验及独立输出审计全部完成。

## 实现

新增 `--short-prefill-attention` 开关（默认关闭），要求开启
`--vectorized-target-numerics`。同一内核用于target-only的cached decode、
DFlash2八位置验证及必要的缓存回放，原生prefill路径不变。

- QK以固定16行计算tile处理1–8个query，直接读取每个请求的compact KV；
  通过KV head索引实现GQA，不再展开KV或逐轮复制到整段padding缓冲。
- QK中融合缩放、有效长度和prefix-offset causal mask；保留BF16分数的两处舍入边界。
- softmax对所有位置一次调用；PV用同样的逐行累加规则并行计算query。
- 固定上下文padding和归约tile，保证同一query在单步／整块路径的运算一致。
- 不引入持久工作区或逐token完整GDN快照；新临时张量由CUDA峰值统计覆盖。
- attention内部不再逐位置QK/softmax/PV；LM head仍逐位置执行，线性投影未修改。

注意：新Triton attention与旧PyTorch矩阵乘不是逐位等价的数值实现。
不能宣称新路径与历史输出恒等；验收要求新target-only与新DFlash2严格逐token一致，
并独立记录新旧target输出差异。不能将改变target数值路径造成的接受率变化解释为
draft模型自身能力提高。

## 实现前数值探针

直接把旧tokenwise torch.matmul替换为整块torch.matmul，在真实4299上下文、
C4、24/4 heads、head_dim256随机输入中出现QK 509个元素和PV 26个元素不相同。
因此没有直接移除循环后声称精确等价。

新固定行Triton实现，context32/257/4299、head_dim4/256，8位置与逐步输出逐位一致。
实际pytest还覆盖3位置及异长prefix。

## 回归与微基准

服务器全量 `python -m pytest -q tests`：
**285 passed、2 skipped，50.20秒**。
包含新旧路径、4/8位置、接受0–7、异长batch、EOS、验证异常和提交异常rollback；
提交状态与逐步reference逐位相同。独立FP32数学参考检查只用于算子数值精度，
不能替代严格token或cache-state一致性验收。

attention微基准：C4、Q8、24/4 heads、head_dim256、固定4299上下文，
四个prefix为187/1000/2048/4000，5次预热、30次CUDA event计时。

| 指标 | 原attention | 新短块attention |
|---|---:|---:|
| 平均CUDA事件区间耗时 ms | 4.84236 | 0.20145 |
| 临时分配峰值 bytes | 529026560 | 33016320 |

这仅测attention核心与缓存整理，不含模型投影、GDN、LM head、proposal或commit，
也不是完整模型吞吐加速倍数。

## 完整实验协议

同一A800-SXM4-80GB，C4、K=7、一次验证8位置、micro-batch4。
同四个自然长代码／推理任务，输入187/194/190/201，各固定4096输出，含推理token；
greedy、ignore_eos、seed20260907。加载与预热排除正式计时。
跳过初始化最大窗口预热，在线预热4请求×64输出；正式请求始终4096。
关闭profiler与细分DFlash profiling，保留memory_trace。

初轮依次运行新target-only、新DFlash2、旧DFlash2（相同源码关闭新开关）。
新DFlash2必须与新target-only的原始request JSONL完全一致，否则停止；
旧DFlash2与既有旧target输出比较。新旧输出公共前缀另行报告。
每轮保存源码、diff、配置、模型snapshot路径、fairness、JSONL、阶段耗时和显存。

服务器证据：
`/root/autodl-tmp/Mini-vLLM/results/qwen38_dflash2_perf/short_prefill_20261012/`。
部署前保存完整原代码归档并核对被修改文件哈希，远端原有HEAD与未提交改动保留。

## 完整结果

三种模式各一次，顺序为新target-only、新DFlash2、旧DFlash2。
均为4请求×4096输出，未发生失败或取消。三个manifest源树SHA256相同：
`44137c26d04bd6d76e1cd26745e80af255577043af195aa33de8b9e0bffed8f2`。
本地与服务器六个实现／测试文件SHA256逐一相同。
参数差异仅实验名称、target-only/DFlash2模式及short-prefill-attention开关。

| 指标 | 新target-only | 旧DFlash2（8位置） | 新DFlash2（8位置） |
|---|---:|---:|---:|
| 吞吐 tok/s | 40.2235 | 39.3712 | **42.6830** |
| 端到端墙钟 秒 | 407.3237 | 416.1417 | 383.8531 |
| target decode / verify 秒 | 346.7473 | 283.0228 | **246.5053** |
| 每轮verify均值 ms | — | 210.4259 | **169.7695** |
| proposal 秒 | — | 27.9225 | 30.2044 |
| commit 秒 | — | 45.5360 | 48.4137 |
| 平均TTFT 秒 | 59.5377 | 59.2987 | 58.3521 |
| 平均TPOT ms | 84.9294 | 84.5357 | 75.2155 |
| 请求TPOT p99 ms | 84.9295 | 87.0523 | 79.3786 |
| token间隔ITL p99 ms | 89.8419 | 382.6647 | 367.6899 |
| 峰值分配 GiB | 52.7277 | 59.3931 | **58.4544** |
| 峰值保留 GiB | 53.1328 | 60.0566 | 59.1699 |

新DFlash2相对同内核的新target-only吞吐 **+6.114%**。
新旧DFlash2的观测差异：吞吐 **+8.412%**、累计verify **-12.903%**、
平均每轮verify约 **-19.32%**。新旧输出和接受分布不同，因此累计耗时差不是固定序列的纯算子消融。
微基准也包含CUDA事件区间内的发射等待，不能将其24倍区域延时变化直接外推为整模型加速。

### 接受率与执行工作量

| 指标 | 旧DFlash2 | 新DFlash2 |
|---|---:|---:|
| 接受／提议 | 11202 / 36229 | 10969 / 37821 |
| 接受率 | 30.91998% | **29.00241%** |
| 请求投机周期 | 5178 | 5411 |
| 每请求每周期平均接受数+1 | 3.16338 | 3.02717 |
| 引擎verify调用数 | 1345 | 1452 |
| 实际target块前向 | 1349 | 1462 |
| 实际请求token验证位置 | 41407 | 43232 |

新接受长度0–7直方图：`[1495,1312,878,582,404,240,161,339]`。
旧直方图：`[1319,1230,875,587,362,228,166,411]`。
接受率下降约1.918个百分点，导致更多验证轮次，抵消了部分单轮成本收益。
不能据此认定draft训练能力退化；数值路径改变后生成的内容也不同。

### 严格正确性与数值路径边界

- 新DFlash2与新target-only：**4/4请求、16384/16384 token完全一致**。
- 旧DFlash2与此前旧target-only：同样4/4、16384/16384完全一致。
- 新旧target输出不相同。以下为相同的开头token数（0-based首次分叉索引）：
  request000000=162、000001=144、000002=337、000003=270。
  对应从第163、145、338、271个输出token开始不同。
- 本次没有证明新旧实现逐位等价，也未做模型质量评测；
  通过的是新路径内部target-only/DFlash2严格greedy一致性。
- 新旧DFlash2紧凑更新峰值均170164224字节，逐token完整state快照均0。
- 所有模式各一次，不能宣称稳定加速。完整3轮配对、C8及TP2尚未完成。

## 启用与证据

在现有K=7、8位置、vectorized target、fused GDN配置基础上，添加：
```text
--short-prefill-attention
```
target-only和DFlash2必须同时使用此开关进行正确性／性能配对。
该开关默认关闭，原路径可回退。不要用新DFlash2对旧target-only声称逐token等价。

服务器证据目录含：
`before.tar.gz`、`before.patch`、`before_hashes.json`、
`deployed_hashes.json`、`edits.json`、`candidate.patch`、
`kernel_probe.py/json/log`、`attention_microbenchmark.py/json/log`、
`tests.log`、`tests_exit.json`、`driver.py`、`status.json`、
三份独立output audit以及`analyze.py`、`comparison.json`。
精确启动命令、模型snapshot路径与运行参数在status和各manifest中。

原始结果及对应logs：
- `c4_tasks4096_shortprefill_target_new_r0_20261012`
- `c4_tasks4096_shortprefill_dflash_new_r0_20261012`
- `c4_tasks4096_shortprefill_dflash_old_r0_20261012`

后续若继续降低verifier，应对当前新路径重新测量投影、LM head和CPU调度占比；
本轮没有更改这些部分，不能用旧路径的占比直接预测下一项收益。
