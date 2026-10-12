# Qwen3.8 DFlash2：K=7 单次8位置验证

日期：2026-10-12。实现 checkpoint：`b01170c`，基线 `fe42517`。
用户要求先将 anchor + 7 个 draft token 在一个完整块中验证。

## 改动范围

- 并行 verifier 的缺省块长设为8；benchmark 缺省块长也设为8。
- 正式运行显式传入 `--parallel-target-verification --parallel-verification-chunk-size 8`。
- `--parallel-verification-chunk-size 4` 保留为对照；显式0仍保留原来的整块路径。
- 沿用 vectorized target、融合GDN、融合前缀恢复、采样复用和紧凑更新。
- 未修改模型权重、attention 数学、selector 精度或采样策略。
- 完整K=7且同长度的C4 micro-batch执行一次8位置target前向。
  结束窗口不足8时只验证实际长度；不同剩余长度按长度分组，不能把这种分组误报为每个请求仍分两块。
- 本次未优化 attention 内部逐位置 QK/softmax/PV；单次模型块前向不代表所有内部算子均已融合。

## 回归与保留

服务器全量 `python -m pytest -q tests`：**263 passed、2 skipped，42.20秒**。
接受长度0–7新增4/8两种块长对照；8位置模式检查一次提交完整查询，
并核对部分提交状态与逐步target状态逐位一致。EOS、验证异常和提交异常回滚走8位置路径。
紧凑更新底层storage计数检查和无逐token完整state快照检查保留。

本地/远端修改前五个关键文件SHA256一致，远端部署前再次检查三个被修改文件的哈希。
远端HEAD仍为70239ac，原有未提交文件全部保留；修改前src/scripts/tests/reports已归档。
证据目录：`/root/autodl-tmp/Mini-vLLM/results/qwen38_dflash2_perf/verify8_20261012/`。

## 实验约定

同一A800-SXM4-80GB，C4、K=7、micro-batch4，四个自然代码／推理任务，
输入187/194/190/201 tokens，各输出4096（含推理token），greedy、ignore_eos、seed20260907。
先8位置再4位置，各一次；加载不计入正式计时，均跳过初始化最大窗口预热，
各做4请求×64输出在线预热，正式输出始终4096。关闭profiler和细分DFlash profiling，保留memory_trace。

同一源码仅验证块长不同。每轮保存manifest、源码归档、diff、模型路径、JSONL、fairness、
显存和分阶段耗时；独立读取request JSONL，比较既有已审计target-only的完整输出，
并检查接受数与分布。任一正确性门槛失败即停止后续实验。

除4/8对照外，补充同源码、同预热设置的target-only单轮基线。所有模式均只有一次观测。

## 完成结果

三种模式按8位置、4位置、target-only顺序各运行一次，均无失败、取消或schema错误。
三个manifest源码树SHA256相同：
`f99244b25d310f9733bcfb51b864e22a82229ef617ff6a0601739d0197d32f2c`。
三份workload文件逐字节一致；参数差异仅run_id、块长和是否启用DFlash2。

| 指标 | target-only | DFlash2 4位置 | DFlash2 8位置 |
|---|---:|---:|---:|
| 吞吐 tok/s | 37.4535 | 34.9819 | **39.1779** |
| 端到端墙钟 秒 | 437.4490 | 468.3570 | 418.1946 |
| 普通decode / verifier 秒 | 377.3111 | 338.9574 | 284.0944 |
| proposal 秒 | — | 27.8366 | 28.5628 |
| commit 秒 | — | 42.9819 | 46.0723 |
| 平均TTFT 秒 | 59.1147 | 58.2135 | 59.0978 |
| 平均TPOT ms | 92.3893 | 97.6728 | 84.8845 |
| 峰值分配 GiB | 53.1385 | 59.4762 | 59.3989 |
| 峰值保留 GiB | 53.5273 | 60.2051 | 60.0566 |
| verifier调用数 | — | 1345 | 1345 |
| 实际target块前向数 | — | 2183 | 1349 |
| 实际请求token验证位置 | — | 25372 | 41407 |
| 跳过的请求token位置 | — | 16035 | 0 |

8位置对4位置：吞吐 **+11.995%**，verifier墙钟 **-16.186%**，实际块前向数 **-38.204%**。
commit墙钟增加7.19%，并未随前向次数一起下降。
8位置对本次新target-only：吞吐 **1.04604×，+4.604%**。
这是单次同条件比较，不能宣称已经获得稳定加速或大幅加速。

两种DFlash2均接受11202/36229 = **30.919981%**；
接受长度0–7直方图均为 `[1319,1230,875,587,362,228,166,411]`。
8位置的1345次verifier调用有1349次模型块前向，多出的4次是尾部剩余长度分组，
没有将同一完整K=7请求拆成两次4位置前向。

## 正确性与显存结论

- 4/8两轮的原始request JSONL分别与既有target-only序列完全一致。
- 新target-only与8位置再次独立比较：4/4请求、16384/16384 tokens严格一致。
- 三组输出合计12请求、49152 tokens来自同一组四个任务的重复执行，不表示12个独立任务。
- 两种DFlash2紧凑更新记录峰值均170164224字节，逐token完整recurrent-state快照均0；
  临时内存由CUDA峰值统计覆盖。
- 接受率和输出均未改变，性能变化来自验证执行方式。
- 8位置verifier仍为普通decode累计耗时的约75.3%，尚未接近理想的约三分之一。
  一次模型块前向内部仍有逐位置attention、LM head及缓存整理；本次不声称已优化这些算子。

## 文件与复现

服务器证据目录内：
`tests.log`、`before.tar.gz`、`before.patch`、`after.patch`、
`change.json`、`local_checkpoint.json`、`driver.py`、`status.json`、
`target_driver.py`、`target_status.json`、`width8_audit.json`、`width4_audit.json`、
`fresh_target_audit.json`、`comparison.json`。
精确启动命令保存在两个driver状态文件中。

原始结果及对应logs：
- `c4_tasks4096_verify8_r0_20261012`
- `c4_tasks4096_verify4_r0_20261012`
- `c4_tasks4096_verify8_target_r0_20261012`

之后继续使用8位置时显式保留：
```text
--num-speculative-tokens 7 --parallel-target-verification
--parallel-verification-chunk-size 8
--vectorized-target-numerics --fused-gdn-target
--cached-target-micro-batch-size 4
--fused-prefix-commit --reuse-verification-samples --compact-verification-copies
```

本次任务已完成：服务器代码保留8位置默认块长，GPU实验全部结束。
稳定收益验收仍需至少三轮target-only／8位置配对；C8、TP2与C16/C32不在本次完成范围。
