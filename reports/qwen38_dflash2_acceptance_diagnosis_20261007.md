# Qwen3.8 + DFlash2：4K 低接受率差异定位

日期：2026-10-07。范围：同一 A800、K=7、四个自然代码／推理任务，各输出 4096 tokens。
本次只增加诊断工具，未修改生产模型、缓存或 verifier 数学路径。

## 基线与方法

- Mini 基线：checkpoint `b2081ac`；远端 HEAD `70239ac` 加已有未提交工作。
  生产源码树 SHA256 为 `0cd6529c4b2ed4d4fab87e76f1280541c08dca94e10118b1e3c2a1f67e8900a2`。
- target snapshot：`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；
  draft snapshot：`dedf8df68adfb1afeaf7b7480c0a0243108177b4`。
- 官方对照：服务器既有 Oracle 环境中安装的 vLLM 0.28.0，V2 runner，DFlash2Speculator，
  BF16、greedy、TP1、eager、language_model_only、关闭 prefix caching 和 async scheduling。
- 两次完整 Mini 影子诊断都返回原始 proposal；所有替代结果只作旁路计算。
  第二次另外记录少量缓存和特征，关闭初始化预热以缩短诊断等待，因此两次时间都不作性能结论。
- 独立读取原始 request JSONL，两次各 4 请求 × 4096 输出均与已审计 target-only 完全一致；
  合计 8 请求、32,768 输出 tokens。两次接受数也完全一致。
- 官方固定前缀实验只生成 16 tokens，用于提取同一位置的内部张量；不是短输出性能实验。

## 1. 确认差异：候选路径打分被额外舍入到 BF16

官方 `model_executor/layers/logits_processor.py:get_top_k_tokens` 在取得 top-k 后将
values 转为 FP32，再与 transition score 相加。Mini 的
`src/myvllm/spec_decode/dflash2.py:score_candidate_edges` 使用 BF16 unary scores，
并把 transition 转回 unary dtype，相加后仍是 BF16。最终 greedy walk 的平分规则一致，
但 Mini 的提前舍入会制造原本不存在的平分。

首次分叉：cycle 1、`tasks4096-000002`、输出位置 5，proposal 的第二个 token：

| 项目 | token 25 | token 13 |
|---|---:|---:|
| Mini BF16 总分 | 25.25 | 25.25 |
| 官方 FP32 总分 | 25.1875 | 25.25 |

Mini 选 25，官方规则选 13。该例原路径接受 2 个，FP32 替代路径反而只接受 1 个。

完整固定周期影子对照：

| 指标 | 结果 |
|---|---:|
| 请求投机周期 | 5,178 |
| proposal 发生变化 | 386（约 7.45%） |
| 原路径接受数 | 11,202 / 36,229 = 30.92% |
| FP32 影子接受数 | 11,160 / 36,229 = 30.80% |
| 接受长度提高／降低的周期 | 37 / 55 |

**这是已复现的官方语义对齐问题，但不是已证实的低接受率主因。**
影子结果沿用原周期边界，不能当作修改 selector 后实际生成的接受率或性能。
后续应单独修复精度对齐、重跑正确性；不能宣称该修改会提高接受率。

## 2. 缓存差异定位：投影分段的 BF16 数值形状

对 cycle 0、1、2、16、128、512、1024 的 4 请求共 28 个样本检查：
context positions 连续且对齐；1/28 的缓存 B1 与整窗重算 B1 提议不同；
4/28 的原生 B4 与缓存 B1 提议不同。不能将抽样结果当作全部运行的缺陷率。

进一步完整保存 cycle 1024、request 0 的状态：anchor position 3300，
滑窗 `[1252,3300)`，2048 个 token。按原来的 706 个投影分段重放：
**五层所有 K/V 逐位完全一致，最大误差为 0**。
将相同 hidden taps 一次性投影则 K/V 相对 L2 误差约 0.00335–0.00395，最大绝对误差 0.25。

| 路径 | 七个 proposal token IDs |
|---|---|
| 缓存 B1／原分段重放 B1 | 348, 348, 318, 15, 13, 15, 578 |
| 实际 Mini B4 | 348, 220, 15, 13, 15, 578, 198 |
| 整窗重算 B1 | 348, 220, 15, 13, 15, 198, 1734 |
| 官方对同一长前缀重新 prefill | 348, 220, 15, 13, 15, 198, 1734 |

该样本的缓存重放差异可归因于 BF16 投影 GEMM 形状，未发现丢 token 或位置错位。
官方实验是重新 prefill，不代表官方历史增量缓存必然与整窗相同。
该位置 Mini target 在 anchor 后的下一 token 为 21510，以上路径首个候选均为 348，
所以即使消除该样本的后缀差异，也不会增加其接受长度。

## 3. 固定特征，区分 target 与 draft 核心

分别对原任务初始 prefix 187 和长 prefix 3300，固定前缀、anchor、position：

- 把官方 target hidden taps 喂给 Mini draft：两例均与官方首次 proposal **7/7 一致**。
- 换回 Mini target hidden taps 并整窗投影：两例也都是 **7/7 一致**。
- 两例 Mini/官方 anchor 都一致（1596 和 60631）。
- target taps 有数值差异，初始 prefix 各采样层相对 L2 为 0.40%–2.18%，
  长 prefix 为 0.34%–4.82%；长 prefix 最深层 cosine 0.998837。
- 即使固定官方特征，draft 最终 hidden states 仍存在约 2.2% 相对 L2 差异，
  但这两例没有造成最终 proposal 分叉。

因此这两例没有支持“target 特征取错层／draft 核心算法整体错误导致接受率崩塌”。
数值差异真实存在；两例结果不能证明所有前缀与官方等价。
检查 checkpoint 的非因果 draft attention 配置也与 Mini 一致，未发现遗漏 causal mask 的证据。

## 4. 官方同任务完整 4K 对照：接受率同样约 30%

在官方环境独立完成同一组 4 个 prompt token 序列，各固定输出 4096，K=7、C4。
关闭 prefix caching，greedy，ignore_eos；无预热请求，记录生成前计数全部为零。

| 指标 | Mini | 官方 vLLM DFlash2 |
|---|---:|---:|
| 输出 tokens | 16,384 | 16,384 |
| draft tokens | 36,229 | 36,750 |
| accepted draft tokens | 11,202 | 11,140 |
| 接受率 | 30.9200% | 30.3129% |
| 请求投机周期 | 5,178 | 5,250 |
| 每周期接受数 | 2.1634 | 2.1219 |
| 接受数 + 1 | 3.1634 | 3.1219 |

官方各位置累计接受数 `[3840,2643,1753,1179,787,563,375]`，
第一位置接受概率约 73.14%，七个全收约 7.14%。Mini 对应约 74.53% 和 7.94%。
“30%”是七个提议位置合计的前缀接受率，**不是第一个 token 只有 30% 命中**。
后面即使猜中，一旦前面先拒绝也不能计入本轮接受数。

这是一次官方接受水平对照，不是严格配对的因果消融：两个引擎数值路径不同，
四个输出的公共前缀长度分别只有 173、144、135、125 tokens，后续内容不同。
官方尾部仍按完整 K 记 proposal；两个实现的结束截断统计存在微小口径差别。
因此不能把 0.61 个百分点差值解释为 Mini 更优，也不能据此证明所有实现细节正确。
但这次结果**不支持“官方在该负载上接受率很高，而 Mini 实现使其整体降到约 30%”**。
目前证据更符合该 checkpoint 在这组输入、greedy 路径上的连续预测能力有限。

官方本次生成墙钟 118.157 秒，含冷态 JIT，峰值分配／保留显存 66.629／66.697 GiB。
没有配对官方 target-only，也没有三轮性能重复；这些时间只作诊断记录，
不报告官方投机加速倍数，不与 Mini 既有性能配对混合统计。

## 5. 性能问题与下一步

既有关闭细分 profiling 的 Mini 控制轮：target decode 375.026 秒；
DFlash verify 352.442 秒、proposal 27.463 秒、commit 46.384 秒。
验证仅省 22.584 秒，proposal + commit 已花 73.847 秒。这能直接解释为什么
每周期输出超过 1 token 仍然更慢：每个投机周期的成本也远超过普通 decode 一步。

任务优先级据本次证据调整为：

1. 单独对齐 selector FP32 语义并做严格输出回归；承认其不承诺提高接受率。
2. 继续削减实际 verifier 与 commit 成本，优先细分 full-attention、前缀恢复、
   上下文投影的 GPU/CPU 时间。保持其余成本，verify 需从 352.442 降至约 301.157 秒
   才能在该控制轮解码阶段持平（约减少 14.6%）；这只是成本预算估算。
3. 如要提高接受率，应在固定 target 前缀上扩展官方/本地 proposal 比较，
   或验证更适配该 target 的 draft checkpoint／K 策略。不能将更换输入造成的接受率
   提高归为实现优化；不能靠放宽 greedy 一致性换收益。

本次已定位两类可复现的数值差异，并完成影响量化；没有证据把它们认定为
30.92% 接受率的主要根因。不存在已验证的“一处改动即可恢复高接受率”的结论。

## 证据与复现工具

服务器根目录：`/root/autodl-tmp/Mini-vLLM`。原始文件全部保留。

- `results/qwen38_dflash2_perf/selector_alignment_20261007/`：第一次全量影子对照。
- `results/qwen38_dflash2_perf/selector_capture_20261007/`：重复对照、首次分叉张量、
  缓存检查、5 个保存状态、706 分段重放报告、独立输出审计。
- `results/qwen38_dflash2_perf/oracle_natural_prefix_20261007/`：短前缀官方运行与固定特征重放。
- `results/qwen38_dflash2_perf/oracle_long_prefix_20261007/`：长前缀 token 输入、官方运行、
  完整内部张量与固定特征重放。内部张量约 2.3 GiB，保留于服务器。
- 对应 `c4_tasks4096_selector_shadow_20261007`、`c4_tasks4096_selector_capture_20261007`
  的 results 和 logs 目录保存模型配置、源码归档、diff、原始请求 JSONL 与显存统计。
- `results/qwen38_dflash2_perf/oracle_tasks4096_diagnostic_20261007/`：官方完整 4K
  输入配置、原始输出、计数器、控制台日志、源码副本及哈希、接受率对照与证据清单。
- 新工具：`diagnose_dflash2_selector_precision.py`、`replay_dflash2_context_chunks.py`、
  `compare_dflash2_fixed_features.py`；已有 exporter 增加显式 token prompt 选项，
  避免长前缀经过 decode/reencode 后发生 token 变化。

验证：服务器 Oracle fixture 相关测试 **18 passed**；三个新脚本通过编译检查，
上述 GPU 重放与两次完整输出审计通过。生产路径未改，未重复全性能矩阵。
