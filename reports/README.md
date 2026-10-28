# 实验报告索引 / Experiment index

按日期和配置阅读。请求并发不等于实际验证batch；GPU活动累计、解码耗时和端到端吞吐是不同指标。

| 日期 | 内容 | 证据与范围 |
|---|---|---|
| 2026-10-28 | 官方与Mini的B4固定输入诊断 | [报告](matched_profile_20261028/README.md)、[校验](matched_profile_20261028/validation.json)；K7短前缀，不能外推C24 |
| 2026-10-13 | C1默认头配对 | [数据](batch1_default_comparison_20261013.json)；4×3000输出一致，单次观测 |
| 2026-10-13 | C1合并头失败 | [数据](batch1_merged_comparison_20261013.json)；BF16分歧，默认关闭依据 |
| 2026-10-13 | C24合并头A/B | [数据](merged_lm_head_real_comparison_20261013.json)；两边均为DFlash2，24×3000 |
| 2026-10-13 | commit与LM head探查 | [commit](commit_diagnosis_20261013.json)、[LM head](merged_lm_head_probe_20261013.json) |
| 2026-10-12 | direct-KV与块验证 | [short prefill](qwen38_dflash2_short_prefill_20261012.md)、[verify8](qwen38_dflash2_verify8_20261012.md) |
| 2026-10-07 | 接受率、长输出与成本 | [接受率](qwen38_dflash2_acceptance_diagnosis_20261007.md)、[4096输出](qwen38_dflash2_decode4096_20261007.md)、[成本](qwen38_dflash2_cost_optimization_20261007.md) |
| 2026-09-30 | GDN融合与短请求 | [报告](qwen38_dflash2_gdn_profile_20260930.md)；不能代表长输出 |
| 2026-09 | K7/TP2与早期矩阵 | [阶段记录](qwen38_dflash2_k7_tp2.md)、[并发矩阵](qwen38_dflash2_high_concurrency_2026-09-07.md)；历史实现 |

历史报告中的BLOCKED和性能结论只对应当时配置。原始trace、大体积结果留在实验服务器，Git保留摘要和选定证据。
