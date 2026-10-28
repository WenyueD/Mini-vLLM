# 固定输入诊断复现

这些脚本针对2026-10-28的B4/K7诊断，不是通用吞吐benchmark。

准备独立目录：

```text
dflash-matched/
  inputs.json
  diagnose_official_matched_profile.py
  diagnose_mini_matched_profile.py
  analyze_matched_traces.py
  candidate/              # 本仓库完整源码副本
    src/
    scripts/
```

从 `reports/matched_profile_20261028/inputs.json` 复制四个prompt和配置，只修改target/draft为实际模型目录；从scripts复制三个脚本到上述目录。

1. 在独立官方环境运行 `diagnose_official_matched_profile.py`。历史对照为vLLM 0.28.0 / g2cf0a6915、V2、eager；内部API不保证兼容其他版本。输出official_evidence.json和trace。
2. 在Mini的Python 3.11环境运行 `diagnose_mini_matched_profile.py`。它从candidate/src加载代码，将正式首轮候选替换为官方输入，断言prompt、anchor、batch和query相同。
3. 用 `DIAG_MERGED_HEAD=1` 再运行Mini脚本，证据输出到merged/。
4. 运行 `python analyze_matched_traces.py /path/to/dflash-matched`，再对其merged子目录运行一次。
5. 从仓库运行 `python scripts/validate_matched_evidence.py /path/to/dflash-matched`，校验输入、接受长度和纠正token，生成validation.json。

两套环境依次运行，避免GPU竞争。同步、Python标记和profiler会影响时延，不能当作无插桩吞吐。官方采样与下一次draft共用外层阶段，commit边界也不同，不可直接按阶段名比较。后续生成只用于结束请求，不构成全序列官方一致性验证。模型权重和大体积trace不随Git分发。
