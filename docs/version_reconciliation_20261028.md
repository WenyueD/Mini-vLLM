# 版本统一记录 / Version reconciliation

## 审计起点

- GitHub `main`：`c911bc4415cd3a63f673860b887e3ae8e8c881a7`，2026-09-12；查询时远端仅此分支。
- 本地 `main`：`1b3235d`，领先25个提交，另有未提交的合并头、测试和诊断数据。
- 服务器主目录：`codex/qwen38-dflash2-k7-memory`，HEAD `70239ac`，叠加未提交修改。
- 10月28日诊断使用独立的本地源码快照candidate，不是服务器旧主目录。

## 取舍与来源

已下载服务器src、scripts、tests、reports和元数据，忽略CRLF/LF后逐文件核对。不同的实现/测试/基准文件和三份报告中，本地包含较新的KV事务复用、验证后处理、合并头及相应测试和报告，保留本地版本。

服务器独有的 `profile_qwen35_commit_cost.py` 与本地 `profile_qwen35_verifier.py` 内容完全相同，保留文件名以兼容原命令。egg-info、notebook checkpoints、pytest临时目录和传输bundle不作为实现来源。

本次归档10月13日、28日的选定证据和诊断脚本；合并头仍默认关闭。仅整理runner一处参数缩进，没有改变默认数值路径。中英文README更新为Qwen/DFlash2入口，旧介绍移至 `legacy_minivllm_readme*.md`，许可证保留。

## 备份与版本规则

服务器旧工作树保存到 `backup/pre-unify-20261028`，统一版本使用main；旧结果、日志和环境保留。本地审计快照在被忽略的 `results/version_audit_20261028/`，服务器另有 `/tmp/minivllm-before-unify-20261028.patch` 和审计归档，长期恢复以备份分支为准。

后续共享基线为GitHub main。实验记录应同时包含 `git rev-parse HEAD` 和 `git status --short`；有未提交源码时不能仅用提交哈希标识。历史candidate目录是实验快照，不是持续维护的主目录。

## 检查结果

- 本地项目虚拟环境完整回归：266 passed，60 skipped（30.70秒）。
- 服务器统一候选目录完整回归：266 passed，60 skipped（54.62秒）；通过模块路径确认加载的是候选源码。
- 服务器首次测试因SSH断开未完成，后台重试结果如上；重试时PyTorch为2.9.1+cu128，但CUDA不可用、device_count=0。本次没有新增GPU回归或真实权重吞吐验证，不能将跳过项计为通过。
- 固定输入归档校验通过：默认头与合并头实际B4、候选/前缀一致、接受数均为3/3/3/7，纠正token匹配。
- 中英文README及报告索引本地链接有效；文档中的benchmark参数均存在于CLI帮助；Git whitespace检查通过。
- pytest限定 `testpaths=["tests"]`，避免收集results审计备份里的旧测试。
- 服务器备份分支提交：`add35c7`。未提交的个人 `req.txt` 不属于本次发布，保持原样。
