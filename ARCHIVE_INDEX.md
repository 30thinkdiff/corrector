# AutoDL冻结部署源码归档 · 2026-11-14

此分支保留4090服务器实际使用过的源文件快照，用于追溯执行版本。当前研究开发主线在 `feature/j36m-lvm-ogg-20261112`，robot-lab工作台属于另一个仓库。

- `dev/expanded-4fe058f5c20c`：普通LIBERO历史桥接与J补测。
- `dev/expanded-2e6296eff15f`：17方法Plus评测、有限复测和Replay采集。
- `dev/object-j36m-*`：早期普通LIBERO Pilot及预测保留策略版本；不混入正式结果。
- `dev/j36m-*`：J36M正式架构、训练与部署冻结快照。
- `dev/multilayer-training-mvp-*`：早期统一多层MVP。
- 其余独立Python文件：只读健康检查与安全结果导出。

所有收录文件按原字节复制并核对SHA，详见ARCHIVE_MANIFEST_20261114.json。未修改原部署目录，未启动/停止实验，未复制模型、原始数据、视频或环境。历史源码包含当时的计划与状态，实际结果以主线的最终报告为准。
