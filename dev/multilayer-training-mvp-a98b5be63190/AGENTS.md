# 项目协作规则

- 本仓库对应远程`/root/autodl-tmp/corrector`根目录；robot-lab是另一个独立仓库，不更换它的origin。
- 当前任务是版本管理/分析；未经用户安排不启动GPU实验、购买资源、删除数据或关闭实例。
- `night-code-1e80242789`、`action-code-9274415dd6`、`guardian-code-2263c95a33`及其它历史部署目录按冻结版本保留。改进在新目录/新输出进行，保留模型协议与旧代码可复算性。
- 用户练习以主动实现为目的，不自动填完TODO；生产入口与练习明确区分。
- 计划统一维护`studies/policy_generalization/work_log.md`，证据更新`research_report.md`及对应日期results。
- 不提交模型、原始数据、完整环境、访问令牌、SSH私钥或个人邮箱；提交描述用中文，作者用GitHub noreply。
- 现有best/latest可能包含系统盘符号链接；不删除或移动未经验证的活动权重，不破坏备份关系。
- 说明结果时区分开发验证与闭环、正式回合与强制检查、计算耗时与异步控制等待。未经验证的方法不得写成已提升。
