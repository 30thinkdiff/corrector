# SmolVLA相机扰动、执行窗口与Corrector

最新状态（2026-11-11）：15组训练和五组正式闭环结果已只读取回，1200/1200正式回合、程序错误0；所有best/索引/计数核验通过。单动作MLP/T-H1/T-H3完整OGG为55/60、55/60、56/60；A10-H1/H3为50/60、51/60。逐步trace证实A10预热后评分覆盖仅59.02%/58.87%，最长空窗0.95秒。OGG与LVM在本批各自60场景上成败逐一相同。下一步固定权重做实际执行动作条件补全与覆盖率对照；尚未实现新算法收益。详见[研究报告](research_report.md)、[工作日志](work_log.md)、[本轮完整审计](results/20261111/RESULTS.md)。以下按日期保留历史进度。


当前进展与行动安排统一见[工作日志](work_log.md)，按任务和日期维护；结论与证据见[研究报告](research_report.md)。

2026-11-09（最终300回合已完成）：Object10全部98个原始分片、454条/66984帧特征已完成；共享38M Corrector训练50轮/4900次更新，第45轮best，逐轮计算合计49.24分钟。开发验证cosine0.80436、MSE3.99592（零预测11.58570），best重载通过；数据364训练/90验证、无episode交叉。补图与10回合程序检查已通过，正式十任务300回合配对已完成，错误0：H50 44/60、H20 51/60、完整Corrector 55/60；监测为同步实现，声明时间跨度已换算为20控制步，采集物理时钟来源仍待核实。用户回传证据见[预测器摘要](results/20261109/object10_predictor_summary.json)及[最终闭环结果](results/20261109/object10_paired_final.json)。旧单任务2394回合结果分开保留，不冒充新模型多任务成绩。
2026-11-09晚：已完成可切换的上游Transformer生产接入和12小时可恢复队列，MLP H1/Transformer H1/H3各3个种子；复用454条缓存、共同时间锚点、预测器BF16，seed42各做有限闭环。CPU回归129项通过，后续用户已回传H1 s42真实GPU训练15轮，验证cosine约0.77704；原条目为部署时状态，新动作序列版尚未部署。时间审计回传454条特征/300回合错误0，跨度20控制步；细节、部署与恢复命令统一见[工作日志](work_log.md)。


- 运行入口：[数据与训练Notebook](../../scripts/corrector_data_prepare.ipynb)、[CPU/GPU命令行](../../scripts/corrector_train.py)。
- 技术参考：[上游源码导读](corrector_source_walkthrough.md)、[算法核心练习与审查](corrector/practice.md)。
- AutoDL入口：[并发预下载](../../scripts/corrector_prefetch.py)、[分片与特征流水线](../../scripts/corrector_stream_workbench.py)、[闭环评测](../../scripts/corrector_closed_loop_eval.py)。公开固定清单见[Object十任务manifest](corrector/manifests/libero_object10_8695891.json)，可用于重新打包；不包含图像、权重或账号信息。
- 流水线新增`run --watch-local`：只读并核验已完整下载的本地分片，自动等待新数据、复用编码器，不删除下载器文件；可与prefetch并行。默认run仍按串行使用。新模式已通过本地CPU检查及远端真实GPU多轨迹提取，现已完成全部454条轨迹特征。
- 既有结果：[执行窗口实验](replanning_results.md)、[T4 LoRA实测](training_results_20261105.md)、[离线评分](offline_eval_results_20261105.md)。
- 公开证据：[2026-10-31结果](results/20261031)、[2026-11-07数据与基线摘要](results/20261107)。

旧行动指南已合并，原路径仅跳转。历史文件中的“尚未提取/尚未运行”须结合日期阅读，以工作日志顶部状态为准。

2026-11-09后续：新增可切换的当前帧/历史帧＋10动作条件预测器及多动作OGG。用户选择先完成原9组；独立20h调度器等待并用原冻结版本续完，再启动新6组及有限闭环。140项CPU回归通过；新序列版无GPU成绩。详见[工作日志](work_log.md)。

2026-11-09 23:18：既有后续调度器正常等待原H1训练；新增独立36h监督包（待上传），延续15组工作并在提前完成时追加初始化配对评测，带双盘无损续训状态归档/资源监测。详见工作日志；尚无36h新增实验结果。
