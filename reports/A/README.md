# A 方向：视觉编码与 Token 优化

本目录保存视觉 token 预算、ToMe 系列方法和目标硬件验证报告。建议按以下顺序阅读：

1. `visual_token_cost_profile.md`：视觉 token 对端到端耗时的影响。
2. `tome_reproduction_plan.md`：ToMe 复现计划与边界。
3. `tome_complete_evaluation.md`：固定 ToMe 与 proportional attention 完整结果。
4. `pitome_evaluation.md`：PiToMe 匹配策略适配与 150 题评估。
5. `dtome_evaluation.md`：DToMe 动态阈值、分桶校准与 150 题评估。

当前结论是：PiToMe 和 DToMe 均已完成可开关适配，但没有优于固定 ToMe 的
准确率与性能折中；PPU 端到端结果也不支持继续将视觉 token 合并作为主优化路线。

`images/` 保存报告引用的可视化；CSV 是报告中的逐样本或逐配置数据，不是运行入口。
