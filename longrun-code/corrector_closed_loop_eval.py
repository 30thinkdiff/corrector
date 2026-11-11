"""单回合/批量Corrector入口；共享原基线的环境、初始化、录像与逐回合恢复逻辑。

--methods H50 H20 monitor lvm ogg；--initial-state-ids 0为单初始化，多个编号/seed为批量。
控制练习留在corrector_online_runtime.py的student_*；显式reference可先测试完整连接。
"""
from smolvla_baseline_benchmark import main

if __name__=='__main__':main()
