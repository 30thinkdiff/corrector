"""任务 A：读取固定版本任务表，亲手补全三个小步骤。无需 GPU。"""

from pathlib import Path
import runpy


project_root = Path(__file__).resolve().parents[2]
source_path = (
    project_root
    / "outputs/libero_plus_sources/4976dc3/libero_suite_task_map.py"
)

# 这是官方固定版本的纯数据文件；run_path 执行文件并返回其全局变量字典。
namespace = runpy.run_path(str(source_path))
task_map = namespace["libero_task_map"]
object_tasks = task_map["libero_object"]
print("libero_object 任务条目数：", len(object_tasks))

print(type(object_tasks), type(object_tasks[0]), object_tasks[0])

def parse_task(task_id):
    task_name = object_tasks[task_id]
    print(f"解析任务 {task_id}：{task_name}")
    base_task, rest = task_name.split("_view_", 1)
    print(f"基础任务：{base_task}，剩余部分：{rest}")
    view_fields, initstate = rest.split("_initstate_", 1)
    print(f"视角字段：{view_fields}，initstate 标记：{initstate}")
    view_fields = view_fields.split("_")
    print(f"视角字段列表：{view_fields}")

    return {
        "task_id": task_id,
        "task_name": task_name,
        "base_task": base_task,
        "view_fields": view_fields,
        "initstate": initstate,
    }

# TODO 1：用一个 for 循环打印 ID 0、653、654 及其完整名称。
# object_tasks 是列表，task_id 是列表索引，不是基础操作的编号。
for id in [0, 653, 654]:
    task = object_tasks[id]
    # print(f"ID {id}：{task}")

parse_task(653)
# TODO 2：选择 653、654，分别拆出基础名称、五个视角字段、initstate 标记。
# 每条记录用一个普通 dict 表示即可，不需要类、异常框架或通用解析器。
# 目标字段：task_id、task_name、base_task、view_fields、initstate。
# 提示：按完整分隔符拆，不要一开始就按所有下划线拆散整个操作名称。



# TODO 3：比较两条记录，打印：基础名称是否相同、initstate 是否相同，
# 以及五个视角字段中，具体是哪些位置发生了变化。
# 注意：字段相同不能代替之后的仿真运行时核对。
