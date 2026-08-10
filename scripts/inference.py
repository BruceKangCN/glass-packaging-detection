# %%
#
# 1. 导入所需的包并进行全局配置
#

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from ultralytics import YOLO

try:
    # 检测 IPython 环境以判断是否处于笔记本模式中
    get_ipython() # type: ignore
    # 若是，则使用 widget 渲染后端
    mpl.use("widget")
except NameError:
    # 若未处于笔记本环境中，则使用 QtAgg 渲染后端
    mpl.use("QtAgg")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# %%
#
# 1. 加载模型权重
#

model = YOLO(PROJECT_ROOT / "weights" / "best.pt")

# %%
#
# 2. 进行推理
#

results = model.predict(PROJECT_ROOT / "data" / "images" / "test" / "1.jpg")

# %%
#
# 3. 可视化
#

annotated_img = results[0].plot()[:, :, ::-1] # type: ignore

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(annotated_img)

plt.show()
