# %%
#
# 1. 导入所需的包并进行全局配置
#

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from ultralytics import YOLO

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

results = model.predict(PROJECT_ROOT / "data" / "images" / "val" / "IMG_20260805_153829.jpg")

# %%
#
# 3. 可视化
#

annotated_img = results[0].plot()[:, :, ::-1] # type: ignore

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(annotated_img)

plt.show()
