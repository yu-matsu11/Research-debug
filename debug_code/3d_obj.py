import trimesh, os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path

# 1. データの読み込み
script_path = os.path.abspath(__file__)
debug_dir = os.path.dirname(script_path) #c:\Users\YutoMatsuo\Desktop\Research\debug\debug_code
debug_path = os.path.dirname(debug_dir) #c:\Users\YutoMatsuo\Desktop\Research\debug

script_dir = Path(__file__).resolve().parent.parent
model_name = "single_triangle7"
model = script_dir/"debug_ply"/f"{model_name}.ply"

data = trimesh.load(model)

mesh = trimesh.load(model)

# 頂点データを取り出す (単位変換が必要な場合は Global_vertices 等に置き換えてください)
vertices = mesh.vertices

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 2. 三角形の描画
# 頂点をつないで面を作る（ポリゴン法で扱っている三角形を表示）
for face in mesh.faces:
    v = vertices[face]
    # 面を閉じるために最初の頂点を最後に追加
    v = np.vstack((v, v[0]))
    ax.plot(v[:, 0], v[:, 1], v[:, 2], '-o')

# 3. 軸の設定
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_title(f'3D View: {model_name}')

# --- ここで視点を固定できます ---
# ax.view_init(elev=垂直方向の角度, azim=水平方向の角度)

# 例1：正面 (XY平面を見たい場合、Z軸方向から見る)
ax.view_init(elev=90, azim=-90)

# 例2：真上 (XZ平面を見たい場合)
# ax.view_init(elev=0, azim=-90)

# 例3：真横 (YZ平面を見たい場合)
# ax.view_init(elev=0, azim=0)

# マウスで自由に動かしたい場合は設定せずに show() する
plt.show()