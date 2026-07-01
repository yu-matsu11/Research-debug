import trimesh, os

# モデルの読み込み
# model_name = "triple_triangle2.ply"
# model_name = "single_triangle2.ply" # ファイル名を指定
# model_name = "texbunny_lowpoly_300.ply" # ファイル名を指定
model_name = "texbunny.ply" # ファイル名を指定
# model_name = "stuck_triangle.ply" # ファイル名を指定
model = os.path.join("debug_ply", model_name)
mesh = trimesh.load(model)

# 3Dで画面に表示（マウスで回転・拡大縮小が可能）
mesh.show()