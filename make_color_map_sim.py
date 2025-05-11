import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import matplotlib.animation as animation

# 設定
num_particles = 10  # 粒子数
space_size = 1  # 空間サイズ
d_min = 0.8  # 最小距離
k = 0.01  # バネの強さ
gamma = 0.5  # 減衰定数（摩擦係数）
dt = 0.001  # 時間ステップ
velocity_threshold = 0.000001  # 収束条件：最大速度がこの値以下なら終了
iterations = 10000  # シミュレーション回数

avoid_points = [(0, 0, 0), (1, 1, 1)]  # 避けるべき位置

# 初期配置（ランダム）
np.random.seed(42)
positions = np.random.rand(num_particles, 3) * space_size
velocities = np.zeros((num_particles, 3))  # 初期速度はゼロ

# プロット設定
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(positions[:, 0], positions[:, 1],
                positions[:, 2], color="red", s=30, depthshade=False)

ax.set_xlim(0, space_size)
ax.set_ylim(0, space_size)
ax.set_zlim(0, space_size)

# max_velocity を表示するためのテキストボックス
velocity_text = ax.text2D(
    0.05, 0.95, f"Max Velocity: 0.0", transform=ax.transAxes)

# 目標位置を避ける反発力の計算


def repulsive_force(position, target, strength=0.1, threshold=0.1):
    distance = np.linalg.norm(position - target)

    direction = (position - target) / distance  # 反発方向
    force = strength * (1 / distance**2) * direction  # 反発力（距離の2乗に反比例）
    return force


def update(frame):
    global positions, velocities
    for _ in range(10):
        forces = np.zeros((num_particles, 3))

        # 粒子間の力を計算
        for i in range(num_particles):
            for j in range(num_particles):
                if j == i:
                    continue
                direction = positions[i] - positions[j]
                distance = np.linalg.norm(direction)
                if distance > 0:
                    force_magnitude = -0.125 * \
                        (distance) * np.exp(0.5*distance)
                    force_coulomb = 0.064 * (1/distance**2)
                    force = 0.5*(force_magnitude + force_coulomb) * \
                        (direction / distance)
                    forces[i] += force
                    forces[j] -= force  # 反作用

        # 減衰力を計算（速度に基づいて）
        damping_forces = -0.1 * velocities  # 速度に比例した減衰力
        forces += damping_forces  # 減衰力を追加

        # 目標位置 (0, 0, 0) と (1, 1, 1) からの反発力を追加
        for i in range(num_particles):
            for target in avoid_points:
                # 反発力を加える
                forces[i] += repulsive_force(positions[i], np.array(target))

        # for i in range(num_particles):
        #     for j in range(3):
        #         force = np.random.rand(3)
        #         forces[i] += 0.0001*force/np.linalg.norm(force)

        # 速度と位置の更新
        velocities += forces * dt
        positions += velocities * dt

        # 空間の範囲内に収める
        for i in range(num_particles):
            for j in range(3):  # X, Y, Z の各軸についてチェック
                if positions[i, j] < 0:  # 位置が最小値を下回った場合
                    velocities[i, j] = -1.0 * \
                        velocities[i, j]  # 対応する軸の速度をゼロにリセット
                elif positions[i, j] > space_size:  # 位置が最大値を上回った場合
                    velocities[i, j] = -1.0 * \
                        velocities[i, j]  # 対応する軸の速度をゼロにリセット
        positions = np.clip(positions, 0, space_size)

        # 最大速度のチェック
        max_velocity = np.max(np.linalg.norm(velocities, axis=1))
        if max_velocity < velocity_threshold:
            ani.event_source.stop()  # 収束したらアニメーション停止
            print("収束しました")
            return

    # 描画の更新
    velocity_text.set_text(f"Max Velocity: {max_velocity:.10f}")
    sc._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    sc.set_color(positions)


# アニメーションの設定
ani = animation.FuncAnimation(fig, update, frames=200, interval=100)
plt.show()
