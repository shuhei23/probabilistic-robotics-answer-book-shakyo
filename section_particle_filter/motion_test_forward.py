import sys
sys.path.append('../scripts/')
from robot import *

import copy
import pandas as pd


world = World(40.0,0.1)

initial_pose = np.array([0, 0, 0]).T
robots = []

for i in range(10):
    r = Robot(initial_pose, sensor=None, agent=Agent(0.0,0.1))   
    world.append(r) # アニメーションの際に動くように登録
    robots.append(r) # オブジェクト参照リストにロボットのオブジェクトを登録

world.draw()

poses = pd.DataFrame([[math.sqrt(r.pose[0]**2+r.pose[1]**2),r.pose[2]] for r in robots], columns=['r','theta'])
poses.transpose() # 縦横を入れ替えて表示
print(poses)

r_var = poses["r"].var()
theta_var = poses["theta"].var()
r_mean = poses["r"].mean()
theta_mean = poses["theta"].mean()

print(f"r_var: {r_var}")
print(f"theta_var: {theta_var}")
print(f"r_mean: {r_mean}")
print(f"theta_mean: {theta_mean}")


# sigma_nn = math.sqrt(r_var/r_mean) # rが1m進むあたりの距離の分散
# sigma_on = math.sqrt(theta_var/r_mean) # rが1m進むあたりの角度の分散
# print(f"sigma_nn: {sigma_nn}")
# print(f"sigma_on: {sigma_on}")


sigma_oo = math.sqrt(theta_var/theta_mean) # rが1m進むあたりの角度の分散
print(f"sigma_oo: {sigma_oo}")