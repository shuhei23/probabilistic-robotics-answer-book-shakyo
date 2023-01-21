import sys
sys.path.append('../scripts/')
from robot import *

import copy
import pandas as pd


world = World(40.0,0.1, debug=True)

initial_pose = np.array([0, 0, 0]).T
robots = []
r = Robot(initial_pose, sensor=None, agent=Agent(0.1,0.0))

for i in range(100):
    copy_r = copy.copy(r)
    copy_r.distance_until_noise = copy_r.noise_pdf.rvs() # 最初に雑音が発生するタイミングを変える
    world.append(copy_r) # アニメーションの際に動くように登録
    robots.append(copy_r) # オブジェクト参照リストにロボットのオブジェクトを登録

world.draw()

poses = pd.DataFrame([[math.sqrt(r.pose[0]**2+r.pose[1]**2),r.pose[2]] for r in robots], columns=['r','theta'])
poses.transpose() # 縦横を入れ替えて表示
print(poses)