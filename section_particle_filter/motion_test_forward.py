import sys
sys.path.append('../scripts/')
from robot import *

import copy

world = World(40.0,0.1)

initial_pose = np.array([0, 0, 0]).T
robots = []
r = Robot(initial_pose, sensor=None, agent=Agent(0.1,0.0))

for i in range(100):
    copy_r = copy.copy(r)
    copy_r.distance_until_noise = copy_r.noise_pdf.rvs() # 最初に雑音が発生するタイミングを変える
    world.append(copy_r) # アニメーションの際に動くように登録
    robots.append(copy_r) # オブジェクト参照リストにロボットのオブジェクトを登録

world.draw()