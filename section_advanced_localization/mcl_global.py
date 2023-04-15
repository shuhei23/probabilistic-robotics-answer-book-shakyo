import sys
sys.path.append('../scripts/')
from mcl import *

class GlobalMcl(Mcl): # 大域的自己位置推定MCL
    def __init__(self, envmap, num, motion_noise_stds={"nn":0.19,"no":0.001,"on":0.13, "oo":0.2}, \
                 distance_dev_rate=0.14, direction_dev=0.05):
        super().__init__(envmap, np.array([0,0,0]).T, num, motion_noise_stds, distance_dev_rate, direction_dev) # 初期姿勢は適当
        for p in self.particles: # ランダムに姿勢を初期化
            p.pose = np.array([np.random.uniform(-5.0,5.0),np.random.uniform(-5.0,5.0), np.random.uniform(-math.pi,math.pi)]).T

def trial(animation):
    time_interval = 0.1
    world = World(30, time_interval, debug=not animation) # アニメーションのON/OFFをdebugで制御

    ## 地図を生成して3つのランドマークを追加
    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    ## ロボットを作る

    init_pose = np.array([np.random.uniform(-5.0,5.0), np.random.uniform(-5.0,5.0), np.random.uniform(-math.pi,math.pi)]).T
    pf = GlobalMcl(m,100)
    a = EstimationAgent(time_interval, 0.2, 10/180.0*math.pi, pf)
    r = Robot(init_pose,sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()

    return (r.pose, pf.pose) # 真の姿勢と推定姿勢を返す


if __name__=='__main__':
    ok = 0
    for i in range(1000):
        actual, estm = trial(False)
        diff = math.sqrt((actual[0] - estm[1])**2 + (actual[1] - estm[1])**2 )
        print(i, "真値:", actual, "推定値:", estm, "誤差:", diff)
        if diff <= 1.0:
            ok += 1

    ok

