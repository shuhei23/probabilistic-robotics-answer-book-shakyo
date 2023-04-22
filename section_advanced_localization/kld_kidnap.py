import sys
sys.path.append('../scripts/')
from kld_mcl import *

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
    robot_pose = np.array([np.random.uniform(-5.0,5.0), np.random.uniform(-5.0,5.0), np.random.uniform(-math.pi,math.pi)]).T
    pf = KldMcl(m,init_pose,100)
    a = EstimationAgent(time_interval, 0.2, 10/180.0*math.pi, pf)
    r = Robot(robot_pose,sensor=Camera(m), agent=a, color="red")
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
