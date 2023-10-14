import sys

sys.path.append("../scripts/")
from dp_policy_agent import *


def trial(animation):
    time_interval = 0.1
    world = PuddleWorld(30, time_interval, debug=not animation)

    ## ランドマークの追加(意地悪な位置に) ##
    m = Map()
    for ln in [(1, 4), (4, 1), (-4, -4)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    ## ゴールの追加
    goal = Goal(-3, -3, radius=2)
    world.append(goal)

    ## 水たまりの追加
    world.append(Puddle((-2, 0), (0, 2), 0.1))
    world.append(Puddle((-0.5, -2), (2.5, 1), 0.1))

    ## ロボットを作る ##
    init_pose = np.array([2.5, 2.5, 0]).T
    pf = Mcl(m, init_pose, 100)
    a = DpPolicyAgent(time_interval, pf, goal)
    r = Robot(
        init_pose,
        sensor=Camera(m),
        agent=a,
        color="red",
    )
    world.append(r)
    world.draw()


if __name__ == "__main__":
    trial(True)
