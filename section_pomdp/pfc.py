import sys

sys.path.append("../scripts/")
from qmdp import *
from sensor_reset_mcl import *


class PfcAgent(QmdpAgent):
    def __init__(
        self,
        time_interval,
        estimator,
        goal,
        puddles,
        sampling_num=10,
        widths=np.array([0.2, 0.2, math.pi / 18]).T,
        puddle_coef=100,
        lowerleft=np.array([-4, -4]).T,
        upperright=np.array([4, 4]).T,
        magnitude=2
    ):
        super().__init__(
            time_interval,
            estimator,
            goal,
            puddles,
            sampling_num,
            widths,
            puddle_coef,
            lowerleft,
            upperright,
        )
        self.magnitude = magnitude

    def evaluation(self, action, indexes): 
        v = self.dp.value_function
        vs = [abs(v[i]) if abs(v[i]) > 0.0 else 1e-10 for i in indexes]
        qs = [self.dp.action_value(action, i, out_penalty=False) for i in indexes]

        return sum([q/(v**self.magnitude) for (v,q) in zip(vs,qs)])

    def policy(self, pose, goal=None):
        # 本当はパーティクルのクラスに実装しないといけないのですが・・・"

        # ゴールに入ったパーティクルの重みを下げる
        for p in self.estimator.particles:
            if self.goal.inside(p.pose):
                p.weight *= 1e-10
        self.estimator.resampling()

        return super().policy(pose, goal)


def trial(animation):
    time_interval = 0.1
    world = PuddleWorld(300, time_interval, debug=not animation)

    ## ランドマークの追加(1つだけにした) ##
    m = Map()
    m.append_landmark(Landmark(0, 0))
    world.append(m)

    ## ゴールの追加
    goal = Goal(-1.5, -1.5)
    world.append(goal)

    # ## 水たまりの追加
    # puddles = [Puddle((-2, 0), (0, 2), 0.1), Puddle((-0.5, -2), (2.5, 1), 0.1)]
    # world.append(puddles[0])
    # world.append(puddles[1])

    ## ロボットを作る ##
    init_pose = np.array([3.5, 3.5, np.pi]).T
    pf = ResetMcl(m, [-10, -10, 0], 100)
    a = PfcAgent(time_interval, pf, goal, [])
    r = Robot(
        init_pose,
        sensor=Camera(m),
        agent=a,
        color="red",
    )
    world.append(r)
    world.draw()
    return a


if __name__ == "__main__":
    trial(True)
