import sys

sys.path.append("../scripts")
from sarsa import *


class SarsaLamdaAgent(SarsaAgent):
    def __init__(
        self,
        time_interval,
        estimator,
        puddle_coef=100,
        alpha=0.5,
        widths=np.array([0.2, 0.2, math.pi / 18]).T,
        lowerleft=np.array([-4, -4]).T,
        upperright=np.array([4, 4]).T,
        dev_borders=[0.1, 0.2, 0.4, 0.8],
        lmd=0.9,
    ):
        super().__init__(
            time_interval,
            estimator,
            puddle_coef,
            alpha,
            widths,
            lowerleft,
            upperright,
            dev_borders,
        )

        self.s_trace = []
        self.a_trace = []
        self.lmd = lmd

    def set_action_value_function(self):
        ss = {}
        for index in self.indexes:
            ss[index] = StateInfo(len(self.actions))
            for i, a in enumerate(self.actions):
                ss[index].q[i] = -1000.0

        return ss

    def decision(self, observation=None):
        ### 終了処理 ###
        if self.update_end:
            return 0.0, 0.0
        if self.in_goal:
            # print("update_end")
            self.update_end = True  # ゴールに入った後も一回だけ更新があるので即終了はしない

        ### カルマンフィルタの実行 ###
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.estimator.observation_update(observation)

        ## 行動決定と報酬の処理
        s_, a_ = self.policy(self.estimator.pose)
        r = self.time_interval * self.reward_per_sec()
        self.total_reward += r

        ## Q値の更新とs', a'の更新
        self.q_update(self.s, self.a, r, s_, a_)
        self.s = s_
        self.a = a_
        self.s_trace.append(s_)
        self.a_trace.append(a_)

        ## 出力
        self.prev_nu, self.prev_omega = self.actions[a_]

        return self.actions[a_]

    def q_update(self, s, a, r, s_, a_):
        if s == None:
            return

        q = self.ss[s].q[a]
        if self.in_goal:
            # ゴールに着いたら最終値を入れる
            q_ = self.final_value
        else:
            q_ = self.ss[s_].q[a_]
        diff = r + q_ - q  # 元々の行動価値と今観測した行動価値の差を計算しておく

        for i in range(len(self.s_trace)):
            s_back = self.s_trace[-i - 1]
            a_back = self.a_trace[-i - 1]
            self.ss[s_back].q[a_back] += self.alpha * diff * (self.lmd**i)


class WarpRobot2(WarpRobot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_pose(self):
        return np.array(
            [
                random.random() * 8 - 4,
                random.random() * 8 - 4,
                random.random() * 2 * math.pi,
            ]
        ).T

    def one_step(self, time_interval):
        if self.agent.update_end:
            with open("log.txt", "a") as f:
                f.write("{}\n".format(self.agent.total_reward + self.agent.final_value))
            self.reset()
            return
        elif len(self.poses) >= 300:
            with open("log.txt", "a") as f:
                f.write("DNF\n")
            self.reset()
            return

        super().one_step(time_interval)


def trial():
    time_interval = 0.1
    world = PuddleWorld(400000, time_interval, debug=False)  # 長時間アニメーションを取る

    m = Map()
    for ln in [(-4, 2), (2, -3), (4, 4), (-4, -4)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    ## ゴールの追加
    goal = Goal(-3, -3, radius=2)
    world.append(goal)

    ## 水たまりの追加
    world.append(Puddle((-2, 0), (0, 2), 0.1))
    world.append(Puddle((-0.5, -2), (2.5, 1), 0.1))

    ## ロボットを1台登場させる##
    init_pose = np.array([3, 3, 0]).T
    kf = KalmanFilter(m, init_pose)
    a = SarsaLamdaAgent(time_interval, kf)
    r = WarpRobot2(
        init_pose,
        sensor=Camera(m, distance_bias_rate_stddev=0, direction_bias_stddev=0),
        agent=a,
        color="red",
        bias_rate_stds=(0, 0),
    )
    world.append(r)
    world.draw()


if __name__ == "__main__":
    trial()
