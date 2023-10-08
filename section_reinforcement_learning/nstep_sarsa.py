import sys

sys.path.append("../scripts")

from dp_policy_agent import math, np
from puddle_world import math, np
from sarsa import *


class NstepSarsaAgent(SarsaAgent):
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
        nstep=10,
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
        self.r_trace = []
        self.nstep = nstep

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
        self.r_trace.append(r)
        self.total_reward += r

        ## Q値の更新とs', a'の更新
        self.q_update(s_, a_, self.nstep)
        self.s_trace.append(s_)
        self.a_trace.append(a_)

        ## 出力
        self.prev_nu, self.prev_omega = self.actions[a_]

        return self.actions[a_]

    def q_update(self, s_, a_, n):
        # nステップまで貯まるまで待つ
        if n > len(self.s_trace) or n == 0:  # self.in_goal == trueの時の再帰呼び出しの終了条件(n==0)
            return

        s, a = self.s_trace[-n], self.a_trace[-n]  # 更新対象の状態行動対

        q = self.ss[s].q[a]
        r = sum(self.r_trace[-n:])
        print(
            "n:{0}, r_trace = {1}, r_trace[-n:] = {2}".format(
                n, self.r_trace, self.r_trace[-n:]
            )
        )
        if self.in_goal:
            # ゴールに着いたら最終値を入れる
            q_ = self.final_value
        else:
            q_ = self.ss[s_].q[a_]
        self.ss[s].q[a] = (1.0 - self.alpha) * q + self.alpha * (r + q_)

        if self.in_goal:
            # ゴールしたら1～n-1ステップ前のQ値を更新
            self.q_update(s_, a_, n - 1)


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
    a = NstepSarsaAgent(time_interval, kf)
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
