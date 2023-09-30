import sys

sys.path.append("../scripts/")
from puddle_world import math, np
from dp_policy_agent import *
import random, copy  # copyを追加


class SarsaAgent(DpPolicyAgent):
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
    ):
        super().__init__(
            time_interval, estimator, None, puddle_coef, widths, lowerleft, upperright
        )

        nx, ny, nt = self.index_nums
        self.indexes = list(
            itertools.product(range(nx), range(ny), range(nt))
        )  # 全部のインデックスの組み合わせを作成
        self.actions = list(set([tuple(self.policy_data[i]) for i in self.indexes]))
        self.ss = self.set_action_value_function()  # PuddleIgnorePolicyの方策と価値関数の読み込み
        self.alpha = alpha  # ステップサイズパラメータ
        self.s, self.a = None, None  # 状態行動対
        self.update_end = False

    def set_action_value_function(self):  # 状態価値観数を読み込んで行動価値を初期化
        ss = {}  # StateSpace
        for line in open("puddle_ignore_values.txt", "r"):
            d = line.split()
            index = (int(d[0]), int(d[1]), int(d[2]))
            value = float(d[3])
            # インデックスをタプル, 値を数字に
            ss[index] = StateInfo(len(self.actions))  # StateInfoオブジェクトを割り当てて初期化

            for i, a in enumerate(self.actions):  # enumarateはリストの要素とインデックスを返す
                if tuple(self.policy_data[index]) == a:
                    ss[index].q[i] = value
                else:
                    ss[index].q[i] = value - 0.1

        return ss

    def policy(self, pose, goal=None):
        # 状態をタプルにしたものと、行動のインデックスで返す
        index = self.to_index(pose, self.pose_min, self.index_nums, self.widths)
        s = tuple(index)
        a = self.ss[s].pi()
        return s, a

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

        ### 行動決定と報酬の処理 ###
        s_, a_ = self.policy(
            self.estimator.pose
        )  # KFの結果から前回の状態遷移後の状態s'と次の行動a_を得る(a_はmax a'のa'ではないことに注意)
        r = self.time_interval * self.reward_per_sec()  # 状態遷移の報酬
        self.total_reward += r

        ### Q学習と現在の状態と行動の保存 ###
        self.q_update(
            self.s, self.a, r, s_, a_
        )  # self.s,aがQ値更新対象の状態行動対で、報酬rと遷移後の状態s_使って更新。
        self.s, self.a = s_, a_

        ### 出力 ###
        self.prev_nu, self.prev_omega = self.actions[a_]  # nu:速度、omegaが角速度
        return self.actions[a_]

    def q_update(self, s, a, r, s_, a_):
        if s == None:
            return

        q = self.ss[s].q[a]
        if self.in_goal:
            q_ = self.final_value
        else:
            q_ = self.ss[s_].q[a_]  # sarsaでの変更箇所 max_qからQ(s_, a_)に書き換え

        self.ss[s].q[a] = (1.0 - self.alpha) * q + self.alpha * (r + q_)

        # ログを取る
        # with open("log.txt", "a") as f:
        #     f.write(
        #         f"s:{s} r:{r} s:{s_} prev_q:{q:.2f}, next_step_max_q:{q_:.2f}, new_q:{self.ss[s].q[a]:.2f}\n"
        #     )
        # 教科書の実装(後々このlog.txt利用時は下記を参照)
        # f.write(
        #     f"{s} {r} {s_} prev_q:{q:.2f}, next_step_max_q:{q_:.2f}, new_q:{self.ss[s].q[a]:.2f}\n"
        # )


class WarpRobot(Robot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_agent = copy.deepcopy(self.agent)  # エージェントの深いコピーを残しておく

    def choose_pose(self):  # 初期位置をランダムに決めるメソッド
        xy = random.random() * 6 - 2  # -2 から +4 ???
        t = random.random() * 2 * math.pi  # 0 から 2pi
        if random.random() > 0.5:  # 確率 1/2
            return np.array([3, xy, t]).T
        else:
            return np.array([xy, 3, t]).T

    def reset(self):
        ## ssだけ残してエージェントを初期化
        tmp = self.agent.ss  # state space
        self.agent = copy.deepcopy(self.init_agent)
        self.agent.ss = tmp

        # 初期位置をセット(ロボット，カルマンフィルタ)
        self.pose = self.choose_pose()
        self.agent.estimator.belief = multivariate_normal(
            mean=self.pose, cov=np.diag([1e-10, 1e-10, 1e-10])
        )

        # 奇跡の黒い線が残らないように消す
        self.poses = []

    def one_step(self, time_interval):
        if self.agent.update_end:
            with open("log.txt", "a") as f:
                f.write("{}\n".format(self.agent.total_reward + self.agent.final_value))
            self.reset()
        else:
            super().one_step(time_interval)


class StateInfo:
    def __init__(self, action_num, epsilon=0.3):
        self.q = np.zeros(action_num)
        self.epsilon = epsilon

    def greedy(self):
        return np.argmax(self.q)

    def epsilon_greedy(self, epsilon):
        if random.random() < epsilon:
            return random.choice(range(len(self.q)))
        else:
            return self.greedy()

    def pi(self):
        return self.epsilon_greedy(self.epsilon)

    def max_q(self):
        return max(self.q)


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
    a = SarsaAgent(time_interval, kf)
    r = WarpRobot(
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
