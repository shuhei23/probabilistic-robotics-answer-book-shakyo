import sys

sys.path.append("../scripts/")

from puddle_world import np
from dynamic_programming import *


class BeliefDynamicProgramming(DynamicProgramming):
    def __init__(
        self,
        widths,
        goal,
        puddles,
        time_interval,
        sampling_num,
        puddle_coef=100,
        lowerleft=np.array([-4, -4]).T,
        upperright=np.array([4, 4]).T,
        dev_borders=[0.1, 0.2, 0.4, 0.8],
    ):
        super().__init__(
            widths,
            goal,
            puddles,
            time_interval,
            sampling_num,
            puddle_coef,
            lowerleft,
            upperright,
        )

        self.index_nums = np.array(
            [*self.index_nums, len(dev_borders) + 1]
        )  # もう一次元加える dev_bordersがσ
        # ↑ 0 <= σ < 0.1 → i_σ = 0, 0.1 <= σ < 0.2 → i_σ = 1 ... 0.8 <= σ → i_σ = 5 なので +1 する
        nx, ny, nt, nh = self.index_nums
        self.indexes = list(
            itertools.product(range(nx), range(ny), range(nt), range(nh))
        )

        self.value_function, self.final_state_flags = self.init_belief_value_function()
        self.policy = np.zeros(np.r_[self.index_nums, 2])  # 全部ゼロで初期化
        self.dev_borders = dev_borders
        self.dev_borders_side = [
            dev_borders[0] / 10,
            *dev_borders,
            dev_borders[-1] * 10,
        ]  # = [0.01, 0.1, 0.2, 0.4, 0.8, 8]
        self.motion_sigma_transition_probs = (
            self.init_motion_sigma_transition_probs()
        )  # 辞書型のデータが格納される

    def init_motion_sigma_transition_probs(self):
        probs = {}
        for a in self.actions:
            for i in range(len(self.dev_borders) + 1):
                probs[(i, a)] = self.calc_motion_sigma_transition_probs(
                    self.dev_borders_side[i], self.dev_borders_side[i + 1], a
                )
        # key:index, action, value:遷移後のindexとがその確率（indexがkey, 確率がvalueの辞書型
        return probs

    def cov_to_index(self, cov):
        # 共分散行列からsigmaを求めてindexに変換
        sigma = np.power(np.linalg.det(cov), 1.0 / 6.0)
        for i, e in enumerate(self.dev_borders):
            if sigma < e:
                return i

        return len(self.dev_borders)

    def calc_motion_sigma_transition_probs(
        self, min_sigma, max_sigma, action, sampling_num=100
    ):
        # sigmaと行動の組み合わせについてsigmaの遷移確率を計算する
        nu, omega = action
        if abs(omega) < 1e-5:
            omega = 1e-5

        F = matF(nu, omega, self.time_interval, 0.0)  # ロボットの向きは関係ないので0degで固定
        M = matM(
            nu,
            omega,
            self.time_interval,
            {"nn": 0.19, "no": 0.001, "on": 0.13, "oo": 0.2},
        )  # 移動後の誤差モデル
        A = matA(nu, omega, self.time_interval, 0.0)

        ans = {}

        for sigma in np.linspace(
            min_sigma, max_sigma * 0.999, sampling_num
        ):  # 遷移前のσを作る（区間内に一様に分布していると仮定）
            index_after = self.cov_to_index(
                sigma * sigma * F.dot(F.T) + A.dot(M).dot(A.T)
            )
            if index_after not in ans:
                ans[index_after] = 1
            else:
                ans[index_after] += 1

        for e in ans:
            ans[e] /= sampling_num

        return ans

    def init_belief_value_function(self):
        v = np.empty(self.index_nums)
        f = np.zeros(self.index_nums)

        for index in self.indexes:
            f[index] = self.belief_final_state(np.array(index).T)
            if f[index]:
                v[index] = self.goal.value
            else:
                v[index] = -100.0
        return v, f

    def belief_final_state(self, index):
        x_min, y_min, _ = self.pose_min + self.widths * index[0:3]
        x_max, y_max, _ = self.pose_min + self.widths * (index[0:3] + 1)
        # ↑どっちを向いていてもゴールの範囲内外に関係ないのでthetaは読み捨ててよい
        corners = [
            [x_min, y_min, _],
            [x_min, y_max, _],
            [x_max, y_min, _],
            [x_max, y_max, _],
        ]
        return (
            all([self.goal.inside(np.array(c).T) for c in corners]) and index[3] == 0
        )  # エントロピーのインデックスが0であることも条件

    def action_value(self, action, index, out_penalty=True):
        value = 0.0
        # print("action = {0}, index = {1}".format(action,index))
        # print("delta, prob = {}".format(self.state_transition_probs[(action, index[2])]))
        for delta, prob in self.state_transition_probs[(action, index[2])]:
            after, out_reward = self.out_correction(
                np.array(index[0:3]).T + delta
            )  # index を4次元から3次元に
            reward = (
                -self.time_interval
                * self.depths[(after[0], after[1])]
                * self.puddle_coef
                - self.time_interval
                + out_reward * out_penalty
            )
            for sigma_after, sigma_prob in self.motion_sigma_transition_probs[
                (index[3], action)
            ].items():
                # print("sigma_after, sigma_prob  = {0}, {1}".format(sigma_after, sigma_prob))
                # print("[*after, sigma_after] = {}".format([*after, sigma_after]))
                value += (
                    (self.value_function[tuple([*after, sigma_after])] + reward)
                    * prob
                    * sigma_prob
                )

        return value


puddles = [Puddle((-2, 0), (0, 2), 0.1), Puddle((-0.5, -2), (2.5, 1), 0.1)]
dp = BeliefDynamicProgramming(
    np.array([0.2, 0.2, math.pi / 18]).T, Goal(-3, -3), puddles, 0.1, 10
)
delta = dp.value_iteration_sweep()

import seaborn as sns

v = dp.value_function[:, :, 18, 1]  # 4番目のインデックス: エントロピーの離散区分のインデックス
sns.heatmap(np.rot90(v), square=False)
plt.show()
v = dp.value_function[:, :, 18, 0]
sns.heatmap(np.rot90(v), square=False)
plt.show()
# print(dp.motion_sigma_transition_probs)
