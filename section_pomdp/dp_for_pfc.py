import sys

sys.path.append("../scripts/")
from dynamic_programming import *
import seaborn as sns


def main():
    puddles = []
    dp = DynamicProgramming(
        np.array([0.2, 0.2, math.pi / 18]).T, Goal(-1.5, -1.5), puddles, 0.1, 10
    )
    counter = 0

    delta = 1e100

    while delta > 0.01:
        delta = dp.value_iteration_sweep()
        counter += 1
        print(counter, delta)

    # データ保存
    with open("policy_for_pfc.txt", "w") as f:  # 後からpolicy.txtにコピー（あるいはシンボリックリンク）
        for index in dp.indexes:
            p = dp.policy[index]
            f.write("{} {} {} {} {}\n".format(index[0], index[1], index[2], p[0], p[1]))

    with open("value_for_pfc.txt", "w") as f:  # 後からvalue.txtにコピー（あるいはシンボリックリンク）
        for index in dp.indexes:
            p = dp.value_function[index]
            f.write("{} {} {} {}\n".format(index[0], index[1], index[2], p))

    # 描画
    v = dp.value_function[:, :, 18]  ###dp1valuedraw
    sns.heatmap(np.rot90(v), square=False)
    plt.show()

    p = np.zeros(dp.index_nums) ###dp1policydraw
    for i in dp.indexes:
        p[i] = sum(dp.policy[i]) #速度と角速度を足すと、1.0: 直進、2.0: 左回転、-2.0: 右回転になる

    sns.heatmap(np.rot90(p[:, :, 18]), square=False) #180〜190[deg]の向きのときの行動を図示
    plt.show()


if __name__ == "__main__":
    main()