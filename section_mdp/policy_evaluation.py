import sys
sys.path.append('../scripts/')
from puddle_world import *
import itertools
import collections

class PolicyEvaluator:
    def __init__(self, widths, goal, 
                    time_interval, sampling_num, 
                    lowerleft=np.array([-4,-4]).T, upperright=np.array([4,4]).T):
        self.pose_min = np.r_[lowerleft,0]
        self.pose_max = np.r_[upperright,math.pi*2] # thetaの範囲を0 - 2πで固定
        self.widths = widths
        self.goal = goal

        self.index_nums = ((self.pose_max - self.pose_min)/self.widths).astype(int) 
        nx, ny, nt = self.index_nums
        self.indexes = list(itertools.product(range(nx), range(ny), range(nt))) # 全部のインデックスの組み合わせを作成

        self.value_function, self.final_state_flags = self.init_value_function()
        self.policy = self.init_policy()

        self.actions = list(set([tuple(self.policy[i]) for i in self.indexes])) #i = [0,0,0], [0,0,1], [0,0,2], ... [0,10,0]...
        self.state_transition_probs = self.init_state_transition_probs(time_interval, sampling_num)
    
    def init_state_transition_probs(self, time_interval, sampling_num):
        ### セルの中の座標を均等に(sampling_num**3)点サンプリング
        dx = np.linspace(0.001, self.widths[0]*0.999, sampling_num) # np.linspace(最初の値, 最後の値, 要素数)：np.linspace(0, 10, 3) -> [0.0, 5.0, 10.0]
        dy = np.linspace(0.001, self.widths[1]*0.999, sampling_num)
        dt = np.linspace(0.001, self.widths[2]*0.999, sampling_num)
        samples = list(itertools.product(dx, dy, dt)) # 点数はsampling_num**3個

        ### 各行動，各方角でサンプリングした点を移動してインデックスの増分を記録
        tmp = {}
        for a in self.actions: # 各actionに対するfor
            for i_t in range(self.index_nums[2]): # 各thetaに関するfor 
                transitions = []
                for s in samples:
                    # 遷移前の姿勢 (実数)
                    before = np.array([s[0], s[1], s[2]+i_t*self.widths[2]]).T + self.pose_min # self.widths[2]はthetaの幅(10.3.2の例だとmath.pi/18)
                    # 移動前のインデクス
                    before_index = np.array([0,0,i_t]).T

                    # 遷移後の姿勢
                    after = IdealRobot.state_transition(a[0],a[1],time_interval,before)
                    # 遷移後のインデクス
                    after_index = np.floor((after - self.pose_min)/self.widths).astype(int)

                    transitions.append(after_index - before_index) # 相対的な位置(のインデクス)の変化量
                    
                # 上記のデータから集計
                unique, count = np.unique(transitions, axis=0,return_counts = True)
                probs = [c/sampling_num**3 for c in count] # サンプル数で割って確率にする
                tmp[a, i_t] = list(zip(unique,probs))
        return tmp

    def init_value_function(self):
        v = np.empty(self.index_nums)
        f = np.zeros(self.index_nums)

        for index in self.indexes:
            f[index] = self.final_state(np.array(index).T)
            v[index] = self.goal.value if f[index] else -100.0

        return v,f

    def init_policy(self):
        u_index = 2 # 制御入力が2次元
        tmp = np.zeros(np.r_[self.index_nums, u_index]) # 制御出力が2次元なので配列の次元を4次元に
        # = np.zeros([n_x, n_y, n_t, 2])
        # (x, y, theta) -> u(0) : 幅 n_x * n_y * n_t の3次元配列1個
        # (x, y, theta) -> (u(0),u(1), ..., u(d_u)) : 幅 n_x * n_y * n_t の3次元配列d_u個 -> d_u個並べて4次元
        # (x, y, theta, i) -> u(i) : 4次元
        for index in self.indexes:
            center = self.pose_min + self.widths * (np.array(index).T + 0.5)
            tmp[index] = PuddleIgnoreAgent.policy(center, self.goal)
        return tmp


    def final_state(self,index):
        x_min, y_min, _ = self.pose_min + self.widths*index         # xy平面の左下の座標
        x_max, y_max, _ = self.pose_min + self.widths*(index + 1)   # 右上の座標

        corners = [[x_min, y_min, _], [x_min, y_max, _], [x_max, y_min, _], [x_max, y_max, _]]
        return all([self.goal.inside(np.array(c).T) for c in corners])

def trial():
    pe = PolicyEvaluator(np.array([0.2, 0.2, math.pi/18]).T, Goal(-3, -3),0.1,10) # PolicyEvaluatorのオブジェクトを生成
    
    import seaborn as sns

    ### 10.3.2 後半 ###
    print(pe.state_transition_probs)
    # ((0.0, -2.0), 0): [(array([ 0,  0, -2]), 0.2), (array([ 0,  0, -1]), 0.8)]
    # u(0), u(1) = (0.0, -2.0), theta を 0 度ずらすとき，
    # インデクスの相対変化が array([ 0,  0, -2]) となる確率 0.2
    #                  array([ 0,  0, -1]) となる確率 0.8

    ### 10.3.2 ###
    # p = np.zeros(pe.index_nums)
    # for i in pe.indexes:
    #     p[i] = sum(pe.policy[i]) # 0.2(オレンジ色):直進、0.5(黒色):左回転、-0.5(クリーム色):右回転
    
    # sns.heatmap(np.rot90(p[:, :, 18]), square=False) #180~190degの向きの時の行動を図示
    # plt.show()

    ### 10.3.1 ###
    # v = pe.value_function[:,:,0]
    # sns.heatmap(np.rot90(v), square=False)
    # plt.show()

    # f = pe.final_state_flags[:,:,0]
    # sns.heatmap(np.rot90(f), square=False)
    # plt.show()
    
    # print(pe.index_nums)
    
    # 例題: policy_ evaluation1.ipynb-[4]
    # print(np.floor((pose-pe.pose_min)/pe.widths).astype(int))
    
    # pose = np.array([2.9,-2,math.pi]).T
    # print(np.floor((pose-pe.pose_min)/pe.widths).astype(int))
    
    # pose = np.array([-5,-2,math.pi/6]).T
    # print(np.floor((pose-pe.pose_min)/pe.widths).astype(int))
    
    
    

if __name__ == '__main__':
    trial()