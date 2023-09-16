import sys
sys.path.append('../scripts/')
from puddle_world import *
import itertools
import collections

class DynamicProgramming:
    def __init__(self, widths, goal, puddles, 
                    time_interval, sampling_num, 
                    puddle_coef=100.0 ,lowerleft=np.array([-4,-4]).T, upperright=np.array([4,4]).T):
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
        
        self.depths = self.depth_means(puddles, sampling_num) # self.index_nums の定義のあとに呼ばないとエラー出る
        
        self.time_interval = time_interval
        self.puddle_coef = puddle_coef

    def value_iteration_sweep(self):
        max_delta = 0.0
        for index in self.indexes:
            if not self.final_state_flags[index]:
                max_q = -1e100
                max_a = None
                qs = [self.action_value(a, index) for a in self.actions] # 全行動の行動価値を計算，(状態)行動価値関数
                max_q = max(qs) # 最大の行動価値
                max_a = self.actions[np.argmax(qs)] # 最大の行動価値を与える行動

                delta = abs(self.value_function[index] - max_q) # 状態価値関数の変化量
                max_delta = delta if delta > max_delta else max_delta

                self.value_function[index] = max_q # 価値の更新
                self.policy[index] = np.array(max_a).T
    
        return max_delta
        
    def policy_evalutation_sweep(self):
        max_delta = 0.0
        for index in self.indexes:
            if not self.final_state_flags[index]:
                q = self.action_value(tuple(self.policy[index]),index)

                delta = abs(self.value_function[index] - q)
                max_delta = delta if delta > max_delta else max_delta
                
                self.value_function[index] = q
                # self.value_function[index] = self.action_value(tuple(self.policy[index]), index) #actionはタプルに直してから与える
        
        return max_delta
    
    def action_value(self, action, index, out_penalty = True):
        value = 0.0
        for delta, prob in self.state_transition_probs[(action, index[2])]: # index[2]:方角のインデックス, action = s' = 移動先のマスのインデックス
            # after = tuple(self.out_correction(np.array(index).T + delta)) #indexに差分deltaを足してはみ出し処理の後にタプル
            after, out_reward = self.out_correction(np.array(index).T + delta)
            after = tuple(after)
            reward = - self.time_interval * self.depths[(after[0], after[1])] * self.puddle_coef - self.time_interval + out_reward * out_penalty# 最後の　- self.time_interval は移動にかかった時間の計算
            value += (self.value_function[after] + reward) * prob
        
        return value

    def out_correction(self, index):
        out_reward = 0.0
        index[2] = (index[2] + self.index_nums[2]) % self.index_nums[2] #方角の処理

        for i in range(2):
            if index[i] < 0:
                index[i] = 0
                out_reward = -1e100
            elif index[i] >= self.index_nums[i]:
                index[i] = self.index_nums[i] - 1
                out_reward = -1e100

        return index, out_reward
            
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

    def depth_means(self,puddles,sampling_num):
        # セルの中の座標を均等にsampling_num**2点サンプリング
        dx = np.linspace(0, self.widths[0], sampling_num)
        dy = np.linspace(0, self.widths[1], sampling_num)
        samples = list(itertools.product(dx,dy))
        tmp = np.zeros(self.index_nums[0:2])
        for xy in itertools.product(range(self.index_nums[0]),range(self.index_nums[1])): # 空間離散化点に関する for 
        # xy = [0,0], [0,1], ... [0,self.index_nums[1]-1], 
        #      ... [self.index_numx[0]-1,self.index_nums[1]-1] みたいにうごく
            for s in samples: # 相対的な移動に関する for 
                pose = self.pose_min + self.widths*np.array([xy[0],xy[1],0]).T + np.array([s[0],s[1],0]).T
                # self.widths*np.array([xy[0],xy[1],0]).T 空間離散化
                # np.array([s[0],s[1],0]).T particleの相対的な移動分
                for p in puddles: 
                    tmp[xy] += p.depth*p.inside(pose) # 深さに水たまりの中か否か(1 or 0)をかけて足す
            tmp[xy] /= sampling_num**2 # 深さの合計から平均値に変換

        return tmp

def trial():    
    import seaborn as sns
    
    ### 10.3.3 
    puddles = [Puddle((-2,0),(0,2),0.1),Puddle((-0.5,-2),(2.5,1),0.1)]
    dp = DynamicProgramming(np.array([0.2, 0.2, math.pi/18]).T, Goal(-3, -3),puddles,0.1,10)
    counter = 0 #スイープの回数
  
    delta = 1e100

    # while delta > 0.01:
    for i in range(10):
        delta = dp.value_iteration_sweep()
        counter += 1
        print(counter, delta)
          



if __name__ == '__main__':
    trial()