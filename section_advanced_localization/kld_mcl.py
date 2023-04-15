import sys
sys.path.append('../scripts')
from mcl import *
from scipy.stats import chi2 

class KldMcl(Mcl):
    def __init__(self, envmap, init_pose, max_num, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14, direction_dev=0.05,
                 widths = np.array([0.2, 0.2, math.pi/18]).T, epsilon=0.1, delta=0.01): #この行が新しくKLD用のパラメータ 
        super().__init__(envmap, init_pose, 1, motion_noise_stds, distance_dev_rate, direction_dev)
        self.widths = widths    #各ビンのxyθの幅
        self.max_num = max_num  #パーティクルの上限
        self.epsilon = epsilon  #ε
        self.delta = delta      #δ
        self.binnum = 0         #ビンの数k 本来、ローカルの変数でいいけど描画用にオブジェクトに持たせる


    def motion_update(self, nu, omega, time):
        ws = [e.weight for e in self.particles]
        if sum(ws) < 1e-100:
            ws = [e + 1e-100 for e in ws] #重みのわがゼロに丸め込まれるとサンプリングできなくなるので小さな数を足しておく
        
        new_particles = [] #新しいパーティクルのリスト(最終的にself.particlesになる)
        bins = set()
        # 式(7.21)を満たすまでNを1ずつ増やしながらyを計算していく
        for i in range(self.max_num):
            # 手順1：パーティクルを一つp(x|x_{t-1},u_{t})からドローしてXYΘ空間に置く
            chosen_p = random.choices(self.particles, weights=ws)
            p = copy.deepcopy(chosen_p[0]) # p は particle の p (たぶん)
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)    #移動
            # 手順2：パーティクルが入っているビンの数を数える
            bins.add(tuple(math.floor(e) for e in p.pose/self.widths))      #ビンのインデックスをsetに登録(角度を正規化するとより良い)
            new_particles.append(p)                                         #新しいパーティクルのリストに追加
            # 手順3：パーティクルの数が超えたら終了
            self.binnum = len(bins) if len(bins) > 1 else 2 #ビンの数が1の場合に2にしないと次の行の計算ができない
            if len(new_particles) > math.ceil(chi2.ppf(1.0 - self.delta, self.binnum - 1))/(2*self.epsilon): #(7.21) N > y/2ε
                break # 条件満たすくらいN(パーティクル数)が増えたらおしまい
        
        self.particles = new_particles
        for i in range(len(self.particles)): #正規化
            self.particles[i].weight = 1.0 / len(self.particles)
    
    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        self.set_ml()
    
    def draw(self, ax, elems):
        super().draw(ax, elems)
        elems.append(ax.text(-4.5, -4.5, "particle:{}, bin:{}".format(len(self.particles), self.binnum), fontsize=10))
    
def trial():
    time_interval = 0.1
    world = World(30, time_interval, debug=False)
    
    ## 地図を生成して2つランドマークを追加
    m = Map()
    for ln in [(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)
    ### ロボットを作る
    initial_pose = np.array([0,0,0]).T
    pf = KldMcl(m,initial_pose,100)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi,pf)
    r = Robot(initial_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)
    ### アニメーション実行
    world.draw()

if __name__ == '__main__':
    trial()  
        