import sys
sys.path.append('../scripts/')
from kld_mcl import *

class GlobalKldMcl(KldMcl):
    def __init__(self, envmap, max_num, motion_noise_stds={"nn":0.19,"no":0.001,"on":0.13,"oo":0.2},\
        distance_dev_rate=0.14, direction_dev=0.05):
        super().__init__(envmap, np.array([0,0,0]), max_num, motion_noise_stds, distance_dev_rate, direction_dev)
        # パーティクル作り直し
        self.particles = [Particle(None, 1.0/max_num) for i in range(max_num)]
        for p in self.particles:
            p.pose = np.array([np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0), np.random.uniform(-math.pi, math.pi)]).T
        
        self.observed = False # ランドマークを観測している時はTrueにして、無駄なKLDサンプリングを無くす
    
    def motion_update(self, nu, omega, time):
        # 観測がなく、パーティクル数が上限なら単にパーティクルを動かして終わり
        if not self.observed and len(self.particles) == self.max_num:
            for p in self.particles:
                p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)
        else:
            super().motion_update(nu, omega, time)
        return
    
    # 同じ名前の関数の引数の
    # IF 引数の数が同じ # 注: C#/C++ だと，型まで一緒，区別がつかない
    # => オーバーライド (over-ride) 上書き
    # 　 ⇒子クラスで定義したメソッドが呼ばれる
    # 　 親クラスのメソッド呼ぶにはsuper().関数名 にする必要がある。
    # ELSE # 引数の数が違うとき
    # def motion_update(self, nu):
    #    omega = 1.1
    #    return super.motion_update(self, nu, omega, time)
    # => オーバーロード (前に定義したやつと，今定義したやつ，どっちも使える)
    
    def observation_update(self, observation):
        super().observation_update(observation)
        self.observed = len(observation) > 0


def trial(animation):
    time_interval = 0.1
    world = World(30, time_interval, debug=not animation) # アニメーションのON/OFFをdebugで制御

    ## 地図を生成して3つのランドマークを追加
    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    ## ロボットを作る

    init_pose = np.array([np.random.uniform(-5.0,5.0), np.random.uniform(-5.0,5.0), np.random.uniform(-math.pi,math.pi)]).T
    pf = GlobalKldMcl(m,100)
    a = EstimationAgent(time_interval, 0.2, 10/180.0*math.pi, pf)
    r = Robot(init_pose,sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()

    return (r.pose, pf.pose) # 真の姿勢と推定姿勢を返す


if __name__=='__main__':
    ok = 0
    for i in range(1000):
        actual, estm = trial(False)
        diff = math.sqrt((actual[0] - estm[1])**2 + (actual[1] - estm[1])**2 )
        print(i, "真値:", actual, "推定値:", estm, "誤差:", diff)
        if diff <= 1.0:
            ok += 1

    ok
