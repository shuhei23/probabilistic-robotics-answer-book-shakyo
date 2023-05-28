import sys
sys.path.append('../scripts/')
from kf import*

class EstimatedLandmark(Landmark):
    def __init__(self):
        super().__init__(0, 0)
        self.cov = np.array([[1, 0], [0, 2]]) #描画テスト用 あとで削除
        
    def draw(self, ax, elems):
        if self.cov is None: #共分散が設定されていない時は描画しない
            return
        ###推定位置に青い星を描く###
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="blue")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))
        
        ###誤差楕円を描く###
        e = sigma_ellipse(self.pos, self.cov, 3)
        elems.append(ax.add_patch(e))

class MapParticle(Particle):
    def __init__(self, init_pose, weight, landmark_num):
        super().__init__(init_pose, weight)
        self.map = Map() # Mapクラスのインスタンスを生成
        
        for i in range(landmark_num):
            self.map.append_landmark(EstimatedLandmark())

class FastSlam(Mcl):
    def __init__(self, envmap, init_pose, particle_num, landmark_num, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2 }, distance_dev_rate=0.14, direction_dev=0.05):
        super().__init__(envmap, init_pose, particle_num, motion_noise_stds, distance_dev_rate, direction_dev)

        self.particles = [MapParticle(init_pose, 1.0/particle_num, landmark_num) for i in range(particle_num)]
        self.ml = self.particles[0] # 最尤のパーティクルを新しく作ったパーティクルのリストを先頭にしておく
        
    def draw(self, ax, elems):
        super().draw(ax, elems)
        self.ml.map.draw(ax, elems)

def trial():
    time_interval = 0.1
    world = World(30,time_interval, debug=False)

    ## 真の地図を作成 ##
    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
        # Landmark は ideal_robot.py に書いてある
        # *ln はアンパック: Landmark(*(-4,2)) = Landmark(-4,2)
    world.append(m)
    
    ### ロボットを作る
    initial_pose = np.array([0,0,0]).T
    estimator = FastSlam(m,initial_pose,100, len(m.landmarks))
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi,estimator)
    r = Robot(initial_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)
    
    world.draw()
    
if __name__ == '__main__':
    trial()
