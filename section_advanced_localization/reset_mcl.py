import sys
sys.path.append('../scripts/')
from mcl import*

class ResetMcl(Mcl):
    def __init__(self, envmap, init_pose, num, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},\
        distance_dev_rate=0.14, direction_dev=0.05):
        super().__init__(envmap, init_pose, num, motion_noise_stds, distance_dev_rate, direction_dev)
        self.alphas = {} # alpha値の記録（ランドマークの観測数ごと）
    

    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        
        # alpha値の記録
        alpha = sum([p.weight for p in self.particles]) # 式(7.28)
        obsnum = len(observation)
        if not obsnum in self.alphas:
            self.alphas[obsnum] = [] # 
                                     # 例： {"0":[0.001, 0.002], "1":[0.01, 0.002], "2":[0.1]}
        
        self.alphas[obsnum].append(alpha)

        self.set_ml()
        self.resampling() # ここで重みの合計は1になる
        

def trial():
    time_interval = 0.1
    world = World(30,time_interval, debug=False)

    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
        # Landmark は ideal_robot.py に書いてある
        # *ln はアンパック: Landmark(*(-4,2)) = Landmark(-4,2)
    world.append(m)

    ### ロボットを作る
    initial_pose = np.array([-4,-4,0]).T
    estimator = ResetMcl(m,initial_pose,100)
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    
    initial_pose_robot = np.array([0,0,0]).T
    r = Robot(initial_pose_robot, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    ### アニメーション実行
    world.draw()
    
    return estimator

if __name__ == '__main__':
    estimator = trial()
    for n in estimator.alphas.keys(): # ["0", "1", "2"]
        print("landmarks: ", n, "particles: ", len(estimator.particles), "min: ", min(estimator.alphas[n]), "max: ", max(estimator.alphas[n]))
    