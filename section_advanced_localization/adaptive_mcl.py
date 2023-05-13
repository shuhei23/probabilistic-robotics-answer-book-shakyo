import sys
sys.path.append('../scripts/')
from mcl import*

class ResetMcl(Mcl):
    def __init__(self,envmap,init_pose,num,motion_noise_stds={"nn":0.19, "no":0.001,"on":0.13,"oo":0.2}, \
                 distance_dev_rate=0.14, direction_dev=0.05, amcl_params={"slow":0.001,"fast":0.1,"nu":3.0}):
        super().__init__(envmap, init_pose, num, motion_noise_stds, distance_dev_rate, direction_dev)
        self.amcl_params=amcl_params
        self.slow_term_alpha, self.fast_term_alpha = 1.0, 1.0 # alpha_fast, alpha_slow の初期値
        
    
    def sensor_resetting_draw(self, particle, landmark_pos, ell_obs, phi_obs):
        # センサリセット
        ##パーティクルの位置を決める##
        psi = np.random.uniform(-np.pi, np.pi) #ランドマークからの方角を選ぶ
        ell = norm(loc=ell_obs, scale=(ell_obs*self.distance_dev_rate)**2).rvs() #ランドマークからの距離ellを選ぶ，norm は scipy.statsの正規分布のコマンド
        particle.pose[0] = landmark_pos[0] + ell*math.cos(psi)
        particle.pose[1] = landmark_pos[1] + ell*math.sin(psi)
        
        ##パーティクルの向きを決める##
        phi = norm(loc=phi_obs, scale=(self.direction_dev)**2).rvs() #ランドマークが見える向きを決める
        particle.pose[2] = math.atan2(landmark_pos[1] - particle.pose[1], landmark_pos[0] - particle.pose[0]) - phi
        
        particle.weight = 1.0/len(self.particles) # 重みを1/Nに

    def sensor_resetting(self,observation):
        nearest_obs = np.argmin([obs[0][0] for obs in observation])
        values, lankmark_id = observation[nearest_obs]

        for p in self.particles:
            self.sensor_resetting_draw(p, self.map.landmarks[lankmark_id].pos, *values)

    def adaptive_resetting(self, observation):
        if len(observation) == 0:
            return
        
        ## センサリセットするパーティクルの数を決める 式(7.30), (7.31), (7.32) ##
        alpha = sum([p.weight for p in self.particles])
        self.slow_term_alpha += self.amcl_params["slow"]*(alpha - self.slow_term_alpha)
        self.fast_term_alpha += self.amcl_params["fast"]*(alpha - self.fast_term_alpha)
        sl_num = len(self.particles)*max([0, 1.0-self.amcl_params["nu"]*self.fast_term_alpha/self.slow_term_alpha])

        self.resampling() # とりあえず普通にリサンプリング

        nearest_obs = np.argmin([obs[0][0] for obs in observation]) # 距離が一番近いランドマークを選択
        values, landmark_id = observation[nearest_obs]
        for n in range(int(sl_num)): # n回パーティクルを選んで姿勢を変える（2回以上姿勢を変えられるパーティクルがあるけど気にしない）
            p = random.choices(self.particles)[0] # 一つ選ぶ
            self.sensor_resetting_draw(p, self.map.landmarks[landmark_id].pos, *values)
            
    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)

        self.set_ml()
        self.adaptive_resetting(observation) # 変更
    
def trial_phantom(animation): ###mclkidnap1test
    time_interval = 0.1
    world = World(300, time_interval, debug=not animation) 

    ## 地図を生成して3つランドマークを追加 ##
    m = Map()
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)

    ## ロボットを作る ##
    init_pose = np.array([np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0), np.random.uniform(-math.pi, math.pi)]).T
    robot_pose = np.array([np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0), np.random.uniform(-math.pi, math.pi)]).T
    pf = ResetMcl(m, init_pose, 1000)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, pf)
    r = Robot(robot_pose, sensor=Camera(m, phantom_prob=0.1), agent=a, color="red")
    world.append(r)

    world.draw()
    
    return (pf)

if __name__ == '__main__':
    estimator = trial_phantom(True)