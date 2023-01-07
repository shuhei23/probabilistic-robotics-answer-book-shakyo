import sys
import math
sys.path.append('../scripts/')
from ideal_robot import *
from scipy.stats import expon, norm, uniform

class Robot(IdealRobot):
    # pass
    def __init__(self,pose,agent=None, sensor=None, color="black",\
                noise_per_meter=5, noise_std=math.pi/60,\
                bias_rate_stds=(0.1,0.1),\
                expected_stuck_time = 1e100, expected_escape_time = 1e-100,\
                expected_kidnap_time = 1e100, kidnap_range_x=(-5.0, 5.0), kidnap_range_y=(-5.0, 5.0)):
        super().__init__(pose, agent, sensor, color)
        self.noise_pdf = expon(scale=1.0/(1e-100 + noise_per_meter))
        self.distance_until_noise = self.noise_pdf.rvs()
        self.theta_noise = norm(scale = noise_std)
        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0])
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1])
        # self.bias_rate_nu = 1.0 + 0.1 * (N(0,1) の乱数) 66.3% で　0.9 - 1.1 
        self.stuck_pdf = expon(scale=expected_stuck_time)
        self.escape_pdf = expon(scale=expected_escape_time)
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        self.is_stuck = False
        self.kidnap_pdf=expon(scale=expected_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc=(rx[0], ry[0], 0.0), scale=(rx[1]-rx[0], ry[1]-ry[0], 2*math.pi)) # scipy.stats.uniform のリファレンス参照
        

    def noise(self, pose, nu, omega, time_interval):
        self.distance_until_noise -= abs(nu)*time_interval + self.r*abs(omega)*time_interval
        if self.distance_until_noise <= 0.0:
            self.distance_until_noise += self.noise_pdf.rvs() # 瑣末だけど端数を残しておくために=ではなく+=
            pose[2] += self.theta_noise.rvs()
        
        return pose

    def bias(self, nu, omega):
        return nu*self.bias_rate_nu, omega*self.bias_rate_omega
    
    def stuck(self, nu, omega, time_interval): 
        if self.is_stuck: # stuck している，つまりハマってる
            self.time_until_escape -= time_interval # ハマった時間を減算
            if self.time_until_escape <= 0.0: # 脱出
                self.time_until_escape += self.escape_pdf.rvs() # 次にハマったときに脱出までかかる時間
                self.is_stuck = False
        else: # is_stuck = falseなのでハマっていない
            self.time_until_stuck -= time_interval # ハマるまでの時間から減算
            if self.time_until_stuck <= 0.0: # ハマった
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True
        return nu*(not self.is_stuck), omega*(not self.is_stuck)
    
    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose
    
    def one_step(self, time_interval):
        if not self.agent:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None # ここには観測ノイズが乗っている可能性がある
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega) # バイアスがかかった
        nu, omega = self.stuck(nu,omega,time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval) # ノイズは プロセスノイズ(状態量に乗ってくるノイズ)，ロボットが移動した後の座標()
        self.pose = self.kidnap(self.pose, time_interval)
        if self.sensor: # 0 か 否か -> self.sensor がありさえすれば 下記の手順
            self.sensor.data(self.pose)
            # sensor が IdealCamera で入ってきたとき， IdealCamera.data が呼ばれることになる，そこで self.lastdata = observed という更新処理がある
            # Robotクラスの親のIdealRobot
            # Worldクラスのdraw()でWorldクラスのone_step()をFuncAnimation()の引数＝1フレームごとに呼ばれる関数に設定している。
            # Worldクラスのone_step()でIdealRobotクラスのdraw()を呼ぶ
            # IdealRobotクラスのdraw()でIdealCameraクラスのdraw()を呼ぶ
            # IdealCameraクラスのdraw()でself.lastdata を参照する

class Camera(IdealCamera):
    def __init__(self, env_map, 
        distance_range=(0.5, 6.0), 
        direction_range=(-math.pi/3, math.pi/3), 
        distance_noise_rate=0.1, direction_noise=math.pi/90,
        distance_bias_rate_stddev = 0.1, direction_bias_stddev = math.pi/90,
        phantom_prob = 0.0, phantom_range_x =(-5.0, 5.0), phantom_range_y=(-5.0, 5.0),
        oversight_prob=0.1, occlusion_prob = 0.0 ): # init()の引数ここまで
        super().__init__(env_map, distance_range, direction_range) # 元のinitを呼び出す

        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise
        self.distance_bias_rate_std = norm.rvs(scale = distance_bias_rate_stddev)
        self.direction_bias = norm.rvs(scale = direction_bias_stddev)
        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc=(rx[0], ry[0]), scale = (rx[1]-rx[0], ry[1]-ry[0]))
        self.phantom_prob = phantom_prob
        self.oversight_prob = oversight_prob
        self.occlusion_prob = occlusion_prob

    def noise(self, relpos):
        ell = norm.rvs(loc=relpos[0], scale=relpos[0]*self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise)
        return np.array([ell, phi]).T

    def bias(self, relpos):
        return relpos + np.array([relpos[0]*self.distance_bias_rate_std, self.direction_bias]).T
    
    def phantom(self, cam_pose, relpos):
        if uniform.rvs() < self.phantom_prob:
            pos_phantom = np.array(self.phantom_dist.rvs()).T
            return self.observation_function(cam_pose, pos_phantom)
        else:
            return relpos

    def oversight(self,relpos):
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpos

    def occlusion(self, relpos): # 観測対象の大きさを画像処理で求めたいときに，通行者や何かに隠れて実際よりも小さく（遠く）見えるという現象
        if uniform.rvs() < self.occlusion_prob:
            ell = relpos[0] + uniform.rvs()*(self.distance_range[1] - relpos[0]) 
            # (1-alpha) * relpose[0] + alpha * self.distance_range[1](=最大距離) と凸結合になっているので，
            # 最大距離とrelpose[0]の間の値になる
            phi = relpos[1]
            return np.array([ell, phi]).T
        else:
            return relpos

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            z = self.phantom(cam_pose, z)
            z = self.occlusion(z)
            z = self.oversight(z) # phantom も oversight しうる
            if self.visible(z):
                z = self.bias(z)
                z = self.noise(z)
                observed.append((z, lm.id))

        self.lastdata = observed
        return observed


if __name__ == '__main__':

    ### 以下、実行処理 ###
    world = World(30,0.1)
    
    ### 地図を生成してランドマークを追加
    m = Map()
    m.append_landmark(Landmark(-4, 2))
    m.append_landmark(Landmark(2, -3))
    m.append_landmark(Landmark(3, 3))
    world.append(m)
    
    ### ロボットを作る
    circling = Agent(0.2, 10.0/180*math.pi)
    r = Robot(np.array([0,0,0]).T,sensor=Camera(m),agent=circling)
    world.append(r)
        
    # nobias_robot = IdealRobot(np.array([0, 0, 0]).T, sensor=None, agent=circling, color="gray")
    # world.append(nobias_robot)
    # biased_robot = Robot(np.array([0, 0, 0]).T, sensor=None, agent=circling, color="red", noise_per_meter=0, bias_rate_stds=(0.2, 0.2))
    # world.append(biased_robot)
    
    # r = IdealRobot(np.array([0,0,0]).T,sensor=None,agent=circling,color="red")
    # world.append(r)
    
    ### アニメーション実行
    world.draw()
    
    
    