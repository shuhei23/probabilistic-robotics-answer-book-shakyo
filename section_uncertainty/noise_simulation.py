import sys
sys.path.append('../scripts/')
from ideal_robot import *
from scipy.stats import expon, norm

class Robot(IdealRobot):
    # pass
    def __init__(self,pose,agent=None, sensor=None, color="black",\
                noise_per_meter=5, noise_std=math.pi/60,\
                bias_rate_stds=(0.1,0.1),\
                expected_stuck_time = 1e100, expected_escape_time = 1e-100):
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
    
    def one_step(self, time_interval):
        if not self.agent:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None # ここには観測ノイズが乗っている可能性がある
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega) # バイアスがかかった
        nu, omega = self.stuck(nu,omega,time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval) # ノイズは プロセスノイズ(状態量に乗ってくるノイズ)，ロボットが移動した後の座標()
        if self.sensor: # 0 か 否か -> self.sensor がありさえすれば 下記の手順
            self.sensor.data(self.pose)
            # sensor が IdealCamera で入ってきたとき， IdealCamera.data が呼ばれることになる，そこで self.lastdata = observed という更新処理がある
            # Robotクラスの親のIdealRobot
            # Worldクラスのdraw()でWorldクラスのone_step()をFuncAnimation()の引数＝1フレームごとに呼ばれる関数に設定している。
            # Worldクラスのone_step()でIdealRobotクラスのdraw()を呼ぶ
            # IdealRobotクラスのdraw()でIdealCameraクラスのdraw()を呼ぶ
            # IdealCameraクラスのdraw()でself.lastdata を参照する
        

### 以下、実行処理 ###
world = World(30,0.1)


circling = Agent(0.2, 10.0/180*math.pi)
for i in range(0,100):
    r = Robot(np.array([0,0,0]).T,sensor=None,agent=circling,color="gray",\
            noise_per_meter=0,bias_rate_stds=(0.0,0.0),\
            expected_stuck_time=60.0,expected_escape_time=60.0)
    world.append(r)
    
# nobias_robot = IdealRobot(np.array([0, 0, 0]).T, sensor=None, agent=circling, color="gray")
# world.append(nobias_robot)
# biased_robot = Robot(np.array([0, 0, 0]).T, sensor=None, agent=circling, color="red", noise_per_meter=0, bias_rate_stds=(0.2, 0.2))
# world.append(biased_robot)

r = IdealRobot(np.array([0,0,0]).T,sensor=None,agent=circling,color="red")
world.append(r)
world.draw()


