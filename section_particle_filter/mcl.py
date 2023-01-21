import sys
sys.path.append('../scripts/')
from robot import *
from scipy.stats import multivariate_normal

class Particle:
    def __init__(self,init_pose):
        self.pose = init_pose
        
    def motion_update(self, nu, omega, time, noise_rate_pdf):
        ns = noise_rate_pdf.rvs() # 順にnn, no, on, oo
        noised_nu = nu + ns[0] * math.sqrt(abs(nu)/ time) + ns[1] * math.sqrt(abs(omega)/time)
        noised_omega = omega + ns[2] * math.sqrt(abs(nu)/ time) + ns[3] * math.sqrt(abs(omega)/ time)        
        self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)

class Mcl: # Monte Carlo Localization
    def __init__(self,init_pose,num, motion_noise_stds):
        self.particles = [Particle(init_pose) for i in range(num)]
        
        v = motion_noise_stds
        c = np.diag([v["nn"] ** 2, v["no"]**2, v["on"]**2, v["oo"]**2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c) # 多変数共分散行列covから乱数ジェネレータを作成

    def motion_update(self, nu, omega, time):
        for p in self.particles:
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)
            
    def draw(self,ax,elems):
        xs=[p.pose[0] for p in self.particles]
        ys=[p.pose[1] for p in self.particles]
        vxs=[math.cos(p.pose[2]) for p in self.particles]
        vys=[math.sin(p.pose[2]) for p in self.particles]
        elems.append(ax.quiver(xs,ys,vxs,vys,color="blue",alpha=0.5))
        

class EstimationAgent(Agent):
    def __init__(self, time_interval, nu,omega,estimator):
        super().__init__(nu,omega)
        self.estimator = estimator
        self.time_interval = time_interval
        
        self.prev_nu = 0.0
        self.prev_omega = 0.0

    def decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        return self.nu, self.omega
    
    def draw(self,ax,elems):
        self.estimator.draw(ax,elems)


def trial(motion_noise_stds):
    time_interval = 0.1
    world = World(30,time_interval)

    ### ロボットを作る
    initial_pose = np.array([0,0,0]).T
    estimator = Mcl(initial_pose,100, motion_noise_stds)
    circling = EstimationAgent(0.1, 0.2, 10.0/180*math.pi,estimator)
    r = Robot(initial_pose,agent=circling, color="red")
    world.append(r)

    ### アニメーション実行
    world.draw()
    
trial({"nn": 0.01, "no": 0.02, "on": 0.03, "oo": 0.04})