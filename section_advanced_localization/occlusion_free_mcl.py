import sys
sys.path.append('../scripts/')
from mcl import*

class OcclusionFreeParticle(Particle):
    def observation_update(self, observation,envmap,distance_dev_rate,direction_dev):
        for d in observation:
            obs_pos = d[0]
            obs_id = d[1]

            ###パーティクルの位置と地図からのランドマークの距離と方角を算出###
            pos_on_map = envmap.landmarks[obs_id].pos
            particle_suggest_pos = IdealCamera.observation_function(self.pose,pos_on_map)

            ###尤度の計算###
            distance_dev=distance_dev_rate*particle_suggest_pos[0]
            cov=np.diag(np.array([distance_dev**2, direction_dev**2]))

            if(obs_pos[0] > particle_suggest_pos[0]): # 式(7.36)
                # オクルージョンによって物体が小さく見えている可能性があるため、理論値を正規分布に使う
                obs_pos[0] = particle_suggest_pos[0]
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos)


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
    initial_pose = np.array([0,0,0]).T
    estimator = Mcl(m,initial_pose,100)
    estimator.particles = [OcclusionFreeParticle(initial_pose, 1.0/100) for i in range(100)]
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi,estimator)
    #circling = EstimationAgent(0.1, 0.1, 0.0,estimator)
    r = Robot(initial_pose, sensor=Camera(m, occlusion_prob=0.5), agent=circling, color="red")
    world.append(r)

    ### アニメーション実行
    world.draw()

if __name__ == '__main__':
    trial()