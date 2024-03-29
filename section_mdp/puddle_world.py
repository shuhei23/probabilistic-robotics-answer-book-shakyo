import sys
sys.path.append('../scripts/')
from kf import *

class Goal:
    def __init__(self, x, y, radius=0.3, value=0.0):
        self.pos = np.array([x,y]).T
        self.radius = radius
        self.value = value

    def draw(self, ax, elems):
        x, y = self.pos
        c = ax.scatter(x + 0.16, y + 0.5, s=50, marker=">", label = "landmarks", color="red") # 旗
        elems.append(c)
        elems += ax.plot([x,x], [y, y + 0.6], color = "black") # 旗竿
    def inside(self, pose):
        return self.radius > math.sqrt((self.pos[0]-pose[0])**2 + (self.pos[1]-pose[1])**2)

class Puddle:
    def __init__(self, lowerleft, upperright, depth):
        self.lowerleft = lowerleft
        self.upperright = upperright
        self.depth = depth

    def draw(self, ax, elems):
        w = self.upperright[0] - self.lowerleft[0]
        h = self.upperright[1] - self.lowerleft[1]
        r = patches.Rectangle(self.lowerleft, w, h, color="blue", alpha=self.depth)
        elems.append(ax.add_patch(r))  

    def inside(self, pose):
        # 1個の池に対して1個のロボットが池にはまっているか調べる
        return all([ self.lowerleft[i] < pose[i] < self.upperright[i] for i in [0,1]])
        # all: すべて True なら True になる

class PuddleWorld(World):
    def __init__(self, time_span, time_interval, debug=False):
        super().__init__(time_span, time_interval, debug)
        self.puddles = []
        self.robots = []
        self.goals = []
    
    def append(self, obj):
        self.objects.append(obj)
        if isinstance(obj, Puddle): self.puddles.append(obj)
        if isinstance(obj, Robot): self.robots.append(obj)
        if isinstance(obj, Goal): self.goals.append(obj)
    
    def puddle_depth(self, pose):
        # 水たまりの深さを計算する
        return sum([p.depth * p.inside(pose) for p in self.puddles])
    
    def one_step(self, i, elems, ax):
        super().one_step(i, elems, ax)
        for r in self.robots:
            r.agent.puddle_depth = self.puddle_depth(r.pose) # ロボットがいる位置の水たまりの深さを入れとく
            for g in self.goals:
                if g.inside(r.pose):
                    r.agent.in_goal = True
                    r.agent.final_value = g.value

class PuddleIgnoreAgent(EstimationAgent):
    def __init__(self, time_interval, estimator, goal, puddle_coef=100):
        super().__init__(time_interval, 0.0, 0.0, estimator)

        self.puddle_coef = puddle_coef
        self.puddle_depth = 0.0
        self.total_reward = 0.0
        self.in_goal = False
        self.final_value = 0.0
        self.goal = goal
    
    def reward_per_sec(self):
        return -1.0 - self.puddle_depth * self.puddle_coef # 式(10.15)をΔtで割った値
        # -1.0 は経過時間に対するコストに相当
    
    @classmethod
    def policy(cls, pose, goal):
        x, y, theta = pose
        dx, dy = goal.pos[0] - x, goal.pos[1] - y
        # print("x : {}, y : {}, theta : {} ", x,y,theta)
        direction = int((math.atan2(dy, dx) - theta)*180/math.pi) #ゴールの方角(degreeに直す)
        # print("step 1 : {0}".format(self.direction))
        # math.atan2 は　x軸から測った(原点周り)方位角
        # atan2 - theta は　ロボットが自分から見た方位角
        direction = (direction + 360*1000 + 180) % 360 - 180 #方角を-180~180[deg]に正規化(適当、ロボットが-1000回転すると破綻) 
        # print("step 2 : {0}".format(self.direction))
        # math.atan2(dx, dy) - theta => theta 引いているので値域不明
        # (***)%360 => 0 - 360 
        # (***)%360 -180 => -180 -3 180 
        # e.g., (359 + 360*1000 + 180) % 360 -180 = -1
        # 失敗例 (359 + 360*1000) % 360 -180 = 179

        # 以下制御ロジック
        if direction > 10: 
            nu, omega = 0.0, 2.0
        elif direction < -10:
            nu, omega = 0.0, -2.0
        else:
            nu, omega = 1.0, 0.0
        
        return nu, omega
    
    def decision(self, observation = None):
        if self.in_goal:
            return 0.0, 0.0
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.estimator.observation_update(observation)

        self.total_reward += self.time_interval*self.reward_per_sec()

        nu, omega = self.policy(self.estimator.pose, self.goal)
        self.prev_nu, self.prev_omega = nu, omega
        return nu, omega
    
    def draw(self, ax, elems):
        super().draw(ax, elems)
        x, y, _ = self.estimator.pose
        elems.append(ax.text(x+1.0, y-0.5, "reward/sec:" + str(self.reward_per_sec()), fontsize=8))
        elems.append(ax.text(x+1.0, y-1.0, "eval:{:.1f}".format(self.total_reward + self.final_value), fontsize=8))

def trial():
    time_interval = 0.1
    world = PuddleWorld(30, time_interval, debug=False)

    ### 地図を生成して3つランドマークを追加 ###
    m = Map()
    for ln in [(-4, 2), (2, -3), (4, 4), (-4, -4)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    ## ゴールの追加
    goal = Goal(-3,-3)
    world.append(goal)

    ## 水たまりの追加
    world.append(Puddle((-2,0),(0,2),0.1))
    world.append(Puddle((-0.5,-2),(2.5,1),0.1))


    ### ロボットを作る ###
    initial_pose = np.array([2, 2, 0]).T
    kf = KalmanFilter(m, initial_pose)
    circling = PuddleIgnoreAgent(time_interval, kf, goal)
    r = Robot(initial_pose, sensor=Camera(m, distance_bias_rate_stddev=0, direction_bias_stddev=0), agent=circling, color="red", bias_rate_stds=(0,0))
    world.append(r)

    world.draw()

if __name__ == '__main__':
    trial()