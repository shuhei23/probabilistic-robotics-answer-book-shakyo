import sys
sys.path.append('../scripts/')
from puddle_world import *
import itertools

class PolicyEvaluator:
    def __init__(self, widths, goal, lowerleft=np.array([-4,-4]).T, upperright=np.array([4,4]).T):
        self.pose_min = np.r_[lowerleft,0]
        self.pose_max = np.r_[upperright,math.pi*2] # thetaの範囲を0 - 2πで固定
        self.widths = widths
        self.goal = goal

        self.index_nums = ((self.pose_max - self.pose_min)/self.widths).astype(int)
        nx, ny, nt = self.index_nums
        self.indexes = list(itertools.product(range(nx), range(ny), range(nt))) # 全部のインデックスの組み合わせを作成

        self.value_function, self.final_state_flags = self.init_value_function()

    def init_value_function(self):
        v = np.empty(self.index_nums)
        f = np.zeros(self.index_nums)

        for index in self.indexes:
            f[index] = self.final_state(np.array(index).T)
            v[index] = self.goal.value if f[index] else -100.0

        return v,f

    def final_state(self,index):
        x_min, y_min, _ = self.pose_min + self.widths*index         # xy平面の左下の座標
        x_max, y_max, _ = self.pose_min + self.widths*(index + 1)   # 右上の座標

        corners = [[x_min, y_min, _], [x_min, y_max, _], [x_max, y_min, _], [x_max, y_max, _]]
        return all([self.goal.inside(np.array(c).T) for c in corners])

def trial():
    pe = PolicyEvaluator(np.array([0.2, 0.2, math.pi/18]).T, Goal(-3, -3)) # PolicyEvaluatorのオブジェクトを生成
    
    import seaborn as sns

    v = pe.value_function[:,:,0]
    sns.heatmap(np.rot90(v), square=False)
    plt.show()

    f = pe.final_state_flags[:,:,0]
    sns.heatmap(np.rot90(f), square=False)
    plt.show()
    
    # print(pe.index_nums)
    
    # 例題: policy_ evaluation1.ipynb-[4]
    # print(np.floor((pose-pe.pose_min)/pe.widths).astype(int))
    
    # pose = np.array([2.9,-2,math.pi]).T
    # print(np.floor((pose-pe.pose_min)/pe.widths).astype(int))
    
    # pose = np.array([-5,-2,math.pi/6]).T
    # print(np.floor((pose-pe.pose_min)/pe.widths).astype(int))
    
    
    

if __name__ == '__main__':
    trial()