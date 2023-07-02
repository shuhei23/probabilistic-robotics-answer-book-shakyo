import sys
import itertools
sys.path.append('../scripts/')
from kf import * #誤差楕円を描くのに利用

def make_ax(): #axisの準備
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    return ax

def draw_trajectory(xs, ax): #軌跡の描画
    poses = [xs[s] for s in range(len(xs))]
    ax.scatter([e[0] for e in poses], [e[1] for e in poses], s=5, marker=".", color="black")
    ax.plot([e[0] for e in poses], [e[1] for e in poses], linewidth=0.5, color="black")
    
def draw_observations(xs, zlist, ax): #センサ値の描画
    for s in range(len(xs)):
        if s not in zlist:
            continue
        
        for obs in zlist[s]:
            x, y, theta = xs[s]
            ell, phi = obs[1][0], obs[1][1]
            mx = x + ell*math.cos(theta+phi)
            my = y + ell*math.sin(theta+phi)
            ax.plot([x,mx], [y,my], color="pink", alpha=0.5)

def draw_edges(edges, ax):
    for e in edges:
        ax.plot([e.x1[0], e.x2[0]], [e.x1[1], e.x2[1]], color="red", alpha=0.5)

def draw(xs, zlist, edges):
    ax = make_ax()
    draw_observations(xs, zlist, ax)
    draw_edges(edges,ax)
    draw_trajectory(xs, ax)
    plt.show()

def read_data(): #データの読み込み
    hat_xs={} #軌跡のデータ(ステップ数をキーにして姿勢を保存)
    zlist={} #センサ値のデータ(ステップ数をキーにして、さらにその中にランドマークのIDとセンサ値をタプルで保存)
    
    with open("log.txt") as f:
        # log.txt format は2パターン
        # "x", step, x, y, theta
        # "z", step, id, x, y, theta
        for line in f.readlines():
            tmp = line.rstrip().split() # 右端のスペースを削除
            
            step = int(tmp[1])
            if tmp[0] == "x": #姿勢のレコードの場合
                hat_xs[step] = np.array([float(tmp[2]), float(tmp[3]), float(tmp[4])]).T
            elif tmp[0] == "z": #センサ値のレコードの場合
                if step not in zlist: #ランドマークを発見していない時は空を入れておく
                    zlist[step] = []
                zlist[step].append((int(tmp[2]), np.array([float(tmp[3]), float(tmp[4]), float(tmp[5])]).T))
    return hat_xs, zlist


class ObsEdge:
    def __init__(self, t1 , t2, z1, z2, xs, sensor_noise_rate=[0.14,0.05,0.05]): # sensor_noise_rate追加 
        # xs = [step, (x, y, z)]
        # z1 = [id, (ell, phi, psi)]
        assert z1[0] == z2[0] # ランドマークのIDが違ったら処理を止める
        
        self.t1, self.t2 = t1, t2           # 時刻の記録
        self.x1, self.x2 = xs[t1], xs[t2]   # 各時刻の姿勢 slef.x = [x,y,theta]
        self.z1, self.z2 = z1[1], z2[1]     # 各時刻のセンサ値 self.z = [ell, phi, psi]

        s1 = math.sin(self.x1[2] + self.z1[1]) # sin(theta_1 + phi_1)
        c1 = math.cos(self.x1[2] + self.z1[1]) 
        s2 = math.sin(self.x2[2] + self.z2[1])
        c2 = math.cos(self.x2[2] + self.z2[1])

        ### 誤差の計算 ###
        hat_e = self.x2 - self.x1 + np.array([
            self.z2[0]*c2 - self.z1[0]*c1, 
            self.z2[0]*s2 - self.z1[0]*s1, 
            self.z2[1]- self.z2[2] - self.z1[1] + self.z1[2] ])
        
        # -pi <= hat_e[2] < +pi になるように2*pi=360度を足し引き
        while hat_e[2] >= math.pi: hat_e -= math.pi*2
        while hat_e[2] < -math.pi: hat_e += math.pi*2
        
        print(hat_e)
        ## 精度行列の作成 ##
        # y = Ax + Bz のとき，分散共分散行列は A@Sigma_A@A' + B@Sigma_B@B'
        # e_{j,t_1,t_2} = (線形近似) = R_{j,t_1}@z_{j_a} + R_{j,t_2}z_{j_b}
        # となるので，e_{j,t_1,t_2}の分散共分散行列は
        # R_{j,t_1}@Q_{j,t_1}@R_{j,t_1}' + R_{j,t_2}@Q_{j,t_2}@R_{j,t_2}'

        Q1 = np.diag([(self.z1[0] * sensor_noise_rate[0])**2, sensor_noise_rate[1]**2,sensor_noise_rate[2]**2]) # 式(9.29)
        R1 = -np.array([[c1, -self.z1[0]*s1, 0],
                        [s1,  self.z1[0]*c1, 0], 
                        [0,   1,             -1]]) # 式(9.31)

        Q2 = np.diag([(self.z2[0] * sensor_noise_rate[0])**2, sensor_noise_rate[1]**2,sensor_noise_rate[2]**2])
        R2 = -np.array([[c2, -self.z2[0]*s2, 0],
                        [s2,  self.z2[0]*c2, 0], 
                        [0,   1,             -1]]) # 式(9.32)
        
        Sigma = R1.dot(Q1).dot(R1.T) + R2.dot(Q2).dot(R2.T) # 式(9.34)
        self.Omega = np.linalg.inv(Sigma)

        print(Sigma)

def make_edges(hat_xs, zlist):
    landmark_key_zlist = {} # ランドマークのIDをキーにして観測された時刻とセンサ値を記録

    for step in zlist:      # キーを時刻からランドマークのIDへ
        for z in zlist[step]:
            landmark_id = z[0]
            if landmark_id not in landmark_key_zlist:
                landmark_key_zlist[landmark_id] = []

            landmark_key_zlist[landmark_id].append((step,z)) #タプルを追加

    edges = []
    for landmark_id in landmark_key_zlist:
        step_pairs = list(itertools.combinations(landmark_key_zlist[landmark_id],2)) # 時刻のペアを作成
        # ランドマークが j = landmark_id のとき， z_{j,1}, z_{j,5}, z_{j,8} -> 1-5, 1-8, 5-8 全部の組を使うらしい
        edges += [ObsEdge(xz1[0], xz2[0], xz1[1], xz2[1], hat_xs) for xz1, xz2 in step_pairs]

    return edges

hat_xs, zlist = read_data()
edges = make_edges(hat_xs, zlist)
draw(hat_xs,zlist,edges)