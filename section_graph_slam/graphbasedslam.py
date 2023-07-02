import sys
import itertools
sys.path.append('../scripts/')
from kf import * #誤差楕円を描くのに利用


class ObsEdge:
    def __init__(self, t1 , t2, z1, z2, xs, sensor_noise_rate=[0.14,0.05,0.05]): # sensor_noise_rate追加 
        # xs = [step, (x, y, z)]
        # z1 = [id, (ell, phi, psi)]
        assert z1[0] == z2[0] # ランドマークのIDが違ったら処理を止める
        
        self.t1, self.t2 = t1, t2           # 時刻の記録
        self.x1, self.x2 = xs[t1], xs[t2]   # 各時刻の姿勢 slef.x = [x,y,theta]
        self.z1, self.z2 = z1[1], z2[1]     # 各時刻のセンサ値 self.z = [ell, phi, psi]

        s1 = math.sin(self.x1[2] + self.z1[1]) # sin(theta_1 + phi_1)
        c1 = math.cos(self.x1[2] + self.z1[1]) # cos(theta_1 + phi_1)
        s2 = math.sin(self.x2[2] + self.z2[1]) # sin(theta_2 + phi_2)
        c2 = math.cos(self.x2[2] + self.z2[1]) # cos(theta_2 + phi_2)

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
        Omega = np.linalg.inv(Sigma)

        B1 = - np.array([[1,0,-self.z1[0]*s1],
                         [0,1,self.z1[0]*c1],
                         [0,0,          1]])

        B2 = np.array([[1,0,-self.z2[0]*s2],
                       [0,1, self.z2[0]*c2],
                       [0,0,             1]])

        # 式(9.43)
        self.omega_upperleft  = B1.T.dot(Omega).dot(B1)
        self.omega_upperright = B1.T.dot(Omega).dot(B2)
        self.omega_bottomleft = B2.T.dot(Omega).dot(B1)
        self.omega_bottomright = B2.T.dot(Omega).dot(B2)
        
        # 式(9.44)
        self.xi_upper = - B1.T.dot(Omega).dot(hat_e)
        self.xi_bottom = -B2.T.dot(Omega).dot(hat_e)

### Method ###
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

def Draw(xs, zlist, edges):
    ax = make_ax()
    draw_observations(xs, zlist, ax)
    # draw_edges(edges,ax)
    draw_trajectory(xs, ax)
    plt.show()

def ReadData(): #データの読み込み
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

def AddEdge(edge,Omega,xi):
    f1, f2 = edge.t1*3, edge.t2*3
    t1,t2 = f1+3, f2+3
    Omega[f1:t1, f1:t1] += edge.omega_upperleft
    Omega[f1:t1, f2:t2] += edge.omega_upperright
    Omega[f2:t2, f1:t1] += edge.omega_bottomleft
    Omega[f2:t2, f2:t2] += edge.omega_bottomright
    xi[f1:t1] += edge.xi_upper
    xi[f2:t2] += edge.xi_bottom

def MakeEdges(hat_xs, zlist):
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


if __name__ == '__main__':
    hat_xs, zlist = ReadData()
    dim = len(hat_xs)*3 # 軌跡をつなげたベクトルの次元
    
    for n in range(1, 10000): # 繰り返しの回数は適当に大きい値
        ### エッジ、大きな精度行列、係数ベクトルの作成
        edges = MakeEdges(hat_xs, zlist)
        Omega = np.zeros((dim, dim))
        xi = np.zeros(dim)
        Omega[0:3,0:3] += np.eye(3)*1000000 # x0の固定
        
        ## 軌跡を動かす量（差分）の計算
        for e in edges:
            AddEdge(e, Omega, xi) # エッジの精度行列、係数ベクトルをOmega, xiに足す
    
        delta_xs = np.linalg.inv(Omega).dot(xi)
    
        # 推定値の更新
        for i in range(len(hat_xs)):
            hat_xs[i] += delta_xs[i*3:(i+1)*3]
    
        # 終了判定
        diff = np.linalg.norm(delta_xs)
        print("{}回目の繰り返し:{}".format(n,diff))
        if diff < 0.01:
            Draw(hat_xs,zlist,edges)
            break