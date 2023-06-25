import sys
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
        
def draw(xs, zlist):
    ax = make_ax()
    draw_observations(xs, zlist, ax)
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

hat_xs, zlist = read_data()
draw(hat_xs, zlist)

