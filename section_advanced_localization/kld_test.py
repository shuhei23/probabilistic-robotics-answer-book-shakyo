# KLD(Kullback-Leibler divergence)

import sys
sys.path.append('../scripts/')
from robot import *
from scipy.stats import norm as nd# nd:ガウス分布
from scipy.stats import chi2 # chi2:カイ二乗分布

def num(epsilon, delta, binnum):
    # 真の信念分布を表現できる最小のパーティクル数Nを計算する
    return math.ceil(chi2.ppf(1.0 -delta, binnum -1)/(2*epsilon))

fig, (axl, axr) = plt.subplots(ncols=2, figsize=(10,4)) # 図を横並び2つで出力する準備

bs = np.arange(2,10)
n = [num(0.1, 0.001, b) for b in bs] # ビン2～10までのパーティクル数
axl.set_title("bin:2-10")
axl.plot(bs, n)

bs = np.arange(2,100000)
n = [num(0.1, 0.001, b) for b in bs] # ビン2～10までのパーティクル数
axr.set_title("bin:2-100000")
axr.plot(bs, n)

plt.show()


def num_wh(epsilon, delta, binnum):
    dof = binnum -1
    z = nd.ppf(1.0 - delta)
    return math.ceil(dof/(2*epsilon)*(1.0 - 2.0/(9*dof) + math.sqrt(2.0/(9*dof))*z)**3)
                     
for binnum in 2,4,8,1000,10000,100000:
    print("ビン:",binnum, "ε=0.1, δ=0.01", num(0.1, 0.01, binnum), num_wh(0.1, 0.01, binnum))
    print("ビン:",binnum, "ε=0.5, δ=0.01", num(0.1, 0.01, binnum), num_wh(0.1, 0.01, binnum))
    print("ビン:",binnum, "ε=0.5, δ=0.05", num(0.1, 0.01, binnum), num_wh(0.1, 0.01, binnum))