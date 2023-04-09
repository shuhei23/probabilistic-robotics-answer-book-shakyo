# KLD(Kullback-Leibler divergence)

import sys
sys.path.append('../scripts/')
from robot import *
from scipy.stats import norm, chi2 # norm:ガウス分布、chi2:カイ二乗分布

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