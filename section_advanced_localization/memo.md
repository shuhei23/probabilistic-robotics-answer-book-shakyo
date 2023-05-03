# 詳解確率ロボディクス 7.1.2 (p.156の箇所の計算について)

$P[2\log\lambda_N \leq y] = \int_0^y \chi_{k-1}^2(x)\,\mathrm{d}x = 1 - \delta$ となる $y$ をさがす．
(カイ2乗分布が登場するのは $2\log\lambda_N = \chi_{k-1}^2$ の確率密度関数)
$P[2\log\lambda_N \leq y] < P[2\log\lambda_N \leq 2N\epsilon]$ になればよいので， $y < 2N\epsilon$ となるように$N$をえらぶ．
($P[* \leq y] < P[ * \leq 2N\epsilon]$ は $y < 2N\epsilon$ だから)
すると $1-\delta = P[2\log\lambda_N \leq y] < P[2\log\lambda_N \leq 2N\epsilon]$ になっている．