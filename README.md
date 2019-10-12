# AdvancedDataAnalysis
## 適応正則化学習
二乗ヒンジ損失に基づく適応正則化分類を線形モデル

<div align="center">
<img src="formula/lec8-1.png" width="200">
</div>

に対して適用する．

### 結果
<div align="center">
<img src="output/Lec8/result.png" width="500">
</div>
異常値に対してロバストな結果が得られている．


## 半教師付き学習
ガウスカーネルモデルに対してラプラス正則化最小二乗分類を実装する．

<div align="center">
<img src="formula/lec9-1.png" width="600">
</div>

によってパラメータを決める．

### 結果
<div align="center">
<img src="output/Lec9/result.png" width="500">
</div>

データのなす領域に沿った識別面が得られている．

## 転移学習
線形モデルに対してクラス比重み付き最小二乗法を実装する．

