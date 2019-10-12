# AdvancedDataAnalysis
## 適応正則化学習
二乗ヒンジ損失に基づく適応正則化分類を線形モデル

<div align="center">
<img src="formula/lec8-1.png" width="200">
</div>

に対して適用する．[（実装）](src/Lec8/adaptive.py)

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

によってパラメータを決める．[（実装）](src/Lec9/lrls.py)

### 結果
<div align="center">
<img src="output/Lec9/result.png" width="500">
</div>

データのなす領域に沿った識別面が得られている．

## 転移学習
線形モデルに対してクラス比重み付き最小二乗法を実装する．訓練データとテストデータでクラスを構成する点の比率が異なるデータを用いた．[（実装）](src/Lec9/cwls.py)

### 結果
左が訓練データ，右がテストデータ．

##### 重みなしの最小二乗法による分類結果
<div align="center">
<img src="output/Lec9/result_no_weight_train.png" width="330">
<img src="output/Lec9/result_no_weight_test.png" width="330">
</div>

##### 重み付き最小二乗法による分類結果
<div align="center">
<img src="output/Lec9/result_cwls_train.png" width="330">
<img src="output/Lec9/result_cwls_test.png" width="330">
</div>

重みなしのときはテストデータが正しく分類できていなかったが，重み付き最小二乗法によって，テストデータも正しく分類する識別面が得られていることが確認できる．

## 局所性保存射影による次元削減
類似度行列を

<div align="center">
<img src="formula/lec10-1.png" width="350">
</div>

として実装した．[（実装）](src/Lec10/lpp.py)

### 結果
<div align="center">
<img src="output/Lec10/result.png" width=500>
</div>

データのクラスタ構造を保持したまま射影することができている．
