# 阵列频谱前端起点

Python 3.11 环境中安装 `requirements.txt` 后，运行 `python demo.py`。演示处理一段连续频谱，随后使用正定噪声协方差计算单方向 MVDR 权重。

现有接口：

- `dereverb.single_wpe(spectrum, taps=2, delay=2, iterations=3)`：输入 `F×C×T`，返回相同尺寸的连续录音 WPE 残差。每频点单独估计功率下界。
- `spatial.mvdr(noise_psd, steering)`：一个频点的正定 `C×C` 协方差与长度 C 的 steering，返回满足 `steeringᴴ weights=1` 的权重。
- `spatial.weighted_psd(segments, frame_weights)`：各段 `F×C×T_s` 残差与 `F×T_s` 非负帧权重，跨全部段、全部帧归一化求PSD；某频点总权重为0时返回零矩阵。
- `spatial.apply_weights(weights, spectrum)`：将 `F×C` 权重的共轭作用于 `F×C×T` 频谱，得到 `F×T`。

`fixtures/cases.json` 描述输入数组。每个WPE NPZ的 `segment_i` 是复频谱，`noise_weight_i` 是用于PSD的外部非负帧权重，`steering` 是各频点的方向向量。空间输入集中在 `fixtures/spatial.npz`，清单给出每组 R/C/g 的数组键。NPZ只含原始输入，不含求解结果。

`vendor/nara_wpe_subset.py` 来自固定 NARA-WPE commit `5fbfc3639274e5a70dffcf08e20615a38c372ebb`；所选函数体保持上游原样，仅省略CLI及无关实现。原路径、源文件摘要写在文件头，MIT许可见同目录 `LICENSE.nara_wpe`。本工程不依赖完整NARA-WPE、TorchAudio或房间模拟包，不进行STFT、音频设备访问或网络下载。
