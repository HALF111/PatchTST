# Experiments on exploring model architecture

cd PatchTST_supervised/
sh scripts/all_models/etth1.sh
sh scripts/all_models/etth2.sh
......

目前all_models有4个脚本，剩余的数据集的脚本可以参考Autoformer/FEDformer中的scripts来完成。


需要跑的模型目前包括自回归/非自回归的encoder-decoder（即Transformer_patch），以及自回归/非自回归的Decoder，共计4个。

后续需要补完整Masked_encoder和fixed_decoder等模型。


注意1：patch_size和stride必须一样大，也即是非重叠的。

注意2：输入长度seq_len和预测长度pred_len必须都是patch_size的倍数，否则不完全的切割可能会出问题。
