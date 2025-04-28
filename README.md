# [Self-supervised Reflective Learning through Self-distillation and Online Clustering for Speaker Representation Learning](https://ieeexplore.ieee.org/document/10938970)

### Stage1: DINO

#### Train

```
nohup python train_dino_mc_ali.py  > dino_mc_ali_clodstart.log
```

#### Inference

```
python infer_dino.py --save_dir save_dir --val_data_name vox_test --val_save_name test  --model_num 0  --onlyscore False --scoring True --gpu 0 &
```

#### Kmeans clustering to obtain pseudo labels

```
python cluster.py
```

### Stage2: SSRL

#### Train

```
nohup python train_lnc_nlm_dino_ali.py --save_dir dino_ali_lnc_nlm_8k_aam --gpu 2,3,4,5,6,7 --port 8441 --arcface True >> dino_ali_lnc_nlm_8k_aam.log & 
```

#### Inference

```
python infer_dino.py --model_id ema_teacher --save_dir save_dir --model_num 0 --val_data_name vox_test --val_save_name test --onlyscore False --scoring True --gpu 0 &
```

### Integrating WavLM into the SSRL Framework

#### Download

Download the [WAVLM-Large model](https://github.com/microsoft/unilm/tree/master/wavlm) and place it under the **wavlm** folder.

#### Train

```
# step1
nohup python train_lnc_nlm_dino_ali_wavlm_step1.py --save_dir dino_ali_lnc_nlm_8k_wavlm_layer24_step1 --layer_nums 24  --gpu 2,3,4,5,6,7 --port 8443 --arcface True  >> dino_ali_lnc_nlm_8k_wavlm_layer24_step1.log & 
# step2
nohup python train_lnc_nlm_dino_ali_wavlm_step2.py --save_dir dino_ali_lnc_nlm_8k_wavlm_layer24_step2  --layer_nums 24  --gpu 2,3,4,5,6,7 --port 8444 --arcface True  >> dino_ali_lnc_nlm_8k_wavlm_layer24_step2.log & 
# step2 fine-tuning (Unfreeze the wavlm module)
nohup python train_lnc_nlm_dino_ali_wavlm_step2_ft.py --save_dir dino_ali_lnc_nlm_8k_wavlm_layer24_step2_ft  --layer_nums 24 --gpu 2,3,4,5,6,7 --port 8445 --arcface True --start_epoch 0 --freeze_ptm False >> dino_ali_lnc_nlm_8k_wavlm_layer24_step2_ft.log & 
```

#### Inference

```
python infer_wavlm.py --model_id ema_teacher --save_dir save_dir --model_num 0 --val_data_name vox_test --val_save_name test --onlyscore False --scoring True --gpu 0 &
```

## Citations

```
@ARTICLE{10938970,
  author={Cai, Danwei and Cai, Zexin and Li, Ze and Li, Ming},
  journal={IEEE Transactions on Audio, Speech and Language Processing}, 
  title={Self-Supervised Reflective Learning Through Self-Distillation and Online Clustering for Speaker Representation Learning}, 
  year={2025},
  volume={33},
  number={},
  pages={1535-1550},
  keywords={Training;Representation learning;Iterative methods;Noise;Adaptation models;Noise measurement;Predictive models;Speech processing;Robustness;Convergence;Knowledge distillation;noisy label modeling;self-labeling;self-supervised learning;speaker recognition},
  doi={10.1109/TASLPRO.2025.3555132}}
```

