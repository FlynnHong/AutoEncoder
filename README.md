### Pytorch-Autoencoder-

# This model can use resample sound data.

Data file shape : .wav (10s)



#System version
 - Python : 3.8+ 
 - conda : 4.9.1
 - Spyder : 4.1.5
 - Pytorch : 1.7.1
 - Numpy : 1.19.2
 - Librosa : 0.8.0
 - Cuda 11.2 (if you use) 
 
 model needs Librosa to upload .wav data
 
 How to install Librosa:
 
     Go to conda powershell prompt
 
     >>> pip install librosa
 
 
 when you write notebook, import librosa 
 
 
 
 # Difficulties:
     
 Sound data is so large.. 
 Recommend to put in less data.
     
 # Dataset
 https://zenodo.org/record/3678171#.YA1urMht9PY
 
 I used dev_data_fan.
 
 # Result:
 
    runfile('C:/train1/autotrain.py', wdir='C:/train1')
    0%|          | 0/50 [00:00<?, ?it/s]AutoEncoder(
    (encoder): Linear(in_features=20000, out_features=3000, bias=True)
    (hidden1): Linear(in_features=3000, out_features=1000, bias=True)
    (hidden2): Linear(in_features=1000, out_features=3000, bias=True)
    (decoder): Linear(in_features=3000, out_features=20000, bias=True)
    )
    0.010361242949875305
    ok
    2%|▏         | 1/50 [00:06<05:06,  6.25s/it]0.000906443891130948
    ok
    6%|▌         | 3/50 [00:16<04:30,  5.75s/it]5.811382439792942
    8%|▊         | 4/50 [00:21<04:12,  5.48s/it]0.030767276865663006
    10%|█         | 5/50 [00:26<03:58,  5.30s/it]0.06770632123952965
    12%|█▏        | 6/50 [00:31<03:47,  5.16s/it]0.017730689811287448
    14%|█▍        | 7/50 [00:36<03:38,  5.09s/it]0.0012838360911700875
    0.0007908386833150871
    ok
    16%|█▌        | 8/50 [00:42<03:43,  5.33s/it]0.0006815471759182401
    ok
    18%|█▊        | 9/50 [00:48<03:46,  5.52s/it]0.0006501009787825751
    ok
    20%|██        | 10/50 [00:54<03:46,  5.66s/it]0.0006216851081262576
    ok
    24%|██▍       | 12/50 [01:05<03:29,  5.51s/it]0.0006225834309589118
    26%|██▌       | 13/50 [01:10<03:16,  5.31s/it]0.0006251688753764028
    28%|██▊       | 14/50 [01:14<03:06,  5.17s/it]0.0006546306378368172
    30%|███       | 15/50 [01:19<02:58,  5.10s/it]0.0006850740554000367
    .
    .
    .
    .
    .
    90%|█████████ | 45/50 [03:46<00:24,  4.98s/it]0.008837707494967617
    92%|█████████▏| 46/50 [03:51<00:19,  4.95s/it]0.009094579829252325
    94%|█████████▍| 47/50 [03:56<00:14,  4.93s/it]0.008899411937454716
    96%|█████████▌| 48/50 [04:00<00:09,  4.93s/it]0.008034881012281402
    98%|█████████▊| 49/50 [04:05<00:04,  4.93s/it]0.008866294578183442
    100%|██████████| 50/50 [04:10<00:00,  5.02s/it]0.011111363882082514

# if you want to save, go to console window

    >>> t1 = np.array(list(output))
    >>> np.save('address', output)


