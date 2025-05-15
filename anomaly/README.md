# Anomaly Detection Algorithms
## 1. Datasets
Below are all datasets from ``ADBench``, we split them to **two** groups:
- ``ad47``: all below datasets
- ``ad39``: all below datasets ***except*** image datasets (image datasets: 1, 8, 17, 20, 24, 26, 28, 33)

***IMPORTANT*** : normal data are labeled **0**, abnormal data are labeled **1**

| Number |        Data       | # Samples | # Features | # Anomaly | % Anomaly |   Category   |
|--------|-------------------|-----------|------------|-----------|-----------|--------------|
|    1   | ad_ALOI              |   49534   |     27     |   1508    |   3.04    |    Image     |
|    2   | ad_annthyroid        |   7200    |     6      |    534    |   7.42    |   Healthcare |
|    3   | ad_backdoor          |   95329   |    196     |   2329    |   2.44    |   Network    |
|    4   | ad_breastw           |    683    |     9      |    239    |   34.99   | Healthcare   |
|    5   | ad_campaign           |   41188   |     62     |   4640    |   11.27   | Finance      |
|    6   | ad_cardio            |   1831    |     21     |    176    |   9.61    | Healthcare   |        
|    7   | ad_Cardiotocography  |   2114    |     21     |    466    |   22.04   | Healthcare   |
|    8   | ad_celeba             |  202599   |     39     |   4547    |   2.24    |    Image     |
|    9   | ad_census             |  299285   |    500     |   18568   |   6.20    | Sociology    |
|   10   | ad_cover             |  286048   |     10     |   2747    |   0.96    | Botany       | 
|   11   | ad_donors             |  619326   |     10     |   36710   |   5.93    | Sociology    |
|   12   | ad_fault             |   1941    |     27     |    673    |   34.67   | Physical     |
|   13   | ad_fraud              |  284807   |     29     |    492    |   0.17    | Finance      |
|   14   | ad_glass             |    214    |     7      |     9     |   4.21    | Forensic     |
|   15   | ad_Hepatitis         |    80     |     19     |    13     |   16.25   | Healthcare   |
|   16   | ad_http              |  567498   |     3      |   2211    |   0.39    | Web          |      
|   17   | ad_InternetAds       |   1966    |    1555    |    368    |   18.72   | Image        |
|   18   | ad_Ionosphere        |    351    |     32     |    126    |   35.90   | Oryctognosy  |
|   19   | ad_landsat           |   6435    |     36     |   1333    |   20.71   | Astronautics |     
|   20   | ad_letter            |   1600    |     32     |    100    |   6.25    | Image        |    
|   21   | ad_Lymphography      |    148    |     18     |     6     |   4.05    | Healthcare   |  
|   22   | ad_magic.gamma       |   19020   |     10     |   6688    |   35.16   | Physical     | 
|   23   | ad_mammography       |   11183   |     6      |    260    |   2.32    | Healthcare   |       
|   24   | ad_mnist             |   7603    |    100     |    700    |   9.21    | Image        |   
|   25   | ad_musk              |   3062    |    166     |    97     |   3.17    | Chemistry    |      
|   26   | ad_optdigits         |   5216    |     64     |    150    |   2.88    | Image        |    
|   27   | ad_PageBlocks        |   5393    |     10     |    510    |   9.46    | Document     |
|   28   | ad_pendigits         |   6870    |     16     |    156    |   2.27    | Image        | 
|   29   | ad_Pima              |    768    |     8      |    268    |   34.90   | Healthcare   |
|   30   | ad_satellite         |   6435    |     36     |   2036    |   31.64   | Astronautics |    
|   31   | ad_satimage-2        |   5803    |     36     |    71     |   1.22    | Astronautics |     
|   32   | ad_shuttle           |   49097   |     9      |   3511    |   7.15    | Astronautics |       
|   33   | ad_skin              |  245057   |     3      |   50859   |   20.75   |    Image     |
|   34   | ad_smtp              |   95156   |     3      |    30     |   0.03    | Web          | 
|   35   | ad_SpamBase          |   4207    |     57     |   1679    |   39.91   | Document     |
|   36   | ad_speech            |   3686    |    400     |    61     |   1.65    | Linguistics  |     
|   37   | ad_Stamps            |    340    |     9      |    31     |   9.12    | Document     |
|   38   | ad_thyroid           |   3772    |     6      |    93     |   2.47    | Healthcare   |   
|   39   | ad_vertebral         |    240    |     6      |    30     |   12.50   | Biology      |  
|   40   | ad_vowels            |   1456    |     12     |    50     |   3.43    | Linguistics  |       
|   41   | ad_Waveform          |   3443    |     21     |    100    |   2.90    | Physics      |
|   42   | ad_WBC               |    223    |     9      |    10     |   4.48    | Healthcare   |
|   43   | ad_WDBC              |    367    |     30     |    10     |   2.72    | Healthcare   |
|   44   | ad_Wilt              |   4819    |     5      |    257    |   5.33    | Botany       |
|   45   | ad_wine              |    129    |     13     |    10     |   7.75    | Chemistry    |      
|   46   | ad_WPBC              |    198    |     33     |    47     |   23.74   | Healthcare   |      
|   47   | ad_yeast             |   1484    |     8      |    507    |   34.16   | Biology      |

# 2. Baseline Performance
***Running speed note***: `pyod_loci` and `pyod_lscp` are super slow ,`pyod_rod` is super low on large datasets.

***Error Troubleshoot***: 
- NaN error in `pyod_abod`, increase `n_nieghbors`
- RuntimeWarning and None performance in `pyod_sos`, increase `eps` or `perplexity`
- Warning in `pyod_cof`, increase `n_neighbors`
- **Warning: VAE is very sensitive to hyperparamter choice**, NaN error in `pyod_vae`, decrease `lr`, decrease `dropout_rate`, increase `hidden_dim`, set `preprocess`=False, or set `batch_norm`=True
- inf error in `pyod_pca`, check if n_components is null (the default value by PyOD), change to a float
- ValueError in `pyod_sod`, make sure `n_neighbors` > 10

The N/A in the Baseline column means ADBench didn't release the result for the algorithm.

`pyod_mad` is only for univariate data, so we can't run this on our data.

## 2.1 Performance on ``ad47``

| Unsupervised Algorithm | First | Second | Baseline|
| ---------------------- | -------- | --------| ------- | 
|       pyod_lof         | 64.6028 | 68.3400 | 61.4964 |
|     pyod_iforest       | 76.3668  | 76.7680 | 76.1657 |
|      pyod_ocsvm        | 71.5165 | 72.3310 | 69.9821 |
|       pyod_abod        | nan | nan | N/A |
|      pyod_cblof        | 74.5527 | 76.4313 | 74.4719 |
|       pyod_cof         | 65.7590 | 67.0852 | 62.7523 |
|      pyod_copod        | 74.6820 | 74.6820 | 74.3921 |
|       pyod_ecod        | 74.4502 | 74.4502 | 73.9472 |
|  pyod_feature_bagging  | 63.7865 | 66.0679 | N/A |
|       pyod_hbos        | 74.5371 | 76.3578 | 74.037 |
|       pyod_knn         | 77.0997 | 78.3994 | 69.8653 |
|      pyod_lmdd         | slow | slow | N/A |
|      pyod_loda         | 67.9641 | 72.8237 | 64.9287 |
|      pyod_loci         | s-slow | s-slow | N/A |
|      pyod_lscp         | s-slow | s-slow | N/A |
|       pyod_mad         | N/A | N/A | N/A |
|       pyod_mcd         | 76.7159 | 76.7159 | N/A |
|       pyod_pca         | 74.4616 | 75.4084 | 73.5964 |
|       pyod_rod         | slow | slow | N/A |
|       pyod_sod         | 69.7714 | 70.5317 | 68.7268 |
|       pyod_sos         | 74.7392 | 74.8902 | N/A |
|    pyod_autoencoder    | 74.431 | 78.4636 | N/A |
|       pyod_vae         | 72.5557 | 72.5557 | N/A |
|     pyod_so_gaal       | 38.8355 | 46.608 | N/A |
|     pyod_mo_gaal       | 33.5397 | 39.4509 | N/A |
|     pyod_deepsvdd      | 73.3760 | 76.3629 | 53.6115 |
|         DAGMM          |  |  | 64.5100 |


| (semi-)Supervised Algorithm | First | Second | Baseline la=0.01|
| -------------------- | ------- | --------| ------- | 
|      pyod_xgbod      | slow | slow | 80.0309 |
|       GANomaly       |  |  | 67.803 |
|        DeepSAD       |  |  | 75.2543 |
|         REPEN        |  |  | 77.2047 |
|        DevNet        |  |  | 79.0513 |
|        PReNet        |  |  | 79.0447 |
|        FEAWAD        |  |  | 73.9274 |
|          NB          |  |  | 60.604 |
|         SVM          |  |  | 60.6964 |
|         MLP          |  |  | 59.0104 |
|       ResNet         |  |  | 64.6724 |
|    FTTransformer     |  |  | 77.7261 |
|          RF          |  |  | 61.3826 |
|         LGB          |  |  | 61.9985 |
|         XGB          |  |  | 78.1832 |
|        CatB          |  |  | 83.3683 |


## 2.2 Performance on ``ad39``

| Unsupervised Algorithm | First | Second | Baseline|
| ---------------------- | -------- | --------| ------- | 
|       pyod_lof         | 65.9391  | 69.3328 | 61.9779 |
|     pyod_iforest       | 77.6082 |  78.009 | 77.1223 |
|      pyod_ocsvm        | 72.8164 | 73.4035 | 70.9743 |
|       pyod_abod        |  |  | N/A |
|      pyod_cblof        | 75.2095 | 76.7696 | 74.3056 |
|       pyod_cof         | 68.0258 | 70.6575 | 63.9903 |
|      pyod_copod        | 76.4842 | 76.4842 | 75.9217 |
|       pyod_ecod        | 76.2974 | 76.2974 | 75.6997 |
|  pyod_feature_bagging  | 64.9517 | 67.0165 | N/A |
|       pyod_hbos        | 75.8585 | 77.7208 | 75.0749 |
|       pyod_knn         | 77.7531 | 78.6164 | 70.2479 |
|      pyod_lmdd         | slow | slow | N/A |
|      pyod_loda         | 69.2386 | 74.4398 | 65.7879 |
|      pyod_loci         | s-slow | s-slow | N/A |
|      pyod_lscp         | s-slow | s-slow | N/A |
|       pyod_mad         | N/A | N/A | N/A |
|       pyod_mcd         | 77.7108 | 77.7108 | N/A |
|       pyod_pca         |  |  | 75.2592 |
|       pyod_rod         | slow | slow | N/A |
|       pyod_sod         |  |  | 69.9903 |
|       pyod_sos         |  |  | N/A |
|    pyod_autoencoder     | 76.3747  | 79.6897 | N/A |
|       pyod_vae         |  |  | N/A |
|     pyod_so_gaal       |  |  | N/A |
|     pyod_mo_gaal       | 31.979 | 38.4729 | N/A |
|     pyod_deepsvdd      | 75.3709 | 78.2577 | 54.5126 |
|         DAGMM          |  |  | 65.8606 |


| (semi-)Supervised Algorithm | First | Second | Baseline la=0.01|
| -------------------- | ------- | --------| ------- | 
|      pyod_xgbod      | slow | slow | 80.3048 |
|       GANomaly       |  |  | 68.9585 |
|        DeepSAD       |  |  | 75.3972 |
|         REPEN        |  |  | 77.9128 |
|        DevNet        |  |  | 80.2772 |
|        PReNet        |  |  | 80.4082 |
|        FEAWAD        |  |  | 74.2951 |
|          NB          |  |  | 61.1410 |
|         SVM          |  |  | 59.8228 |
|         MLP          |  |  | 57.4951 |
|       ResNet         |  |  | 64.7666 |
|    FTTransformer     |  |  | 78.4203 |
|          RF          |  |  | 61.8808 |
|         LGB          |  |  | 64.0056 |
|         XGB          |  |  | 78.6477 |
|        CatB          |  |  | 83.7497 |
