# [CRPGCN]:Predicting circRNA-DiseaseAssociations Using Graph Convolutional Network Based on Heterogeneous Network(https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04467-z)

### Abstract
**Background:**
The existing studies show that circRNAs can be used as biomarkers} of diseases and play a prominent role in the treatment and diagnosis of diseases. However, the relationships between the vast majority of circRNAs and diseases are still unclear, and more experiments are needed to study the mechanism of circRNAs. Nowadays, some scholars use the \textcolor{red}{attributes} between circRNAs and diseases to study and predict their associations. Nonetheless, most of the existing experimental methods use less information about the attributes of circRNAs, which has a certain impact on the accuracy of the final prediction results. On the other hand, some scholars also apply experimental methods to predict the associations between circRNAs and diseases. But such methods are usually expensive and time-consuming. \textcolor{red}{Based on these shortcomings, follow-up studies are needed to propose more effective computation-based methods to predict the associations between circRNAs and diseases.

**Results:**
In this study, a novel algorithm (method) is proposed, which is based on the Graph Convolutional Network (GCN) constructed with Random Walk with Restart (RWR) and Principal Component Analysis (PCA) to predict the associations between circRNAs and diseases (CRPGCN). In the construction of CRPGCN, the RWR algorithm is used to improve the similarity associations of the computed nodes with their neighbours. After that, the PCA method is used to dimensionality reduction and extract features, it makes the \textcolor{red}{connections} between circRNAs with higher similarity and diseases closer. Finally, The GCN algorithm is used to learn the features between circRNAs and diseases and calculate the final similarity scores, and the learning data are constructed from the adjacency matrix, similarity matrix and feature matrix as a heterogeneous adjacency matrix and a heterogeneous feature matrix.

**Conclusions:**
After 2-fold cross-validation (CV), 5-fold CV and 10-fold CV, the area under the ROC curve (AUC) of the CRPGCN are 0.9490, 0.9720 and 0.9722, respectively. The CRPGCN method has a valuable effect in predict the associations between circRNAs and diseases.

### The flowchart of CRPGCN
![avatar](https://github.com/KajiMaCN/CRPGCN/blob/main/figure/Figure%201.png)

### Contributions
**The main contributions of this work are summarized as follows:**

- The CRPGCN method incorporates the RWR similarity calculation method and the PCA feature extraction method, allowing the calculated nodes to better combine the similarity between neighbouring nodes while greatly reducing the impact on the prediction results.
	
- The CRPGCN algorithm improves prediction accuracy and has the highest AUC values and AUPR values when compared to advanced algorithms.

- The GCRGCN algorithm is more stable than some of the advanced algorithms, and its AUCs are stable when compared by a variety of methods with different datasets.

- By comparing various evaluation metrics, the CRPGCN algorithm outperforms other advanced algorithms in terms of overall performance.

### Requirements
```
numpy~=1.19.4
pandas~=1.1.5
matplotlib~=3.3.3
pyrwr~=1.0.0
tensorflow~=2.4.0
tensorflow-gpu~=2.4.0
scipy~=1.5.4
networkx~=2.5
tqdm~=4.55.0
sklearn~=0.0
scikit-learn~=0.23.2
```
### cuda version
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Cuda compilation tools, release 11.1, V11.1.74
Build cuda_11.1
```
### Datasets:
1. AM: Adjacency matrix
2. Disease_sim : Disease similarity matrix DS
3. RNA_sim: circRNA similarity matrix CS

### Running steps
1. Install the runtime environment in the Terminal with the command: pip install -r requirements.txt
2. Put the adjacency matrix (.csv), disease similarity matrix (.csv) and RNA similarity matrix (.csv) into the dataset folder
3. run main.py

### Additional Environment Download Address
If you are unable to debug the CRPGCN runtime environment, you can download the environment we have configured in the following way:

- Baidu Netdisk

    ![avatar](figure/baidu.png)
    
        Link：https://pan.baidu.com/s/1tc62HTdaRz3CgKBzZuM0kw 
        Verification Code：pa1u
### Citation
If you found this paper or code helpful, please cite our paper:

    @article{CRPGCN2021,
          title   = {CRPGCN:Predicting circRNA-DiseaseAssociations Using Graph Convolutional Network Based on Heterogeneous Network},
          author  = {Zhihao Ma, Zhufang Kuang and Lei Deng}, 
          journal = {BMC Bioinformatics},
          month   = {Nov},
          year    = {2021}
          doi     = {https://doi.org/10.1186/s12859-021-04467-z}
        }

### Others
**If you have any questions, please submit your issues.**
