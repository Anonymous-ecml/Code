## Requirements
python == 3.9.1 pytorch == 1.12.1 numpy == 1.22.4 pandas == 2.1.4 scikit-learn == 1.3.2 torch_geometric == 2.2.0

## Datasets
The MIMIC-III and eICU databases are obtained from https://mimic.mit.edu/ and https://eicu-crd.mit.edu.
Detailed information of data extraction on the MIMIC-III and eICU databases can be found in ```data construction``` directory. 

## Main Entrance
main.py contains both training code and evaluation code.

## Ablation Studies
$\mathbf{Our_{\alpha}}$ (A variation of our approach that treats only the anchor and its counterparts in different graph views as positives):  
在model.py中用clloss_alpha function替换clloss function  
$\mathbf{Our_{\beta}}$ (A variation of our approach that omits the node connected to the anchor, which tests the efficacy of node-level clustering on patient graphs):  
在model.py中用clloss_beta function替换clloss function  
$\mathbf{Our_{\gamma}}$ (A variation of our approach that omits the neighbors of the anchor):  
在model.py中用clloss_gamma function替换clloss function  
$\mathbf{Our_{\delta}}$ (A variation of our approach that uses edge-based masking instead of path-based masking):  
用model-wo-MaskPath.py替换model.py

## Self-supervised and Supervised Learning Settings
model.py 是Supervised Learning Setting的代码，model-ssl.py 是Self-supervised Learning Setting的代码
