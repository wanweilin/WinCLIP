# WinCLIP
Unofficial implementation of: WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation, CVPR 2023 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html)

## Installation
1. Clone the repository to your local machine:
```
git clone https://github.com/wanweilin/WinCLIP.git
```

2. Navigate to the project directory:
```
cd WinCLIP
pip install -r requirements.txt
```
## Usage Examples

```
python winclip_ac.py
```

## Dataset
Mvtec AD 

## Todo
1. Implement segmentation part

## Experimental Results
| Obj Type    | AUROC   | AUPR    | F1-Max  |
|-------------|---------|---------|---------|
| bottle      | 0.9833  | 0.9947  | 0.9593  |
| cable       | 0.8802  | 0.9222  | 0.8515  |
| capsule     | 0.6635  | 0.8956  | 0.9046  |
| carpet      | 0.9888  | 0.9968  | 0.9773  |
| grid        | 0.9879  | 0.9963  | 0.9825  |
| hazelnut    | 0.9305  | 0.9625  | 0.8921  |
| leather     | 1.0000  | 1.0000  | 1.0000  |
| metal_nut   | 0.9428  | 0.9871  | 0.9278  |
| pill        | 0.7908  | 0.9577  | 0.9156  |
| screw       | 0.6958  | 0.8677  | 0.8561  |
| tile        | 1.0000  | 1.0000  | 0.9941  |
| toothbrush  | 0.8431  | 0.9389  | 0.8889  |
| transistor  | 0.8946  | 0.8605  | 0.8193  |
| wood        | 0.9838  | 0.9949  | 0.9677  |
| zipper      | 0.8851  | 0.9667  | 0.9063  |
|-------------|---------|---------|---------|
| Avg         | 0.8980  | 0.9561  | 0.9229  |
| All Type    | 0.7482  | 0.8918  | 0.8465  |

## Citation
If you find this code useful, please consider citing the original paper:
```
@InProceedings{Jeong_2023_CVPR,
    author    = {Jeong, Jongheon and Zou, Yang and Kim, Taewan and Zhang, Dongqing and Ravichandran, Avinash and Dabeer, Onkar},
    title     = {WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {19606-19616}
}
```

## Acknowledgements
This project borrows some code from [OpenCLip](https://github.com/mlfoundations/open_clip) and [DRAEM](https://github.com/VitjanZ/DRAEM/tree/main), thanks for their admiring contributions!
