from sklearn.metrics import roc_curve, auc
import os
gt = [
    [61, 180],
    [95, 180],
    [1, 146],
    [31, 180],
    [1, 129],
    [1, 159],
    [46, 180],
    [1, 180],
    [1, 120],
    [1, 150],
    [1, 180],
    [88, 180]
]

dirs = os.listdir(os.path.join('UCSDped_patch', 'ped2', 'test'))
print(dirs)
input()
for d in dirs:
    for img_dir in os.listdir(os.path.join('UCSDped_patch', 'ped2', 'test', d, 'box_img')):
        print(img_dir)