# Step 1:获取全部文件路径和类别
import os
"""
使用前需要将PVL_data文件夹中的非数据文件清除干净
"""

# 遍历PVL_dataset文件夹下的PVL和NPVL文件夹
base_dir = '/data/jiazhen/code/3DClassification-framework/dataset/lesions_classification_dataset'
for cls_name in os.listdir(base_dir):
    # 获取文件夹名
    cls_path = os.path.join(base_dir, cls_name)
    for pat_name in os.listdir(cls_path):
        pat_path = os.path.join(cls_path, pat_name)

        # 提取PVL或NPVL标签
        if cls_name == 'Lesions':
            label = 'Lesions'
        elif cls_name == 'No_lesions':
            label = 'No_lesions'
        else:
            print("Class name error!")

        # 将结果格式化为字符串
        result = f"{label},{pat_path}"
        # 将结果写入txt文件
        with open('/data/jiazhen/code/3DClassification-framework/dataset/lesions_dataset_files.txt', 'a') as f:
            f.write(result + '\n')




