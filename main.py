import subprocess
import os

# 遍历data文件夹中的所有文件名并打印为一个列表
data_folder = "path"
file_list = os.listdir(data_folder)
# print(file_list)

# 创建result文件夹
result_folder = "result"
os.makedirs(result_folder, exist_ok=True)

names = []

# 在result文件夹下为每个文件名创建一个文件夹（去掉后缀）
for file_name in file_list:
    file_name_without_extension = os.path.splitext(file_name)[0]
    names.append(file_name_without_extension)
    folder_name = os.path.join(result_folder, file_name_without_extension)
    os.makedirs(folder_name, exist_ok=True)
    
names = sorted(names, key=lambda x: int(x[4:]))


for i in names[37:38]:
    print('kaishiyunxing')
    # 调用SA.py并传入变量
    try:
        subprocess.run(["python", "SA.py", i], check=True)
    except subprocess.CalledProcessError as e:
        print("Error calling SA.py:", e)
