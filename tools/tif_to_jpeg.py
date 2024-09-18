from PIL import Image
import os

# 原始图片所在的文件夹
#input_dir = "/path/to/input/folder"
input_dir = "/root/ISECRET/DRIVE_for_test/images"
# 转换后的图片保存的文件夹
output_dir = "/root/ISECRET/DRIVE_for_test"

# 遍历原始图片文件夹下的所有文件
for filename in os.listdir(input_dir):
    if filename.endswith(".tif"):
        # 构造原始图片的完整路径
        input_path = os.path.join(input_dir, filename)
        
        # 构造转换后的图片的完整路径
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpeg")
        
        # 打开原始图片
        with Image.open(input_path) as img:
            # 将原始图片转换为 JPEG 格式
            img.convert("RGB").save(output_path, "JPEG")
