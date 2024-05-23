import os
import cv2

def convert_wider_to_yolo(wider_annotations_path, images_dir, output_dir, class_id=0):
    # 如果輸出目錄不存在，則創建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 讀取 WIDER Face 標註文件
    with open(wider_annotations_path, 'r') as f:
        lines = f.readlines()

    i = 0
    # 遍歷每一行標註
    while i < len(lines):
        line = lines[i].strip()
        # 如果行結尾是 .jpg，表示這是一個影像文件名
        if line.endswith('.jpg'):
            # 獲取影像的完整路徑並讀取影像
            image_path = os.path.join(images_dir, line)
            img = cv2.imread(image_path)
            img_height, img_width, _ = img.shape

            # 讀取該影像中人臉的數量
            num_faces = int(lines[i + 1].strip())
            yolo_annotations = []
            
            # 讀取每個人臉的邊界框數據
            for j in range(num_faces):
                face_data = list(map(int, lines[i + 2 + j].strip().split()[:4]))
                x1, y1, width, height = face_data

                # 計算 YOLO 格式所需的中心坐標和寬度、高度，並進行標準化
                x_center = (x1 + width / 2) / img_width
                y_center = (y1 + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height

                # 構建 YOLO 格式的標註字符串
                yolo_annotations.append(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}")

            # 創建與影像同名的標註文件（副檔名為 .txt），並保存標註數據
            annotation_file = os.path.join(output_dir, os.path.splitext(os.path.basename(line))[0] + '.txt')
            with open(annotation_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            # 移動到下一個影像標註（跳過當前影像的標註行數）
            i += 2 + num_faces
        else:
            i += 1

# 範例使用
wider_annotations_path = '/Users/liushiwen/Downloads/wider_face_split/wider_face_val_bbx_gt.txt'
images_dir = '/Users/liushiwen/Downloads/WIDER_val/images'
output_dir = '/Users/liushiwen/Downloads/Submission_example'

convert_wider_to_yolo(wider_annotations_path, images_dir, output_dir)
