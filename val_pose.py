import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

# BILIBILI UP 魔傀面具
# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

# 精度小数点保留位数修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第五点
# 最终论文的参数量和计算量统一以这个脚本运行出来的为准

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = '/root/autodl-tmp/ultralytics-yolo11-main/runs/pose/train/v11_pose/weights/best.pt'
    model = YOLO(model_path) # 选择训练好的权重路径
    result = model.val(data='/root/autodl-tmp/ultralytics-yolo11-main/dataset/coco_pose/data.yaml',  # 修改为关键点检测数据集路径
                        split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
                        imgsz=640,
                        batch=16,
                        # iou=0.7,
                        # rect=False,
                        # save_json=True, # if you need to cal coco metrice
                        project='runs/pose/val',
                        name='v11_pose',
                        )
    
    if model.task == 'pose': # 关键点检测任务
        # 获取基本信息
        length = len(result.names) if hasattr(result, 'names') else 1
        model_names = list(result.names.values()) if hasattr(result, 'names') else ['pose']
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
        
        n_l, n_p, n_g, flops = model_info(model.model)
        
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)

        # 模型信息表格
        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}', 
                                  f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s', 
                                  f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}', 
                                  f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
        print(model_info_table)

        # 关键点检测指标表格
        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Pose Model Metrics"
        
        # 检查是否有pose相关的结果
        if hasattr(result, 'pose') and result.pose is not None:
            # 使用pose相关的指标
            model_metrice_table.field_names = ["Class Name", "Box Precision", "Box Recall", "Box mAP50", "Box mAP50-95", "Pose mAP50", "Pose mAP50-95"]
            
            for idx in range(length):
                class_name = model_names[idx] if idx < len(model_names) else f"class_{idx}"
                
                # Box相关指标
                box_precision = f"{result.box.p[idx]:.4f}" if hasattr(result, 'box') and hasattr(result.box, 'p') and idx < len(result.box.p) else "N/A"
                box_recall = f"{result.box.r[idx]:.4f}" if hasattr(result, 'box') and hasattr(result.box, 'r') and idx < len(result.box.r) else "N/A"
                box_map50 = f"{result.box.ap50[idx]:.4f}" if hasattr(result, 'box') and hasattr(result.box, 'ap50') and idx < len(result.box.ap50) else "N/A"
                box_map50_95 = f"{result.box.ap[idx]:.4f}" if hasattr(result, 'box') and hasattr(result.box, 'ap') and idx < len(result.box.ap) else "N/A"
                
                # Pose相关指标
                pose_map50 = f"{result.pose.ap50[idx]:.4f}" if hasattr(result.pose, 'ap50') and idx < len(result.pose.ap50) else "N/A"
                pose_map50_95 = f"{result.pose.ap[idx]:.4f}" if hasattr(result.pose, 'ap') and idx < len(result.pose.ap) else "N/A"
                
                model_metrice_table.add_row([
                    class_name, 
                    box_precision, 
                    box_recall, 
                    box_map50, 
                    box_map50_95,
                    pose_map50,
                    pose_map50_95
                ])
            
            # 平均指标
            avg_box_precision = f"{result.results_dict.get('metrics/precision(B)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/precision(B)'), (int, float)) else "N/A"
            avg_box_recall = f"{result.results_dict.get('metrics/recall(B)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/recall(B)'), (int, float)) else "N/A"
            avg_box_map50 = f"{result.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/mAP50(B)'), (int, float)) else "N/A"
            avg_box_map50_95 = f"{result.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/mAP50-95(B)'), (int, float)) else "N/A"
            avg_pose_map50 = f"{result.results_dict.get('metrics/mAP50(P)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/mAP50(P)'), (int, float)) else "N/A"
            avg_pose_map50_95 = f"{result.results_dict.get('metrics/mAP50-95(P)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/mAP50-95(P)'), (int, float)) else "N/A"
            
            model_metrice_table.add_row([
                "all(平均数据)", 
                avg_box_precision,
                avg_box_recall,
                avg_box_map50,
                avg_box_map50_95,
                avg_pose_map50,
                avg_pose_map50_95
            ])
        else:
            # 如果没有pose结果，显示可用的指标
            model_metrice_table.field_names = ["Metric", "Value"]
            for key, value in result.results_dict.items():
                if isinstance(value, (int, float)):
                    model_metrice_table.add_row([key, f"{value:.4f}"])
                else:
                    model_metrice_table.add_row([key, str(value)])
        
        print(model_metrice_table)

        # 保存结果到文件
        with open(result.save_dir / 'paper_data.txt', 'w+', errors="ignore", encoding="utf-8") as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrice_table))
        
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
    
    else:
        print(f"当前模型任务类型为: {model.task}")
        print("此脚本仅适用于关键点检测任务(pose)，请检查模型是否为关键点检测模型。")
