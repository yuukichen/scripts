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


"""
关键点检测评估指标说明：

1. Box Detection Metrics (边界框检测指标):
   - Precision: 检测到的正确边界框占所有检测框的比例
   - Recall: 检测到的正确边界框占所有真实框的比例
   - mAP50: IoU阈值为0.5时的平均精度均值
   - mAP50-95: IoU阈值从0.5到0.95的平均精度均值

2. Pose/Keypoint Metrics (关键点检测指标):
   - Pose mAP50: OKS阈值为0.5时的关键点平均精度
   - Pose mAP50-95: OKS阈值从0.5到0.95的关键点平均精度
   - OKS (Object Keypoint Similarity): 关键点相似度，用于评估关键点定位精度

3. OKS计算公式:
   OKS = Σ(exp(-di²/2si²κi²)) / Σ(δ(vi>0))
   其中：
   - di: 预测关键点与真实关键点的欧氏距离
   - si: 对象尺度 (√(area))
   - κi: 每个关键点的常数 (控制衰减率)
   - vi: 关键点的可见性标志

4. 其他重要指标:
   - Fitness: 综合适应度分数，通常为 mAP50-95(P)*0.1 + mAP50-95(B)*0.9
   - Speed: 推理速度，包括预处理、推理、后处理时间
"""

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
                        name='v11_pose_01',
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
        print()
        print("📊 关键点检测模型评估包含以下指标:")
        print("   🔲 Box Detection: 人体边界框检测精度")
        print("   🎯 Pose Detection: 关键点定位精度 (基于OKS评估)")
        print("   ⚡ Performance: 模型性能指标 (参数量、计算量、推理速度)")
        print("   📈 OKS Thresholds: 不同OKS阈值下的mAP表现")
        print()
        print("=" * 80)

        # 模型信息表格
        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}', 
                                  f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s', 
                                  f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}', 
                                  f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
        print(model_info_table)
        print("\n" + "=" * 80 + "\n")

        # 检测框指标表格 (Box Detection Metrics)
        box_metrice_table = PrettyTable()
        box_metrice_table.title = "Box Detection Metrics"
        box_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        
        if hasattr(result, 'box') and result.box is not None:
            for idx in range(length):
                class_name = model_names[idx] if idx < len(model_names) else f"class_{idx}"
                
                precision = f"{result.box.p[idx]:.4f}" if hasattr(result.box, 'p') and idx < len(result.box.p) else "N/A"
                recall = f"{result.box.r[idx]:.4f}" if hasattr(result.box, 'r') and idx < len(result.box.r) else "N/A"
                f1_score = f"{result.box.f1[idx]:.4f}" if hasattr(result.box, 'f1') and idx < len(result.box.f1) else "N/A"
                map50 = f"{result.box.ap50[idx]:.4f}" if hasattr(result.box, 'ap50') and idx < len(result.box.ap50) else "N/A"
                map75 = f"{result.box.all_ap[idx, 5]:.4f}" if hasattr(result.box, 'all_ap') and result.box.all_ap.shape[0] > idx and result.box.all_ap.shape[1] > 5 else "N/A"
                map50_95 = f"{result.box.ap[idx]:.4f}" if hasattr(result.box, 'ap') and idx < len(result.box.ap) else "N/A"
                
                box_metrice_table.add_row([class_name, precision, recall, f1_score, map50, map75, map50_95])
            
            # 平均指标
            avg_precision = f"{result.results_dict.get('metrics/precision(B)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/precision(B)'), (int, float)) else "N/A"
            avg_recall = f"{result.results_dict.get('metrics/recall(B)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/recall(B)'), (int, float)) else "N/A"
            avg_f1 = f"{np.mean(result.box.f1[:length]):.4f}" if hasattr(result.box, 'f1') else "N/A"
            avg_map50 = f"{result.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/mAP50(B)'), (int, float)) else "N/A"
            avg_map75 = f"{np.mean(result.box.all_ap[:length, 5]):.4f}" if hasattr(result.box, 'all_ap') and result.box.all_ap.shape[1] > 5 else "N/A"
            avg_map50_95 = f"{result.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/mAP50-95(B)'), (int, float)) else "N/A"
            
            box_metrice_table.add_row(["all(平均数据)", avg_precision, avg_recall, avg_f1, avg_map50, avg_map75, avg_map50_95])
        
        print(box_metrice_table)
        print("\n" + "=" * 80 + "\n")

        # 关键点指标表格 (Pose/Keypoint Metrics)
        pose_metrice_table = PrettyTable()
        pose_metrice_table.title = "Pose/Keypoint Metrics"
        
        if hasattr(result, 'pose') and result.pose is not None:
            pose_metrice_table.field_names = ["Class Name", "Pose mAP50", "Pose mAP75", "Pose mAP50-95", "Pose Precision", "Pose Recall"]
            
            for idx in range(length):
                class_name = model_names[idx] if idx < len(model_names) else f"class_{idx}"
                
                pose_map50 = f"{result.pose.ap50[idx]:.4f}" if hasattr(result.pose, 'ap50') and idx < len(result.pose.ap50) else "N/A"
                pose_map75 = f"{result.pose.all_ap[idx, 5]:.4f}" if hasattr(result.pose, 'all_ap') and result.pose.all_ap.shape[0] > idx and result.pose.all_ap.shape[1] > 5 else "N/A"
                pose_map50_95 = f"{result.pose.ap[idx]:.4f}" if hasattr(result.pose, 'ap') and idx < len(result.pose.ap) else "N/A"
                pose_precision = f"{result.pose.p[idx]:.4f}" if hasattr(result.pose, 'p') and idx < len(result.pose.p) else "N/A"
                pose_recall = f"{result.pose.r[idx]:.4f}" if hasattr(result.pose, 'r') and idx < len(result.pose.r) else "N/A"
                
                pose_metrice_table.add_row([class_name, pose_map50, pose_map75, pose_map50_95, pose_precision, pose_recall])
            
            # 平均指标
            avg_pose_map50 = f"{result.results_dict.get('metrics/mAP50(P)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/mAP50(P)'), (int, float)) else "N/A"
            avg_pose_map75 = f"{np.mean(result.pose.all_ap[:length, 5]):.4f}" if hasattr(result.pose, 'all_ap') and result.pose.all_ap.shape[1] > 5 else "N/A"
            avg_pose_map50_95 = f"{result.results_dict.get('metrics/mAP50-95(P)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/mAP50-95(P)'), (int, float)) else "N/A"
            avg_pose_precision = f"{result.results_dict.get('metrics/precision(P)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/precision(P)'), (int, float)) else "N/A"
            avg_pose_recall = f"{result.results_dict.get('metrics/recall(P)', 'N/A'):.4f}" if isinstance(result.results_dict.get('metrics/recall(P)'), (int, float)) else "N/A"
            
            pose_metrice_table.add_row(["all(平均数据)", avg_pose_map50, avg_pose_map75, avg_pose_map50_95, avg_pose_precision, avg_pose_recall])
            
            print(pose_metrice_table)
            print("\n" + "=" * 80 + "\n")
        
        # 额外关键点指标表格 (Additional Pose Metrics)
        additional_metrics_table = PrettyTable()
        additional_metrics_table.title = "Additional Pose Metrics"
        additional_metrics_table.field_names = ["Metric Name", "Value", "Description"]
        
        # 收集所有可能的关键点相关指标
        pose_metrics = {}
        for key, value in result.results_dict.items():
            if isinstance(value, (int, float)):
                pose_metrics[key] = value
        
        # 关键点特有指标
        metric_descriptions = {
            'metrics/mAP50(P)': '关键点检测 mAP@0.5 (基于OKS)',
            'metrics/mAP50-95(P)': '关键点检测 mAP@0.5:0.95 (基于OKS)',
            'metrics/precision(P)': '关键点检测精确率',
            'metrics/recall(P)': '关键点检测召回率',
            'metrics/mAP50(B)': '边界框检测 mAP@0.5',
            'metrics/mAP50-95(B)': '边界框检测 mAP@0.5:0.95',
            'metrics/precision(B)': '边界框检测精确率',
            'metrics/recall(B)': '边界框检测召回率',
            'fitness': '综合适应度分数 (mAP50-95(P)*0.1 + mAP50-95(B)*0.9)',
        }
        
        for metric_name, value in pose_metrics.items():
            description = metric_descriptions.get(metric_name, '评估指标')
            additional_metrics_table.add_row([metric_name, f"{value:.4f}", description])
        
        print(additional_metrics_table)
        print("\n" + "=" * 80 + "\n")
        
        # OKS阈值相关指标 (如果有的话)
        if hasattr(result, 'pose') and hasattr(result.pose, 'all_ap'):
            oks_thresholds_table = PrettyTable()
            oks_thresholds_table.title = "OKS Thresholds mAP"
            oks_thresholds_table.field_names = ["OKS Threshold", "mAP"]
            
            oks_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
            for i, threshold in enumerate(oks_thresholds):
                if i < result.pose.all_ap.shape[1]:
                    avg_map_at_threshold = np.mean(result.pose.all_ap[:length, i])
                    oks_thresholds_table.add_row([f"{threshold:.2f}", f"{avg_map_at_threshold:.4f}"])
            
            print(oks_thresholds_table)
            print("\n" + "=" * 80 + "\n")
        
        # 如果没有pose结果，显示可用的指标
        if not (hasattr(result, 'pose') and result.pose is not None):
            fallback_table = PrettyTable()
            fallback_table.title = "Available Metrics"
            fallback_table.field_names = ["Metric", "Value"]
            for key, value in result.results_dict.items():
                if isinstance(value, (int, float)):
                    fallback_table.add_row([key, f"{value:.4f}"])
                else:
                    fallback_table.add_row([key, str(value)])
            print(fallback_table)
            print("\n" + "=" * 80 + "\n")

        # 保存结果到文件
        with open(result.save_dir / 'paper_data.txt', 'w+', errors="ignore", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write("YOLO关键点检测模型评估报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(str(model_info_table))
            f.write('\n\n')
            
            if 'box_metrice_table' in locals():
                f.write(str(box_metrice_table))
                f.write('\n\n')
            
            if 'pose_metrice_table' in locals():
                f.write(str(pose_metrice_table))
                f.write('\n\n')
            
            if 'additional_metrics_table' in locals():
                f.write(str(additional_metrics_table))
                f.write('\n\n')
            
            if 'oks_thresholds_table' in locals():
                f.write(str(oks_thresholds_table))
                f.write('\n\n')
            
            if 'fallback_table' in locals():
                f.write(str(fallback_table))
                f.write('\n\n')
            
            f.write("="*60 + "\n")
            f.write("报告生成完成\n")
            f.write("="*60 + "\n")
        
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
    
    else:
        print(f"当前模型任务类型为: {model.task}")
        print("此脚本仅适用于关键点检测任务(pose)，请检查模型是否为关键点检测模型。")
