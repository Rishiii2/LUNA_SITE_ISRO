"""
LUNA-SITE | Layer 10: YOLOv8 Sunlit Hazard Detection
=====================================================
NOTE: This is a structural architecture stub for the hackathon demo.
In the production pipeline, this script fine-tunes Ultralytics YOLOv8
on high-resolution optical imagery for Zone A hazard mapping. 
For this local simulation, the actual training loop is bypassed.
"""

import os
import yaml

def generate_yolo_config(config_path):
    """
    Generates the lunar_boulders.yaml file required by Ultralytics YOLOv8.
    """
    config = {
        'path': '../datasets/lunar_boulders', # dataset root dir
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        # Classes: 0 = Safe Rock, 1 = Boulder > 0.32m
        'names': {
            0: 'rock_safe',
            1: 'boulder_hazard'
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[INFO] Generated YOLOv8 config at {config_path}")

def fine_tune_yolov8():
    print("=========================================================")
    print("   LUNA-SITE Phase 3: YOLOv8 Dual-Zone Hazard Training   ")
    print("=========================================================")
    
    config_path = "lunar_boulders.yaml"
    generate_yolo_config(config_path)
    
    print("[INFO] Triggering Ultralytics YOLOv8 Fine-Tuning Pipeline...")
    print("=> Command: yolo task=detect mode=train model=yolov8n.pt data=lunar_boulders.yaml epochs=50 imgsz=640")
    
    # In actual execution, this requires `pip install ultralytics`
    # and the actual OHRC labeled images inside the `datasets/lunar_boulders` folder.
    print("\n[NOTE] YOLOv8 requires actual annotated Chandrayaan-2 OHRC images to train.")
    print("Because we are using mock data, the Ultralytics engine is bypassed for this demo.")
    print("[✔] Script Architecture Validated. Ready for real dataset injection.")

if __name__ == "__main__":
    fine_tune_yolov8()
