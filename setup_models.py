"""
Helper script to download or setup YOLOv11 models
This script helps you download pretrained YOLOv11 models from Ultralytics
"""

from ultralytics import YOLO
import os

def download_yolo_models():
    """Download YOLOv11 pretrained models and save them with proper naming"""
    
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    models = {
        "nano": "yolo11n.pt",
        "small": "yolo11s.pt", 
        "medium": "yolo11m.pt"
    }
    
    print("=" * 60)
    print("YOLOv11 Model Setup Helper")
    print("=" * 60)
    print("\nThis will download pretrained YOLOv11 models from Ultralytics.")
    print("Models will be saved in the 'weights/' folder.\n")
    
    for size, model_name in models.items():
        output_path = os.path.join(weights_dir, f"best_{size}.pt")
        
        if os.path.exists(output_path):
            print(f"✓ {size.upper()} model already exists at: {output_path}")
            continue
            
        try:
            print(f"\n→ Downloading {size.upper()} model ({model_name})...")
            model = YOLO(model_name)  # This will auto-download if not present
            
            # Save to our naming convention
            model.save(output_path)
            print(f"✓ Saved as: {output_path}")
            
        except Exception as e:
            print(f"✗ Error downloading {size} model: {e}")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nYou can now run: streamlit run app.py")
    print("\nNote: These are pretrained COCO models.")
    print("For custom detection, train your own model and replace these files.")

if __name__ == "__main__":
    download_yolo_models()
