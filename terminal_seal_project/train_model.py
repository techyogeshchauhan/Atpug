import os
from ultralytics import YOLO
import argparse
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for terminal seal detection")
    parser.add_argument('--data', type=str, default='data/dataset.yaml', help='Path to dataset.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Pre-trained model path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--project', type=str, default='runs/train', help='Training output directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--device', type=str, default='cpu', help='GPU device (e.g., "0" or "cpu")')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate')
    return parser.parse_args()

def verify_dataset(data_path):
    """Verify dataset.yaml and folder structure for case where images and labels are in the same directory"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"dataset.yaml not found at {data_path}")
    
    with open(data_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("🔍 Verifying dataset structure...")
    
    for split in ['train', 'val', 'test']:
        img_path = config.get(split)
        if not img_path:
            print(f"⚠️  Warning: {split} path not found in dataset.yaml")
            continue

        # Convert to absolute path if relative
        if not os.path.isabs(img_path):
            img_path = os.path.abspath(img_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"{split} images folder not found at {img_path}")

        img_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        label_files = [f for f in os.listdir(img_path) if f.endswith('.txt')]

        print(f"✅ {split}: {len(img_files)} images, {len(label_files)} labels at {img_path}")

        if len(img_files) == 0:
            raise ValueError(f"No image files found in {split} directory: {img_path}")
        if len(label_files) == 0:
            print(f"⚠️  Warning: No label files found in {split} directory: {img_path}")

        # Check that for every image, a .txt label file exists
        missing_labels = [f for f in img_files if not os.path.exists(os.path.join(img_path, os.path.splitext(f)[0] + '.txt'))]
        if missing_labels:
            print(f"⚠️  Warning: Missing label files for images: {missing_labels}")
        else:
            print(f"✅ All images in {split} have corresponding label files.")
    
    print("✅ Dataset verification passed!")
    return config

def plot_metrics(results, output_dir):
    """Plot training metrics"""
    try:
        metrics = pd.DataFrame({
            'mAP50': results.results_dict['metrics/mAP50(B)'],
            'mAP50-95': results.results_dict['metrics/mAP50-95(B)'],
            'Precision': results.results_dict['metrics/precision(B)'],
            'Recall': results.results_dict['metrics/recall(B)']
        }, index=['Validation'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics.plot(kind='bar', ax=ax)
        ax.set_title('YOLOv8 Validation Metrics')
        ax.set_ylabel('Score')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics.png'))
        plt.close()
        print(f"📊 Metrics plot saved to {os.path.join(output_dir, 'metrics.png')}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create metrics plot: {e}")

def main():
    args = parse_args()
    
    print("🚀 Terminal Seal Detection Model Training")
    print("=" * 50)
    
    # Verify dataset
    config = verify_dataset(args.data)
    
    # Initialize model
    print(f"📦 Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Create output directory
    output_dir = os.path.join(args.project, args.name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    print("🚀 Starting training...")
    print(f"📊 Training parameters:")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch}")
    print(f"   - Image size: {args.imgsz}")
    print(f"   - Device: {args.device}")
    print(f"   - Learning rate: {args.lr0} -> {args.lrf}")
    print(f"   - Patience: {args.patience}")
    
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            device=args.device,
            plots=True,
            save=True,
            patience=args.patience,
            lr0=args.lr0,
            lrf=args.lrf,
            verbose=True
        )
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Validate model
    print("🔍 Validating model...")
    try:
        val_results = model.val(data=args.data, split='val')
        print("✅ Validation completed!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        val_results = None
    
    # Test on test set (optional)
    print("🧪 Testing model...")
    try:
        test_results = model.val(data=args.data, split='test')
        print("✅ Testing completed!")
    except Exception as e:
        print(f"⚠️  Testing failed: {e}")
        test_results = None
    
    # Save model
    best_model_path = os.path.join(output_dir, 'best.pt')
    try:
        model.save(best_model_path)
        print(f"💾 Model saved at: {best_model_path}")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
    
    # Plot metrics if validation was successful
    if val_results:
        plot_metrics(val_results, output_dir)
    
    # Print summary
    print("\n📊 Training Summary:")
    print("=" * 30)
    if val_results:
        print(f"mAP50: {val_results.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"mAP50-95: {val_results.results_dict['metrics/mAP50-95(B)']:.4f}")
        print(f"Precision: {val_results.results_dict['metrics/precision(B)']:.4f}")
        print(f"Recall: {val_results.results_dict['metrics/recall(B)']:.4f}")
    
    print(f"Model saved at: {best_model_path}")
    print(f"Results saved at: {output_dir}")
    print("\n🎉 Training process completed!")

if __name__ == '__main__':
    main()