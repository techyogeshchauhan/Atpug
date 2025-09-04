import os
import cv2
import argparse
from ultralytics import YOLO
from pathlib import Path
import glob

def detect_images(model_path, input_path, output_dir, conf=0.25):
    """
    Detect terminal seals in images
    
    Args:
        model_path: Path to trained YOLO model
        input_path: Path to image or folder of images
        output_dir: Directory to save detected images
        conf: Confidence threshold
    """
    
    # Load model
    print(f"📦 Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images
    if os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(glob.glob(os.path.join(input_path, ext)))
            image_paths.extend(glob.glob(os.path.join(input_path, ext.upper())))
    
    if not image_paths:
        print(f"❌ No images found in {input_path}")
        return
    
    print(f"🔍 Found {len(image_paths)} images to process")
    
    # Process each image
    for i, img_path in enumerate(image_paths, 1):
        print(f"Processing {i}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        # Run detection
        results = model.predict(img_path, conf=conf, save=False)
        
        # Load image for drawing
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Could not load image: {img_path}")
            continue
        
        # Draw detections
        detection_count = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score = float(box.conf[0])
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"Terminal Seal: {conf_score:.2f}"
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detection_count += 1
        
        # Save result
        output_filename = f"detected_{os.path.basename(img_path)}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img)
        
        print(f"   ✅ Detected {detection_count} seals, saved to {output_filename}")
    
    print(f"\n🎉 Detection complete! Results saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Detect terminal seals in images")
    parser.add_argument('--model', type=str, default='runs/train/exp2/best.pt', 
                       help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to image or folder of images')
    parser.add_argument('--output', type=str, default='detected_images',
                       help='Output directory for detected images')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"❌ Input not found: {args.input}")
        return
    
    print("🚀 Terminal Seal Detection")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Confidence: {args.conf}")
    print("=" * 40)
    
    detect_images(args.model, args.input, args.output, args.conf)

if __name__ == '__main__':
    main() 