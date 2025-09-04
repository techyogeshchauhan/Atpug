import os
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

class TrainingMonitor:
    def __init__(self, project_dir="runs/train", exp_name="exp"):
        self.project_dir = project_dir
        self.exp_name = exp_name
        self.results_file = os.path.join(project_dir, exp_name, "results.csv")
        self.metrics_file = os.path.join(project_dir, exp_name, "metrics.json")
        
    def check_training_status(self):
        """Check if training is currently running"""
        if not os.path.exists(self.results_file):
            return False, "Training not started or results file not found"
        
        # Check if file was modified recently (within last 5 minutes)
        file_time = os.path.getmtime(self.results_file)
        current_time = time.time()
        
        if current_time - file_time < 300:  # 5 minutes
            return True, "Training is running"
        else:
            return False, "Training may have stopped"
    
    def get_training_progress(self):
        """Get current training progress"""
        if not os.path.exists(self.results_file):
            return None
        
        try:
            df = pd.read_csv(self.results_file)
            if len(df) == 0:
                return None
            
            latest = df.iloc[-1]
            
            # Calculate progress
            current_epoch = latest.get('epoch', 0)
            total_epochs = 100  # Default, can be made configurable
            
            progress = {
                'current_epoch': current_epoch,
                'total_epochs': total_epochs,
                'progress_percent': (current_epoch / total_epochs) * 100,
                'box_loss': latest.get('train/box_loss', 0),
                'cls_loss': latest.get('train/cls_loss', 0),
                'dfl_loss': latest.get('train/dfl_loss', 0),
                'total_loss': latest.get('train/box_loss', 0) + latest.get('train/cls_loss', 0) + latest.get('train/dfl_loss', 0),
                'last_update': datetime.fromtimestamp(os.path.getmtime(self.results_file)).strftime('%H:%M:%S')
            }
            
            return progress
            
        except Exception as e:
            return None
    
    def estimate_completion_time(self, progress):
        """Estimate time to completion"""
        if not progress:
            return "Unable to estimate"
        
        # Get training start time from results file
        if os.path.exists(self.results_file):
            start_time = os.path.getctime(self.results_file)
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            current_epoch = progress['current_epoch']
            if current_epoch > 0:
                time_per_epoch = elapsed_time / current_epoch
                remaining_epochs = progress['total_epochs'] - current_epoch
                remaining_time = time_per_epoch * remaining_epochs
                
                eta = datetime.now() + timedelta(seconds=remaining_time)
                return eta.strftime('%H:%M:%S')
        
        return "Unable to estimate"
    
    def plot_training_curves(self, save_path=None):
        """Plot training loss curves"""
        if not os.path.exists(self.results_file):
            print("❌ Results file not found")
            return
        
        try:
            df = pd.read_csv(self.results_file)
            if len(df) == 0:
                print("❌ No training data found")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Progress', fontsize=16)
            
            # Box Loss
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], 'b-', label='Box Loss')
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            
            # Classification Loss
            axes[0, 1].plot(df['epoch'], df['train/cls_loss'], 'r-', label='Classification Loss')
            axes[0, 1].set_title('Classification Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
            
            # DFL Loss
            axes[1, 0].plot(df['epoch'], df['train/dfl_loss'], 'g-', label='DFL Loss')
            axes[1, 0].set_title('DFL Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
            
            # Total Loss
            total_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
            axes[1, 1].plot(df['epoch'], total_loss, 'purple', label='Total Loss')
            axes[1, 1].set_title('Total Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"📊 Training curves saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"❌ Error plotting curves: {e}")
    
    def print_status(self):
        """Print current training status"""
        print("🚀 Training Monitor")
        print("=" * 50)
        
        # Check if training is running
        is_running, status_msg = self.check_training_status()
        print(f"Status: {'🟢' if is_running else '🔴'} {status_msg}")
        
        # Get progress
        progress = self.get_training_progress()
        if progress:
            print(f"\n📊 Progress:")
            print(f"   Epoch: {progress['current_epoch']}/{progress['total_epochs']} ({progress['progress_percent']:.1f}%)")
            print(f"   Box Loss: {progress['box_loss']:.4f}")
            print(f"   Classification Loss: {progress['cls_loss']:.4f}")
            print(f"   DFL Loss: {progress['dfl_loss']:.4f}")
            print(f"   Total Loss: {progress['total_loss']:.4f}")
            print(f"   Last Update: {progress['last_update']}")
            
            # Estimate completion time
            eta = self.estimate_completion_time(progress)
            print(f"   Estimated Completion: {eta}")
            
        else:
            print("❌ No training progress found")
        
        print("\n" + "=" * 50)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Monitor YOLO training progress")
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--plot', action='store_true', help='Generate training curves plot')
    parser.add_argument('--continuous', action='store_true', help='Monitor continuously')
    parser.add_argument('--interval', type=int, default=30, help='Update interval in seconds')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.project, args.name)
    
    if args.plot:
        plot_path = os.path.join(args.project, args.name, "training_curves.png")
        monitor.plot_training_curves(plot_path)
        return
    
    if args.continuous:
        print("🔄 Starting continuous monitoring...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
                monitor.print_status()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
    else:
        monitor.print_status()

if __name__ == '__main__':
    main() 