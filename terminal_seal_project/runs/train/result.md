
(asstes) D:\Atpug\terminal_seal_project>python train_model.py
🚀 Terminal Seal Detection Model Training
==================================================
🔍 Verifying dataset structure...
✅ train: 193 images, 0 labels at D:\Atpug\terminal_seal_project\data\images\train
⚠️  Warning: No label files found in train directory: D:\Atpug\terminal_seal_project\data\images\train
⚠️  Warning: Missing label files for images: ['0a7d8ab0113b68e0_20250412_164733.jpeg', '0a7d8ab0113b68e0_20250412_165050.jpeg', '0a7d8ab0113b68e0_20250412_165414.jpeg', '0a7d8ab0113b68e0_20250412_170754.jpeg', '0a7d8ab0113b68e0_20250412_170954.jpeg', '0a7d8ab0113b68e0_20250412_171143.jpeg', '0a7d8ab0113b68e0_20250412_171342.jpeg', '0a7d8ab0113b68e0_20250412_171602.jpeg', '0a7d8ab0113b68e0_20250412_171806.jpeg', '0a7d8ab0113b68e0_20250412_185409.jpeg', '0a7d8ab0113b68e0_20250412_185817.jpeg', '0a7d8ab0113b68e0_20250412_191106.jpeg', '0a7d8ab0113b68e0_20250412_191751.jpeg', '0a7d8ab0113b68e0_20250412_192955.jpeg', '0a7d8ab0113b68e0_20250412_193214.jpeg', '0a7d8ab0113b68e0_20250414_212525.jpg', '0a7d8ab0113b68e0_20250414_212747.jpg', '0a7d8ab0113b68e0_20250414_213003.jpg', '0a7d8ab0113b68e0_20250414_213236.jpg', '0a7d8ab0113b68e0_20250414_213449.jpg', '0a7d8ab0113b68e0_20250414_213657.jpg', '0a7d8ab0113b68e0_20250414_214132.jpg', '0a7d8ab0113b68e0_20250414_214348.jpg', '0a7d8ab0113b68e0_20250414_214547.jpg', '0a7d8ab0113b68e0_20250414_215244.jpg', '0a7d8ab0113b68e0_20250414_215443.jpg', '0a8679f7e9f76eee_20250412_141219.jpeg', '0aac5d7b77d66b6b_20250412_165403.jpg', '0aac5d7b77d66b6b_20250412_173945.jpg', '0d4f5ea21e3c609c_20250412_104643.jpeg', '0d4f5ea21e3c609c_20250412_111011.jpeg', '0d4f5ea21e3c609c_20250412_114209.jpeg', '0d4f5ea21e3c609c_20250412_122310.jpeg', '0d4f5ea21e3c609c_20250412_125150.jpeg', '0d4f5ea21e3c609c_20250412_133536.jpeg', '0d4f5ea21e3c609c_20250412_144029.jpeg', '0d4f5ea21e3c609c_20250412_150539.jpeg', '0d4f5ea21e3c609c_20250412_164356.jpeg', '0d4f5ea21e3c609c_20250412_171646.jpeg', '0d4f5ea21e3c609c_20250412_181407.jpeg', '0d4f5ea21e3c609c_20250412_183950.jpeg', '0d4f5ea21e3c609c_20250412_185723.jppeg', '0d4f5ea21e3c609c_20250412_192337.jpeg', 'ef369392e047fb50_20250418_161310.jpeg', 'ef67061ede767296_20250410_114714.jpeg', 'ef9da9c2c49b820c_20250416_101302.jpeg', 'f1bf6ef29bf3a88d_20250407_202319.jpeg', 'f1bf6ef29bf3a88d_20250410_185738.jpeg', 'f1bf6ef29bf3a88d_20250410_191459.jpeg', 'f1bf6ef29bf3a88d_20250411_144443.jpeg', 'f1bf6ef29bf3a88d_20250411_183718.jpeg', 'f1bf6ef29bf3a88d_20250411_185558.jpeg', 'f1bf6ef29bf3a88d_20250415_150322.jpeg', 'f1bf6ef29bf3a88d_20250417_141644.jpeg', 'f1bf6ef29bf3a88d_20250418_171649.jpeg', 'f1bf6ef29bf3a88d_20250418_173410.jpeg', 'f1bf6ef29bf3a88d_20250418_174626.jpeg', 'f1bf6ef29bf3a88d_20250418_190622.jpeg', 'f1bf6ef29bf3a88d_20250418_213453.jpeg', 'f1bf6ef29bf3a88d_20250419_155640.jpeg', 'f1bf6ef29bf3a88d_20250419_164550.jpeg', 'f1bf6ef29bf3a88d_20250419_170740.jpeg', 'f1bf6ef29bf3a88d_20250419_172403.jpeg', 'f1bf6ef29bf3a88d_20250419_183725.jpeg', 'f1bf6ef29bf3a88d_20250419_192421.jpeg', 'f1c4767541a5d8be_20250417_160035.jpeg', 'f1c4767541a5d8be_20250419_153438.jpeg', 'f35c121b66fc94d6_20250404_135340.jpeg', 'f35c121b66fc94d6_20250411_123215.jpeg', 'f35c121b66fc94d6_20250411_124414.jpeg', 'f35c121b66fc94d6_20250413_105000.jpeg', 'f35c121b66fc94d6_20250414_154315.jpeg', 'f35c121b66fc94d6_20250416_120247.jpeg', 'f35c121b66fc94d6_20250416_122159.jpeg', 'f35c121b66fc94d6_20250417_123205.jpeg', 'f47a041d2233b915_20250304_144119.jpeg', 'f514717a80d1f722_20250406_125945.jpeg', 'f514717a80d1f722_20250412_071949.jpeg', 'f514717a80d1f722_20250412_115805.jpeg', 'f514717a80d1f722_20250412_120618.jpeg', 'f57d8451ce447144_20250419_161200.jpeg', 'f57d8451ce447144_20250420_110726.jpeg', 'f57d8451ce447144_20250420_135139.jpeg', 'f57d8451ce447144_20250420_171948.jpeg', 'f65782572dd89ae8_20250418_135641.jpeg', 'f65782572dd89ae8_20250418_150203.jpeg', 'f683b7037a150e25_20250417_133818.jpeg', 'f683b7037a150e25_20250417_145736.jpeg', 'f683b7037a150e25_20250417_160317.jpeg', 'f6ee65bbfe84811b_20250415_114838.jpeg', 'f6ee65bbfe84811b_20250415_131525.jpeg', 'f6ee65bbfe84811b_20250415_133331.jpeg', 'f6efc915099401d5_20250304_124255.jpeg', 'f72e1edc12788a32_20250418_093525.jpeg', 'f76b106715c00418_20250420_142757.jpeg', 'f7c823236cb5c4d7_20250416_115342.jpeg', 'f7c823236cb5c4d7_20250416_120706.jpeg', 'f7c823236cb5c4d7_20250416_122909.jpeg', 'f7c823236cb5c4d7_20250416_132625.jpeg', 'f7c823236cb5c4d7_20250416_140128.jpeg', 'f878e506e98661f2_20250407_141920.jpeg', 'f878e506e98661f2_20250414_130143.jpeg', 'f878e506e98661f2_20250416_114752.jpeg', 'f878e506e98661f2_20250416_125756.jpeg', 'f878e506e98661f2_20250417_104044.jpeg', 'f878e506e98661f2_20250417_114051.jpeg', 'f878e506e98661f2_20250417_122154.jpeg', 'f878e506e98661f2_20250417_181803.jpeg', 'f878e506e98661f2_20250417_202014.jpeg', 'f878e506e98661f2_20250418_130952.jpeg', 'f878e506e98661f2_20250418_134419.jpeg', 'f878e506e98661f2_20250418_141952.jpeg', 'f878e506e98661f2_20250418_164324.jpeg', 'f878e506e98661f2_20250419_121227.jpeg', 'f878e506e98661f2_20250420_121406.jpeg', 'f878e506e98661f2_20250420_125550.jpeg', 'f878e506e98661f2_20250420_132619.jpeg', 'f878e506e98661f2_20250420_165004.jpeg', 'f878e506e98661f2_20250420_172949.jpeg', 'fa34faf11db1c08b_20250418_153026.jpeg', 'fa34faf11db1c08b_20250420_104306.jpeg', 'fa34faf11db1c08b_20250420_174411.jpeg', 'fa527c12043369d6_20250411_123445.jpeg', 'fa527c12043369d6_20250419_183117.jpeg', 'fa527c12043369d6_20250420_132524.jpeg', 'fa527c12043369d6_20250420_171504.jpeg', 'fa527c12043369d6_20250420_181546.jpeg', 'fa884dc6f35ad91e_20250418_171003.jpeg', 'fb071b665c82ed86_20250404_110349.jpeg', 'fc867f765921d92d_20250405_180125.jpeg', 'fc867f765921d92d_20250406_131526.jpeg', 'fc867f765921d92d_20250406_162251.jpeg', 'fc867f765921d92d_20250406_185429.jpeg', 'fc867f765921d92d_20250407_120759.jpeg', 'fc867f765921d92d_20250414_124254.jpeg', 'fc867f765921d92d_20250414_125634.jpeg', 'fc867f765921d92d_20250415_174740.jpeg', 'fc867f765921d92d_20250416_162302.jpeg', 'fc867f765921d92d_20250416_171932.jpeg', 'fc867f765921d92d_20250417_181622.jpeg', 'fc867f765921d92d_20250417_182919.jpeg', 'fc867f765921d92d_20250417_184411.jpeg', 'fc867f765921d92d_20250417_191406.jpeg', 'fc867f765921d92d_20250418_145811.jpeg', 'fc867f765921d92d_20250418_173344.jpeg', 'fc867f765921d92d_20250419_171421.jpeg', 'fc867f765921d92d_20250419_172633.jpeg', 'fc867f765921d92d_20250419_181312.jpeg', 'fc867f765921d92d_20250419_183545.jpeg', 'fc867f765921d92d_20250419_191035.jpeg', 'fc867f765921d92d_20250420_175512.jpeg', 'fc867f765921d92d_20250420_182150.jpeg', 'fc867f765921d92d_20250420_185125.jpeg', 'fc867f765921d92d_20250420_193705.jpeg', 'fcdcf4fd0aae37ed_20250304_183001.jpeg', 'fd2db96e3a3f50ca_20250416_124827.jpeg', 'fd2db96e3a3f50ca_20250416_141042.jpeg', 'fd2db96e3a3f50ca_20250416_153745.jpeg', 'fd2db96e3a3f50ca_20250416_170737.jpeg', 'fd2db96e3a3f50ca_20250417_180930.jpeg', 'fd2db96e3a3f50ca_20250417_183730.jpeg', 'fd2db96e3a3f50ca_20250417_192727.jpeg', 'fd2db96e3a3f50ca_20250418_171406.jpeg', 'fd2db96e3a3f50ca_20250419_111336.jpeg', 'fd2db96e3a3f50ca_20250419_113629.jpeg', 'fd2db96e3a3f50ca_20250419_114754.jpeg', 'fd2db96e3a3f50ca_20250419_123850.jpeg', 'fd2db96e3a3f50ca_20250419_183433.jpeg', 'fd2db96e3a3f50ca_20250420_111640.jpeg', 'fd2db96e3a3f50ca_20250420_114312.jpeg', 'fd6cbb27bd787c8f_20250419_133528.jpeg', 'fddd8fe479adb861_20250419_112251.jpeg', 'fe241968fe051a6e_20250411_104956.jpeg', 'fe241968fe051a6e_20250411_144800.jpeg', 'fe241968fe051a6e_20250412_182159.jpeg', 'fe241968fe051a6e_20250414_160630.jpeg', 'fe241968fe051a6e_20250416_141347.jpeg', 'fe241968fe051a6e_20250416_182253.jpeg', 'fe241968fe051a6e_20250417_115658.jpeg', 'fe241968fe051a6e_20250417_161937.jpeg', 'fe241968fe051a6e_20250418_114810.jpeg', 'fe241968fe051a6e_20250418_201423.jpeg', 'fe241968fe051a6e_20250419_205237.jpeg', 'fea5d2ae9d1d83c9_20250416_095842.jpeg', 'ff2bc18819a1f0ee_20250412_131201.jpeg', 'ff2bc18819a1f0ee_20250416_155641.jpeg', 'ff85e5e528c2bc1a_20250405_155131.jpeg', 'ff85e5e528c2bc1a_20250407_150758.jpeg', 'ff85e5e528c2bc1a_20250408_142835.jpeg', 'ff85e5e528c2bc1a_20250408_161737.jpeg', 'ff85e5e528c2bc1a_20250408_163059.jpeg', 'ff85e5e528c2bc1a_20250409_143257.jpeg', 'ff85e5e528c2bc1a_20250419_111435.jpeg']
✅ val: 50 images, 0 labels at D:\Atpug\terminal_seal_project\data\images\val
⚠️  Warning: No label files found in val directory: D:\Atpug\terminal_seal_project\data\images\val
⚠️  Warning: Missing label files for images: ['f1bf6ef29bf3a88d_20250419_172403.jpeg', 'f1bf6ef29bf3a88d_20250419_183725.jpegg', 'f1bf6ef29bf3a88d_20250419_192421.jpeg', 'f1c4767541a5d8be_20250417_160035.jpeg', 'f1c4767541a5d8be_20250419_153438.jpeg', 'f35c121b66fc94d6_20250404_135340.jpeg', 'f35c121b66fc94d6_20250411_123215.jpeg', 'f35c121b66fc94d6_20250411_124414.jpeg', 'f35c121b66fc94d6_20250413_105000.jpeg', 'f35c121b66fc94d6_20250414_154315.jpeg', 'f35c121b66fc94d6_20250416_120247.jpeg', 'f35c121b66fc94d6_20250416_122159.jpeg', 'f35c121b66fc94d6_20250417_123205.jpeg', 'f47a041d2233b915_20250304_144119.jpeg', 'f57d8451ce447144_20250419_161200.jpeg', 'f57d8451ce447144_20250420_110726.jpeg', 'f57d8451ce447144_20250420_135139.jpeg', 'f57d8451ce447144_20250420_171948.jpeg', 'f683b7037a150e25_20250417_133818.jpeg', 'f683b7037a150e25_20250417_145736.jpeg', 'f683b7037a150e25_20250417_160317.jpeg', 'f6ee65bbfe84811b_20250415_114838.jpeg', 'f6ee65bbfe84811b_20250415_131525.jpeg', 'f6ee65bbfe84811b_20250415_133331.jpeg', 'f6efc915099401d5_20250304_124255.jpeg', 'f72e1edc12788a32_20250418_093525.jpeg', 'f76b106715c00418_20250420_142757.jpeg', 'f7c823236cb5c4d7_20250416_115342.jpeg', 'f7c823236cb5c4d7_20250416_120706.jpeg', 'f7c823236cb5c4d7_20250416_122909.jpeg', 'f7c823236cb5c4d7_20250416_132625.jpeg', 'f7c823236cb5c4d7_20250416_140128.jpeg', 'f878e506e98661f2_20250407_141920.jpeg', 'f878e506e98661f2_20250414_130143.jpeg', 'f878e506e98661f2_20250416_114752.jpeg', 'f878e506e98661f2_20250416_125756.jpeg', 'f878e506e98661f2_20250417_104044.jpeg', 'f878e506e98661f2_20250417_114051.jpeg', 'f878e506e98661f2_20250417_122154.jpeg', 'f878e506e98661f2_20250417_181803.jpeg', 'f878e506e98661f2_20250417_202014.jpeg', 'f878e506e98661f2_20250418_130952.jpeg', 'f878e506e98661f2_20250418_134419.jpeg', 'f878e506e98661f2_20250418_141952.jpeg', 'f878e506e98661f2_20250418_164324.jpeg', 'f878e506e98661f2_20250419_121227.jpeg', 'f878e506e98661f2_20250420_121406.jpeg', 'f878e506e98661f2_20250420_125550.jpeg', 'f878e506e98661f2_20250420_132619.jpeg', 'f878e506e98661f2_20250420_165004.jpeg']
✅ test: 62 images, 0 labels at D:\Atpug\terminal_seal_project\data\images\test
⚠️  Warning: No label files found in test directory: D:\Atpug\terminal_seal_project\data\images\test
⚠️  Warning: Missing label files for images: ['0a7d8ab0113b68e0_20250412_164733.jpeg', '0a7d8ab0113b68e0_20250412_165050.jpegg', '0a7d8ab0113b68e0_20250412_165414.jpeg', '0a7d8ab0113b68e0_20250412_170754.jpeg', '0a7d8ab0113b68e0_20250412_170954.jpeg', '0a7d8ab0113b68e0_20250412_171143.jpeg', '0a7d8ab0113b68e0_20250412_171342.jpeg', '0a7d8ab0113b68e0_20250412_171602.jpeg', '0a7d8ab0113b68e0_20250412_171806.jpeg', '0a7d8ab0113b68e0_20250412_185409.jpeg', '0a7d8ab0113b68e0_20250412_185817.jpeg', '0a7d8ab0113b68e0_20250412_191106.jpeg', '0a7d8ab0113b68e0_20250412_191751.jpeg', '0a7d8ab0113b68e0_20250412_192955.jpeg', '0a7d8ab0113b68e0_20250412_193214.jpeg', '0a7d8ab0113b68e0_20250414_212525.jpg', '0a7d8ab0113b68e0_20250414_212747.jpg', '0a7d8ab0113b68e0_20250414_213003.jpg', '0a7d8ab0113b68e0_20250414_213236.jpg', '0a7d8ab0113b68e0_20250414_213449.jpg', '0a7d8ab0113b68e0_20250414_213657.jpg', '0a7d8ab0113b68e0_20250414_214132.jpg', '0a7d8ab0113b68e0_20250414_214348.jpg', '0a7d8ab0113b68e0_20250414_214547.jpg', '0a7d8ab0113b68e0_20250414_215244.jpg', '0a7d8ab0113b68e0_20250414_215443.jpg', '0a8679f7e9f76eee_20250412_141219.jpeg', '0aac5d7b77d66b6b_20250412_165403.jpg', '0aac5d7b77d66b6b_20250412_173945.jpg', '0d4f5ea21e3c609c_20250412_104643.jpeg', '0d4f5ea21e3c609c_20250412_111011.jpeg', '0d4f5ea21e3c609c_20250412_114209.jpeg', '0d4f5ea21e3c609c_20250412_122310.jpeg', '0d4f5ea21e3c609c_20250412_125150.jpeg', '0d4f5ea21e3c609c_20250412_133536.jpeg', '0d4f5ea21e3c609c_20250412_144029.jpeg', '0d4f5ea21e3c609c_20250412_150539.jpeg', '0d4f5ea21e3c609c_20250412_164356.jpeg', '0d4f5ea21e3c609c_20250412_171646.jpeg', '0d4f5ea21e3c609c_20250412_181407.jpeg', '0d4f5ea21e3c609c_20250412_183950.jpeg', '0d4f5ea21e3c609c_20250412_185723.jpeg', '0d4f5ea21e3c609c_20250412_192337.jpeg', 'ef369392e047fb50_20250418_161310.jpeg', 'ef67061ede767296_20250410_114714.jpeg', 'ef9da9c2c49b820c_20250416_101302.jpeg', 'f1bf6ef29bf3a88d_20250407_202319.jpeg', 'f1bf6ef29bf3a88d_20250410_185738.jpeg', 'f1bf6ef29bf3a88d_20250410_191459.jpeg', 'f1bf6ef29bf3a88d_20250411_144443.jpeg', 'f1bf6ef29bf3a88d_20250411_183718.jpeg', 'f1bf6ef29bf3a88d_20250411_185558.jpeg', 'f1bf6ef29bf3a88d_20250415_150322.jpeg', 'f1bf6ef29bf3a88d_20250417_141644.jpeg', 'f1bf6ef29bf3a88d_20250418_171649.jpeg', 'f1bf6ef29bf3a88d_20250418_173410.jpeg', 'f1bf6ef29bf3a88d_20250418_174626.jpeg', 'f1bf6ef29bf3a88d_20250418_190622.jpeg', 'f1bf6ef29bf3a88d_20250418_213453.jpeg', 'f1bf6ef29bf3a88d_20250419_155640.jpeg', 'f1bf6ef29bf3a88d_20250419_164550.jpeg', 'f1bf6ef29bf3a88d_20250419_170740.jpeg']
✅ Dataset verification passed!
📦 Loading model: yolov8n.pt
🚀 Starting training...
📊 Training parameters:
   - Epochs: 100
   - Batch size: 8
   - Image size: 640
   - Device: cpu
   - Learning rate: 0.01 -> 0.01
   - Patience: 20
Ultralytics 8.3.169  Python-3.10.16 torch-2.7.1+cpu CPU (AMD Ryzen 5 7530U with Radeon Graphics)
engine\trainer: agnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=8, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=data/dataset.yaml, degrees=0.0, deterministic=True, device=cpu, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8n.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=exp10, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=20, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=runs/train, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=runs\train\exp10, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 129 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs

Transferred 319/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
train: Fast image access  (ping: 0.10.0 ms, read: 52.127.1 MB/s, size: 739.8 KB)
train: Scanning D:\Atpug\terminal_seal_project\data\labels\train... 148 images, 45 backgrounds, 0 corrupt: 100%|██████████| 
train: New cache created: D:\Atpug\terminal_seal_project\data\labels\train.cache
C:\Users\yc993\.conda\envs\asstes\lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
val: Fast image access  (ping: 0.10.0 ms, read: 35.79.0 MB/s, size: 465.7 KB)
val: Scanning D:\Atpug\terminal_seal_project\data\labels\val... 46 images, 4 backgrounds, 0 corrupt: 100%|██████████| 50/50 
val: New cache created: D:\Atpug\terminal_seal_project\data\labels\val.cache
C:\Users\yc993\.conda\envs\asstes\lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
Plotting labels to runs\train\exp10\labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: AdamW(lr=0.002, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 0 dataloader workers
Logging results to runs\train\exp10
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100         0G      2.353      3.663       2.06          1        640: 100%|██████████| 25/25 [01:10<00:00,  2.81s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:08<00:00, 
                   all         50         89     0.0056      0.944      0.282     0.0841

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100         0G      2.295      3.001      2.036          3        640: 100%|██████████| 25/25 [01:09<00:00,  2.78s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:08<00:00,
                   all         50         89      0.485      0.381      0.339     0.0985

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100         0G      2.111       3.09       1.85          0        640: 100%|██████████| 25/25 [01:10<00:00,  2.82s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:07<00:00, 
                   all         50         89      0.288      0.169      0.215     0.0582

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100         0G      2.279      2.707      2.008          3        640: 100%|██████████| 25/25 [01:11<00:00,  2.85s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:07<00:00,
                   all         50         89      0.396       0.42      0.305     0.0744

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100         0G      2.233      2.916      1.957          0        640: 100%|██████████| 25/25 [01:11<00:00,  2.86s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:08<00:00, 
                   all         50         89      0.319      0.483      0.328     0.0955

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100         0G      2.265      2.534      2.052          2        640: 100%|██████████| 25/25 [01:14<00:00,  2.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:07<00:00, 
                   all         50         89      0.304      0.449      0.288     0.0987

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100         0G        2.2      2.384      1.963          5        640: 100%|██████████| 25/25 [01:12<00:00,  2.90s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:07<00:00, 
                   all         50         89      0.384      0.494      0.319     0.0991

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100         0G      2.107      2.623      1.899          0        640: 100%|██████████| 25/25 [01:13<00:00,  2.96s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:07<00:00, 
                   all         50         89      0.241       0.52      0.251     0.0947

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100         0G      2.249      2.452      2.001          2        640: 100%|██████████| 25/25 [01:18<00:00,  3.15s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:08<00:00, 
                   all         50         89      0.381      0.461      0.356      0.105

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100         0G      2.235      2.411      2.055          6        640: 100%|██████████| 25/25 [01:14<00:00,  2.99s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:07<00:00, 
                   all         50         89      0.535      0.472      0.506      0.162

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100         0G      2.168      2.296       2.01          2        640: 100%|██████████| 25/25 [01:13<00:00,  2.93s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:07<00:00, 
                   all         50         89      0.686      0.491      0.591      0.235

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100         0G      2.245      2.296      2.048          3        640: 100%|██████████| 25/25 [29:01<00:00, 69.66s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.557      0.551      0.578      0.211

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100         0G      2.159      2.168       1.98          3        640: 100%|██████████| 25/25 [00:54<00:00,  2.19s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.519      0.573      0.525      0.192

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100         0G       2.12      2.093      2.046          2        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.554      0.596      0.556      0.191

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100         0G      2.148      2.181      1.979          1        640: 100%|██████████| 25/25 [00:58<00:00,  2.35s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89       0.49      0.539      0.463      0.163

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100         0G      2.059      2.113      1.865          2        640: 100%|██████████| 25/25 [00:50<00:00,  2.01s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.571      0.659      0.667      0.257

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100         0G      2.067       2.05      1.858          4        640: 100%|██████████| 25/25 [00:49<00:00,  1.98s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.614      0.573      0.602      0.198

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100         0G      2.062      1.959      1.894          1        640: 100%|██████████| 25/25 [00:49<00:00,  1.99s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.703      0.685      0.702      0.274

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100         0G      2.033      1.953      1.873          2        640: 100%|██████████| 25/25 [00:49<00:00,  1.96s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.644      0.708      0.629      0.211

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100         0G      2.027      1.953      1.867          2        640: 100%|██████████| 25/25 [00:48<00:00,  1.95s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.673      0.685      0.594      0.204

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100         0G      2.045      1.959       1.84          2        640: 100%|██████████| 25/25 [00:49<00:00,  2.00s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.706      0.703      0.702      0.252

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100         0G      1.891      1.911      1.803          0        640: 100%|██████████| 25/25 [00:50<00:00,  2.00s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.565      0.708      0.634      0.255

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100         0G      2.086      1.954      1.971          3        640: 100%|██████████| 25/25 [00:50<00:00,  2.01s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.632      0.798      0.734      0.304

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100         0G      1.934      1.833      1.789          3        640: 100%|██████████| 25/25 [00:51<00:00,  2.05s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00, 
                   all         50         89      0.629      0.743      0.701      0.298

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100         0G      1.937      1.722      1.781          1        640: 100%|██████████| 25/25 [00:49<00:00,  1.99s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89       0.62      0.764      0.687      0.299

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100         0G      1.927      1.806      1.742          0        640: 100%|██████████| 25/25 [00:49<00:00,  1.96s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.679      0.764      0.752      0.329

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100         0G      1.865      1.828      1.736          0        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.732      0.735      0.745      0.329

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100         0G      1.995      1.786      1.879          4        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.763      0.719      0.734      0.326

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100         0G      1.898      1.693      1.811          2        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.701      0.686      0.722      0.312

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100         0G      1.873      1.724      1.733          1        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.747      0.775      0.782      0.338

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100         0G       1.86      1.597      1.731          1        640: 100%|██████████| 25/25 [00:50<00:00,  2.02s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.742      0.685      0.736      0.283

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100         0G      1.946      1.666      1.767          3        640: 100%|██████████| 25/25 [00:50<00:00,  2.03s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.731      0.764      0.791      0.343

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100         0G      1.826      1.587      1.704          4        640: 100%|██████████| 25/25 [00:48<00:00,  1.94s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.779      0.833      0.847      0.376

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100         0G      1.851      1.574      1.761          4        640: 100%|██████████| 25/25 [00:48<00:00,  1.94s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.766      0.719      0.782      0.366

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100         0G      1.878      1.629      1.785         10        640: 100%|██████████| 25/25 [00:48<00:00,  1.95s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.754      0.742      0.812      0.388

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100         0G      1.848      1.597      1.736          5        640: 100%|██████████| 25/25 [00:51<00:00,  2.07s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.835       0.73      0.843      0.384

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100         0G      1.747      1.669      1.683          0        640: 100%|██████████| 25/25 [00:49<00:00,  1.98s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.768      0.782      0.833      0.407

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100         0G      1.848       1.57      1.757          1        640: 100%|██████████| 25/25 [00:48<00:00,  1.95s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.807       0.75      0.831       0.42

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100         0G      1.797      1.501      1.675          2        640: 100%|██████████| 25/25 [00:48<00:00,  1.96s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.741      0.787      0.805      0.395

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100         0G      1.818      1.562      1.696          3        640: 100%|██████████| 25/25 [00:48<00:00,  1.93s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.766      0.685      0.799      0.375

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100         0G       1.83      1.569      1.702          5        640: 100%|██████████| 25/25 [00:51<00:00,  2.04s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.687      0.719      0.755      0.332

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100         0G      1.816      1.528      1.741          6        640: 100%|██████████| 25/25 [00:54<00:00,  2.17s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.872       0.73      0.837      0.399

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100         0G      1.749      1.463      1.701          2        640: 100%|██████████| 25/25 [00:49<00:00,  2.00s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.769      0.775      0.835      0.399

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100         0G      1.757      1.436      1.635          3        640: 100%|██████████| 25/25 [00:49<00:00,  1.98s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89       0.72      0.865      0.843      0.423

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100         0G      1.689      1.397      1.646          3        640: 100%|██████████| 25/25 [00:49<00:00,  1.98s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.824      0.841       0.87      0.416

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100         0G      1.732      1.392      1.601          4        640: 100%|██████████| 25/25 [00:49<00:00,  2.00s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.839      0.831      0.888       0.45

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100         0G      1.711      1.342       1.61          1        640: 100%|██████████| 25/25 [00:48<00:00,  1.96s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.779      0.764      0.787      0.367

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100         0G      1.677      1.371      1.596          1        640: 100%|██████████| 25/25 [00:48<00:00,  1.93s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.872      0.831      0.885       0.49

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100         0G      1.662      1.386       1.58          4        640: 100%|██████████| 25/25 [00:48<00:00,  1.92s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89       0.84      0.809      0.904      0.511

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100         0G      1.624      1.348      1.579          2        640: 100%|██████████| 25/25 [00:48<00:00,  1.93s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.829      0.869       0.91      0.504

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100         0G      1.677      1.371      1.579          3        640: 100%|██████████| 25/25 [00:48<00:00,  1.95s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.826      0.802      0.885      0.477

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100         0G      1.669      1.322      1.605          4        640: 100%|██████████| 25/25 [00:48<00:00,  1.93s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.882      0.831      0.934      0.491

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100         0G      1.563      1.315      1.516          3        640: 100%|██████████| 25/25 [00:50<00:00,  2.01s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.914       0.84      0.911      0.489

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100         0G      1.579      1.297      1.535          3        640: 100%|██████████| 25/25 [00:48<00:00,  1.95s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.832      0.843       0.91       0.52

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100         0G      1.683      1.293      1.612          2        640: 100%|██████████| 25/25 [00:48<00:00,  1.95s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.817      0.853      0.915      0.504

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100         0G      1.525       1.18      1.532          5        640: 100%|██████████| 25/25 [00:48<00:00,  1.93s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.895       0.82       0.93      0.523

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100         0G      1.646       1.32      1.575          4        640: 100%|██████████| 25/25 [00:48<00:00,  1.95s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.927      0.809      0.928      0.557

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100         0G      1.611      1.329      1.565          1        640: 100%|██████████| 25/25 [00:49<00:00,  1.99s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.929      0.843      0.945      0.569

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100         0G      1.547      1.266      1.547          2        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.906      0.863      0.943      0.559

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100         0G      1.552      1.192      1.506          4        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.895      0.854      0.914      0.532

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100         0G      1.528      1.166      1.517          2        640: 100%|██████████| 25/25 [00:48<00:00,  1.96s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.856      0.869      0.934      0.573

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100         0G      1.674       1.18      1.591          2        640: 100%|██████████| 25/25 [00:48<00:00,  1.96s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.908      0.883      0.942      0.574

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100         0G      1.606      1.185      1.537          8        640: 100%|██████████| 25/25 [00:49<00:00,  1.98s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89       0.92      0.865      0.954      0.582

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100         0G      1.546      1.144      1.516          3        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.942      0.909       0.97      0.633

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100         0G      1.444      1.215      1.419          0        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.965       0.82       0.96      0.619

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100         0G      1.502      1.122      1.488          2        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.891      0.916      0.956      0.616

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100         0G      1.451      1.119      1.453          5        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.844       0.91      0.947      0.634

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100         0G      1.504      1.149       1.43          3        640: 100%|██████████| 25/25 [00:49<00:00,  1.98s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89       0.86      0.888      0.944      0.605

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100         0G      1.427      1.049      1.411          3        640: 100%|██████████| 25/25 [00:49<00:00,  1.99s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.906      0.869      0.956      0.619

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100         0G      1.468      1.082      1.467          1        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.888      0.888      0.954      0.602

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100         0G      1.446     0.9843      1.431          3        640: 100%|██████████| 25/25 [00:49<00:00,  1.99s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.912      0.899      0.966      0.644

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100         0G      1.417      1.062      1.415          5        640: 100%|██████████| 25/25 [00:50<00:00,  2.00s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.868      0.933      0.959      0.639

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100         0G      1.411      1.074      1.407          1        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.901      0.919      0.955       0.62

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100         0G      1.401      1.041      1.408          4        640: 100%|██████████| 25/25 [00:48<00:00,  1.95s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.921      0.917      0.964      0.638

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100         0G      1.452      1.075      1.412          8        640: 100%|██████████| 25/25 [00:48<00:00,  1.95s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.932      0.955      0.973      0.679

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100         0G      1.394      1.001      1.373          5        640: 100%|██████████| 25/25 [00:48<00:00,  1.94s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.952      0.944      0.978      0.696

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100         0G      1.369     0.9613      1.366          4        640: 100%|██████████| 25/25 [00:48<00:00,  1.94s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.955      0.955      0.983        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100         0G      1.365     0.9795      1.367          1        640: 100%|██████████| 25/25 [00:48<00:00,  1.94s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.923      0.966      0.979      0.689

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100         0G      1.372      0.959       1.35          5        640: 100%|██████████| 25/25 [00:48<00:00,  1.94s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.965      0.942      0.979      0.693

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100         0G      1.293     0.9873      1.331          5        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.965      0.923       0.98      0.704

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100         0G      1.282     0.9366      1.311          2        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89       0.97      0.933      0.979      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100         0G      1.319     0.9416      1.341          5        640: 100%|██████████| 25/25 [00:49<00:00,  1.98s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.959      0.921      0.976      0.689

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100         0G      1.284     0.8906       1.32          3        640: 100%|██████████| 25/25 [00:48<00:00,  1.96s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.975      0.921      0.978       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100         0G      1.338      1.013       1.34          1        640: 100%|██████████| 25/25 [00:48<00:00,  1.94s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.965      0.926      0.978      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100         0G      1.293     0.8993      1.304          3        640: 100%|██████████| 25/25 [00:51<00:00,  2.04s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.962      0.921      0.979      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/100         0G      1.329     0.9453       1.32          4        640: 100%|██████████| 25/25 [00:48<00:00,  1.93s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.981      0.921      0.979      0.722

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/100         0G      1.344     0.9534      1.364          1        640: 100%|██████████| 25/25 [00:48<00:00,  1.95s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.984      0.921       0.98      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/100         0G      1.176     0.8553      1.248          2        640: 100%|██████████| 25/25 [00:48<00:00,  1.96s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.945      0.962      0.982      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/100         0G      1.271     0.9453      1.302          1        640: 100%|██████████| 25/25 [00:47<00:00,  1.92s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.966      0.951       0.98      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/100         0G      1.213      0.879      1.261          4        640: 100%|██████████| 25/25 [00:48<00:00,  1.93s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.944      0.948      0.979        0.7
Closing dataloader mosaic
C:\Users\yc993\.conda\envs\asstes\lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/100         0G      1.271     0.8685      1.345          1        640: 100%|██████████| 25/25 [00:49<00:00,  1.97s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89          1      0.921      0.983      0.722

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/100         0G      1.217      0.842      1.323          4        640: 100%|██████████| 25/25 [00:48<00:00,  1.93s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.977      0.954      0.984      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/100         0G      1.194     0.7661       1.28          1        640: 100%|██████████| 25/25 [00:47<00:00,  1.91s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.996      0.944      0.984      0.738

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/100         0G       1.16     0.7726      1.281          3        640: 100%|██████████| 25/25 [00:47<00:00,  1.91s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.996      0.944      0.984       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/100         0G      1.134     0.7706      1.269          2        640: 100%|██████████| 25/25 [00:47<00:00,  1.91s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.995      0.944      0.984      0.743

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/100         0G       1.11     0.7357      1.259          2        640: 100%|██████████| 25/25 [00:47<00:00,  1.91s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89          1      0.944      0.984      0.751

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/100         0G      1.081     0.7979      1.196          0        640: 100%|██████████| 25/25 [00:48<00:00,  1.93s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89          1      0.942      0.984      0.747

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100         0G      1.076        0.7      1.204          2        640: 100%|██████████| 25/25 [00:48<00:00,  1.92s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.995      0.944      0.984      0.748

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100         0G       1.04     0.7149      1.212          2        640: 100%|██████████| 25/25 [00:48<00:00,  1.92s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00, 
                   all         50         89      0.966      0.965      0.985      0.752

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100         0G      1.062     0.7341      1.211          5        640: 100%|██████████| 25/25 [00:49<00:00,  2.00s/i
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:05<00:00, 
                   all         50         89      0.966      0.966      0.985      0.753

100 epochs completed in 2.057 hours.
Optimizer stripped from runs\train\exp10\weights\last.pt, 6.2MB
Optimizer stripped from runs\train\exp10\weights\best.pt, 6.2MB

Validating runs\train\exp10\weights\best.pt...
Ultralytics 8.3.169  Python-3.10.16 torch-2.7.1+cpu CPU (AMD Ryzen 5 7530U with Radeon Graphics)
Model summary (fused): 72 layers, 3,005,843 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:04<00:00,
                   all         50         89      0.966      0.967      0.985      0.753
Speed: 1.2ms preprocess, 54.6ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to runs\train\exp10
✅ Training completed successfully!
🔍 Validating model...
Ultralytics 8.3.169  Python-3.10.16 torch-2.7.1+cpu CPU (AMD Ryzen 5 7530U with Radeon Graphics)
Model summary (fused): 72 layers, 3,005,843 parameters, 0 gradients, 8.1 GFLOPs
val: Fast image access  (ping: 0.00.0 ms, read: 1492.6197.4 MB/s, size: 328.1 KB)
val: Scanning D:\Atpug\terminal_seal_project\data\labels\val.cache... 46 images, 4 backgrounds, 0 corrupt: 100%|██████████| 
C:\Users\yc993\.conda\envs\asstes\lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 7/7 [00:04<00:00, 
                   all         50         89      0.966      0.967      0.985      0.753
Speed: 1.0ms preprocess, 47.4ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to runs\train\exp102
✅ Validation completed!
🧪 Testing model...
Ultralytics 8.3.169  Python-3.10.16 torch-2.7.1+cpu CPU (AMD Ryzen 5 7530U with Radeon Graphics)
val: Fast image access  (ping: 0.00.0 ms, read: 51.445.9 MB/s, size: 374.3 KB)
val: Scanning D:\Atpug\terminal_seal_project\data\labels\test... 34 images, 28 backgrounds, 0 corrupt: 100%|██████████| 62/6
val: New cache created: D:\Atpug\terminal_seal_project\data\labels\test.cache
C:\Users\yc993\.conda\envs\asstes\lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 8/8 [00:05<00:00, 
                   all         62         60      0.995      0.967      0.993      0.787
                   all         62         60      0.995      0.967      0.993      0.787
Speed: 0.8ms preprocess, 46.8ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to runs\train\exp103
✅ Testing completed!
💾 Model saved at: runs/train\exp\best.pt
📊 Metrics plot saved to runs/train\exp\metrics.png

📊 Training Summary:
==============================
mAP50: 0.9848
mAP50-95: 0.7530
Precision: 0.9663
Recall: 0.9666
Model saved at: runs/train\exp\best.pt
Results saved at: runs/train\exp

🎉 Training process completed!