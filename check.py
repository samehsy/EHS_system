import torch
import  cv2

 
# # Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
dir = '/home/sameh/Desktop/graduation_project/EHS_system/yolov5/data/images/'
imgs = [dir + f for f in ('1.jpg','2.jpg','3.jpg','4.jpg' ,'5.jpg' )]  # batch of images

model.dnn = True
# model.source = '0'
# GPU Inference
model.cuda()
results = model('0')  # skip one
results = model(imgs)  # profile 
results.print()  # or .show(), .save()
# Speed: 3.1ms pre-process, 6.7ms inference, 1.5ms NMS per image at shape (2, 3, 640, 640)

# print('--------------------')
# # CPU Inference
# model.cpu()
# results = model(imgs)  # skip one
# results = model(imgs)  # profile 
# results.print()  # or .show(), .save()
# # Speed: 6.0ms pre-process, 369.7ms inference, 1.0ms NMS per image at shape (2, 3, 640, 640)


 