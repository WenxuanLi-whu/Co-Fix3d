# BEVFix& Co-Fix3d & Co-Fusion & Co-Stream

welcome to replicate my experiment if you have GPUs available. I used 4 NVIDIA 4090 GPUs, and I estimate that using 8 GPUs would yield even better results.

Co-Fix: [Paper]( https://arxiv.org/pdf/2408.07999)


nuScenes test dataset 
| Method |Modality |mAP|NDS | Ref
| ------------- | ------------- | ------------- | ------------- |------------- |
| BEVFix  | L  |68.0  |71.9 |[result](https://evalai.s3.amazonaws.com/media/submission_files/submission_329188/6354f132-8009-4615-8283-e528c1ac0fbd.json)|
| Co-Fix3D  | L  |69.4  |73.5 |[result](https://evalai.s3.amazonaws.com/media/submission_files/submission_481792/75e2db36-512e-49e3-9499-c0c6fd0f613f.json)|
| BEVFix(resnet-50)  | L+C |71.1  |73.2 |[result](https://evalai.s3.amazonaws.com/media/submission_files/submission_441293/eb01c8ed-00c4-48ae-88f5-8fca826b785b.json)|
| BEVFix(swin-t)  | L+C  |72.3 |74.1 |[result](https://evalai.s3.amazonaws.com/media/submission_files/submission_442241/af5d644f-484e-4ad1-84e7-ef651fdbde33.json)|
| Co-Fix3D(resnet-50)   | L+C  |72.3  |74.7  |[result](https://evalai.s3.amazonaws.com/media/submission_files/submission_482475/b8eb0314-2419-4533-b2d7-b006c35faf3e.json)||
| Co-Fusion  | L+C  |74.1  |75.3  |[result](https://evalai.s3.amazonaws.com/media/submission_files/submission_465446/00c01835-0166-42f2-80e5-fbf741ac32a0.json)||


nuScenes val dataset 
| Method |Modality |mAP|NDS |log |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| BEVFix  | L  |65.5  |70.0 ||
| Co-Fix3D  | L  |67.3  |72.0 |[log](https://drive.google.com/file/d/1awhWDqwUsKc08f3_4F874YV1brpC9S3k/view?usp=drive_link)|
| BEVFix  | L+C  |69.1  |72.1 ||
| Co-Fix3D  | L+C  |70.8  |73.6  ||
| Co-Fusion  | L+C  |73.5  |74.9  |[log](https://drive.google.com/file/d/1gdrjTm1l7gUpTvee13XJtB-YXp52makT/view?usp=drive_link)|
| Co-Stream  | L+C+T  |-  |-  ||

ToDo Multi-Frame Fusion: Co-Stream is coming soon

