d:
cd D:\yolov7\yolov7
call D:\yolov7\volov7-env3\Scripts\activate.bat
python detect-lucas-temp.py --weights runs\train\xtextile-temp-2026-10-013\weights\best.pt --conf-thres 0.5 --img-size 640 --source D:\yolov7\yolov7\inference\complete-input\all-axial --save-crops --crop-margin-px 20
pause