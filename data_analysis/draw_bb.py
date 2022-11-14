from PIL import Image, ImageDraw
import torch
import numpy as np


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y


img = Image.open("/hdd/hdd3/coco_fl/images/node_1_class_56/000000015906.jpg").convert(
    "RGB"
)
img.show()

draw = ImageDraw.Draw(img)
print(img.size)
f = open("/hdd/hdd3/coco_fl/labels/node_1_class_56/000000015906.txt", "r")
lines = f.readlines()
print(lines)
for line in lines:
    line = line.strip()
    line = line.split(" ")[1:]
    # print(line[0])
    line_f = list()
    for idx, i in enumerate(line):
        if idx % 2 == 0:
            line_f.append(float(i) * img.size[0])
        else:
            line_f.append(float(i) * img.size[1])
    print(line_f)
    line_xy = xywh2xyxy(line_f)
    # draw.rectangle([(float(line[0])*640,float(line[1])*640),(float(line[2])*640,float(line[3])*640)], outline=(0,255,0), width = 3)
    draw.rectangle(
        [(line_xy[0], line_xy[1]), (line_xy[2], line_xy[3])],
        outline=(0, 255, 0),
        width=3,
    )
f.close()


# draw = ImageDraw.Draw(img)
# draw.rectangle((100,100,300,300), outline=(0,255,0), width = 3)

img.save("test.jpg", "JPEG")
