import argparse
import yaml

import cv2
import torch
from torch.autograd import Variable

from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights


def main():
    """
    Visualize the detection result for the given image and the pre-trained model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg')
    parser.add_argument('--ckpt', type=str,
                        help='path to the checkpoint file')
    parser.add_argument('--weights_path', type=str,
                        default=None, help='path to weights file')
    parser.add_argument('--image', type=str)
    parser.add_argument('--background', action='store_true',
                        default=False, help='background(no-display mode. save "./output.png")')
    parser.add_argument('--detect_thresh', type=float,
                        default=None, help='confidence threshold')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)

    imgsize = cfg['TEST']['IMGSIZE']
    model = YOLOv3()
    confthre = cfg['TEST']['CONFTHRE'] 
    nmsthre = cfg['TEST']['NMSTHRE']

    if args.detect_thresh:
        confthre = args.detect_thresh

    img = cv2.imread(args.image)
    img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
    img, info_img = preprocess(img, imgsize)  # info = (h, w, nh, nw, dx, dy)
    img = torch.from_numpy(img).float().unsqueeze(0)

    if args.gpu >= 0:
        model.cuda(args.gpu)
        img = Variable(img.type(torch.cuda.FloatTensor))
    else:
        img = Variable(img.type(torch.FloatTensor))

    if args.weights_path:
        print("loading yolo weights %s" % (args.weights_path))
        parse_yolo_weights(model, args.weights_path)
    else:
        print("loading checkpoint %s" % (args.ckpt))
        model.load_state_dict(torch.load(args.ckpt))

    model.eval()

    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, 80, confthre, nmsthre)

    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()

    bboxes = list()
    classes = list()
    colors = list()

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:

        cls_id = coco_class_ids[int(cls_pred)]
        print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
        print('\t+ Label: %s, Conf: %.5f' %
              (coco_class_names[cls_id], cls_conf.item()))
        box = yolobox2label([y1, x1, y2, x2], info_img)
        bboxes.append(box)
        classes.append(cls_id)
        colors.append(coco_class_colors[int(cls_pred)])

    if args.background:
        import matplotlib
        matplotlib.use('Agg')

    from utils.vis_bbox import vis_bbox
    import matplotlib.pyplot as plt

    vis_bbox(
        img_raw, bboxes, label=classes, label_names=coco_class_names,
        instance_colors=colors, linewidth=2)
    plt.show()

    if args.background:
        plt.savefig('output.png')


if __name__ == '__main__':
    main()
