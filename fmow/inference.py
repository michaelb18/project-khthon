import sys
sys.path.append('/home/michael/project_khthon/fmow')
from train_classifier import *
from dataloader import *
import sys
sys.path.append('/home/michael/project_khthon')
from get_data_from_eo import *
import torchvision

def get_rgb_image(image):
    img = image.sel(band=[3, 2, 1])
    b04 = img[0, :, :]  # Red band
    b03 = img[1, :, :]  # Green band
    b02 = img[2, :, :]  # Blue band

    # Stack into RGB image
    rgb_image = (np.stack([b04, b03, b02], axis=-1) * 255).astype(np.uint8)
    return rgb_image

def get_bboxes(image):
    from rasterio.transform import rowcol
    from pyproj import Transformer
    import cv2

    img = image.sel(band = [2, 3, 4, 5])
    channels = (np.stack([img[i, ...] for i in range(len(img))], axis=-1) * 255).astype(np.uint8)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(channels)

    ss.switchToSelectiveSearchFast()

    rects = ss.process()
    return rects

def get_top_k_categories(softmax_scores, categories, k):
    softmax_scores = np.array(softmax_scores)
    top_k_indices = np.argsort(softmax_scores)[::-1][:k]
    top_k_categories = [categories[i] for i in top_k_indices]
    top_k_scores = softmax_scores[top_k_indices]

    return top_k_categories, top_k_scores

def get_anchors():
    config = load_config('/home/michael/project_khthon/fmow/config.yaml')

    # Extract parameters with defaults
    args = type('Args', (), {
        'root_dir': config.get('root_dir', '/home/michael/project_khthon/fmow'),
        'output_dir': config.get('output_dir', './outputs'),
        'batch_size': config.get('batch_size', 32),
        'epochs': config.get('epochs', 20),
        'lr': config.get('lr', 1e-4),
        'num_workers': config.get('num_workers', 4),
        'model_name': config.get('model_name', 'Sentinel2_SwinT_SI_MS'),
        'infill_nulls': config.get('infill_nulls', True),
        'image_size': config.get('image_size', 224),
        'categories': config.get('categories', None),
        'max_samples_per_category': config.get('max_samples_per_category', None)
    })()
    train_loader = create_dataloader(
            root_dir=args.root_dir,
            split='train',
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            infill_nulls=args.infill_nulls,
            image_size=args.image_size,
            categories=args.categories,
            max_samples_per_category=50
        )

    return compute_anchors(train_loader, n_anchors = 1)

def iou(w1, h1, w2, h2):
    inter_w = min(w1, w2)
    inter_h = min(h1, h2)

    inter_area = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2

    # Union area
    union_area = float(area1 + area2 - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0

@torch.no_grad()
def inference_rcnn(model, image, iou_thresh = 0.5, confidence_threshold = 0.95, do_nms = True):
    categories = get_categories()

    anchors = get_anchors()
    transform = get_default_transform('val', normalize = False)

    rects = get_bboxes(image)

    # 3. Add boxes using patches
    # We'll just show the first 20 to keep the plot readable
    from copy import deepcopy
    rgb_image = deepcopy(get_rgb_image(image))
    image_w_text = deepcopy(rgb_image)
    min_val, max_val = np.percentile(image_w_text, (2, 98)) # Optional: Use percentiles
    scaled_band = np.clip(image_w_text, min_val, max_val)
    image_w_text = (scaled_band - min_val) / (max_val - min_val)
    boxes = []
    classes = []
    scores = []
    for i, rect in enumerate(tqdm(rects)):
        x, y, w, h = rect
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        img = np.stack([rgb_image[y1:y2, x1:x2, 0], rgb_image[y1:y2, x1:x2, 1], rgb_image[y1:y2, x1:x2, 2]], axis=0)
        img = torch.from_numpy(img)
        t = torch.unsqueeze(transform(img).to("cuda"), dim = 0)
        softmaxes = torch.nn.functional.softmax(model(t)).cpu().numpy()[0, ...]
        result = categories[np.argmax(softmaxes)]
        pseudoconfidence = softmaxes[np.argmax(softmaxes)]
        if pseudoconfidence > confidence_threshold:
            if iou(x2 - x1, y2 - y1, anchors[result][0][0], anchors[result][0][1]) > iou_thresh:
                boxes.append([x1, y1, x2, y2])
                classes.append(result)
                scores.append(pseudoconfidence)

    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    
    nms_boxes = []
    nms_scores = []
    nms_classes = []

    if do_nms:
        for category in categories:
            boxes_class = boxes[classes == category]
            scores_class = scores[classes == category]
            classes_class = classes[classes == category]
            if len(boxes) > 0:
                boxes_t = torch.from_numpy(boxes_class).float()
                scores_t = torch.from_numpy(scores_class)

                idxs = torchvision.ops.nms(boxes_t, scores_t, 0.0).cpu().numpy()

                boxes_class = boxes_class[idxs]
                scores_class = scores_class[idxs]
                classes_class = classes_class[idxs]
                
                nms_boxes.extend(boxes_class)
                nms_scores.extend(scores_class)
                nms_classes.extend(classes_class)
    else:
        nms_boxes.extend(boxes)
        nms_scores.extend(scores)
        nms_classes.extend(classes)

    return nms_boxes, nms_scores, nms_classes

def inference_location(lat, lon, model):
    image = get_image(lat, lon, 5000, 5000, days = 180)
    return inference_rcnn(model, image)

def inference_img(image, model, **kwargs):
    return inference_rcnn(model, image, **kwargs)

def get_model():
    model = get_foundation_model(20, model_name="resnet152", pretrained = True, in_channels = 3)
    model.load_state_dict(torch.load('/home/michael/project_khthon/fmow/large_objects/best_model.pth'))
    model.eval()
    return model

def get_categories():
    train_loader = create_dataloader(
            root_dir="/home/michael/project_khthon/fmow",
            split='train',
            batch_size=1,
            shuffle=True,
            num_workers=1,
            infill_nulls=False,
            image_size=224,
            categories=['airport',
                        'airport_terminal',
                        'amusement_park',
                        'aquaculture',
                        'crop_field',
                        'dam',
                        'golf_course',
                        'impoverished_settlement',
                        'interchange',
                        'lighthouse',
                        'nuclear_powerplant',
                        'port',
                        'railway_bridge',
                        'road_bridge',
                        'runway',
                        'solar_farm',
                        'space_facility',
                        'stadium',
                        'tunnel_opening',
                        'wind_farm'],
            max_samples_per_category=1
        )

    return train_loader.dataset.categories
