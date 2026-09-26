from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
import json, os, shutil, zipfile, math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = ('swimmer', 'boat', 'jetski', 'life saving appliance', 'buoy')
COLORS = ('#ff595e', '#35baff', '#ffca3a', '#ab70ff', '#49dc93')
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(temp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def resolve_image(root, split, name):
    root = Path(root).resolve()
    name = name.replace('\\', '/')
    for candidate in (root / 'images' / split / name, root / name,
                      root / 'images' / split / Path(name).name):
        candidate = candidate.resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            return candidate
    raise FileNotFoundError(f'{split}: {name} under {root}')

def locate_dataset(settings):
    source = Path(settings['dataset_root']).expanduser()
    if (source / 'annotations/instances_train.json').is_file():
        return source.resolve()
    if source.exists():
        candidates = list(source.glob('*/annotations/instances_train.json'))
        if len(candidates) == 1:
            return candidates[0].parent.parent.resolve()
    archive = Path(settings['dataset_zip']).expanduser()
    if not archive.is_file():
        raise FileNotFoundError('Set DATASET_ROOT to the extracted folder, or DATASET_ZIP to Compressed Version.zip.')
    dest = Path(settings['root']) / 'dataset'
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = []
        for info in z.infolist():
            p = PurePosixPath(info.filename.replace('\\', '/'))
            if '__MACOSX' in p.parts or p.name.startswith('._') or info.is_dir():
                continue
            parts = p.parts
            image_member = any(parts[i:i+2] in (('images','train'), ('images','val')) for i in range(len(parts)-1))
            annotation_member = p.name in ('instances_train.json', 'instances_val.json') and 'annotations' in parts
            if image_member or annotation_member:
                target = (dest / str(p)).resolve()
                if not target.is_relative_to(dest.resolve()) or ':' in str(p):
                    raise ValueError('Unsafe ZIP path: ' + info.filename)
                members.append((info, target))
        if not members:
            raise ValueError('ZIP must contain annotations/instances_train.json and images/train, images/val.')
        needed = sum(info.file_size for info, p in members if not p.exists() or p.stat().st_size != info.file_size)
        if shutil.disk_usage(dest).free < needed + 12 * 2**30:
            raise OSError('Insufficient disk: extraction needs its uncompressed size plus 12 GiB for checkpoints.')
        for i, (info, target) in enumerate(members):
            if not target.exists() or target.stat().st_size != info.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_suffix(target.suffix + '.partial')
                with z.open(info) as src, temp.open('wb') as out:
                    shutil.copyfileobj(src, out)
                os.replace(temp, target)
            if i % 1000 == 0:
                print(f'Extracted/checked {i+1}/{len(members)}', flush=True)
    candidates = list(dest.rglob('instances_train.json'))
    if len(candidates) != 1:
        raise ValueError('Ambiguous annotation root after extraction; set DATASET_ROOT explicitly.')
    return candidates[0].parent.parent

def annotated(image, boxes, title='', scores=None, threshold=0.25, maximum=80):
    image = image.convert('RGB').copy()
    ratio = min(1., 1200 / max(image.size))
    image = image.resize((max(1, round(image.width*ratio)), max(1, round(image.height*ratio))))
    draw = ImageDraw.Draw(image)
    shown = 0
    for i, row in enumerate(boxes):
        x1,y1,x2,y2,label = row[:5]
        if scores is not None and scores[i] < threshold:
            continue
        if shown >= maximum:
            break
        color = COLORS[int(label)]
        xy = tuple(float(x)*ratio for x in (x1,y1,x2,y2))
        draw.rectangle(xy, outline=color, width=2)
        caption = NAMES[int(label)] + (f' {scores[i]:.2f}' if scores is not None else '')
        draw.text((max(0, xy[0]), max(0, xy[1]-12)), caption, fill=color, stroke_width=1, stroke_fill='black')
        shown += 1
    draw.rectangle((0,0,image.width,23), fill='black')
    draw.text((7,5), title, fill='white')
    return image

def montage(images, dest, columns=2):
    width = max(im.width for im in images)
    height = max(im.height for im in images)
    canvas = Image.new('RGB', (width*columns, height*math.ceil(len(images)/columns)), '#14212b')
    for i, im in enumerate(images):
        canvas.paste(im, ((i%columns)*width, (i//columns)*height))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)

def gt_boxes(coco, image_id):
    return [[*a['bbox'][:2], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3], a['category_id']-1]
            for a in coco['annotations'] if a['image_id']==image_id and a['category_id'] in range(1,6)]

def tile_windows(width, height, size=1536, overlap=0.25):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError('Invalid tile settings')
    def starts(length):
        if length <= size:
            return [0]
        stride = max(1, int(size*(1-overlap)))
        return sorted(set(list(range(0, length-size+1, stride)) + [length-size]))
    return [(x,y,min(x+size,width),min(y+size,height)) for y in starts(height) for x in starts(width)]

def prepare(settings):
    root = locate_dataset(settings)
    out = Path(settings['root']) / 'prepared'
    vis = Path(settings['root']) / 'visuals'
    out.mkdir(parents=True, exist_ok=True)
    vis.mkdir(parents=True, exist_ok=True)
    reports, docs, ids_by_split = {}, {}, {}
    for split, expected in [('train',8930), ('val',1547)]:
        d = read_json(root / f'annotations/instances_{split}.json')
        if len(d['images']) != expected:
            raise ValueError(f'{split}: expected {expected}, got {len(d["images"])}. Verify dataset version.')
        categories = {c['id']: c['name'].lower().replace('_',' ').strip() for c in d['categories']}
        for cid, expected_name in enumerate(NAMES, 1):
            if categories.get(cid, '').rstrip('s') != expected_name.rstrip('s'):
                raise ValueError(f'Unexpected category mapping: {categories}')
        image_ids = [im['id'] for im in d['images']]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError('Duplicate image IDs')
        ids_by_split[split] = set(image_ids)
        images = {im['id']: im for im in d['images']}
        for i, im in enumerate(d['images']):
            p = resolve_image(root, split, im['file_name'])
            with Image.open(p) as decoded:
                if decoded.size != (im['width'],im['height']):
                    raise ValueError(f'Image size disagrees with COCO: {p}')
            # Absolute names are portable because prepared JSONs are generated on each machine.
            im['file_name'] = str(p)
            if i % 2000 == 0:
                print(f'{split}: verified {i+1}/{len(images)} image headers', flush=True)
        annotations, counts, image_cats = [], Counter(), defaultdict(set)
        ignored = 0
        for a in d['annotations']:
            if a['category_id'] == 0:
                ignored += 1
                continue
            if a['category_id'] not in range(1,6) or a['image_id'] not in images:
                raise ValueError('Unknown category or image ID')
            x,y,w,h = a['bbox']
            if not np.isfinite([x,y,w,h]).all() or w <= 0 or h <= 0:
                raise ValueError(f'Invalid annotation {a["id"]}; inspect source, do not silently delete.')
            a.setdefault('iscrowd',0)
            a.setdefault('area',w*h)
            annotations.append(a)
            if not a.get('iscrowd') and not a.get('ignore'):
                counts[a['category_id']] += 1
                image_cats[a['image_id']].add(a['category_id'])
        if len({a['id'] for a in annotations}) != len(annotations):
            raise ValueError('Duplicate annotation IDs')
        d.update(annotations=annotations, categories=[dict(id=i+1,name=n) for i,n in enumerate(NAMES)])
        d.setdefault('info',{})
        save_json(out/f'{split}.json',d)
        docs[split] = d
        frequency = {cid:sum(cid in image_cats[k] for k in image_ids)/len(image_ids) for cid in range(1,6)}
        repeats = [max([1.] + [min(settings['repeat_cap'], (settings['repeat_threshold']/frequency[c])**0.5) for c in image_cats[k]]) for k in image_ids]
        reports[split] = dict(images=len(image_ids), objects={NAMES[c-1]:counts[c] for c in range(1,6)},
                              image_frequency={NAMES[c-1]:frequency[c] for c in range(1,6)},
                              excluded_category0=ignored, repeated_training_images=sum(math.ceil(r) for r in repeats) if split=='train' else None)
    if ids_by_split['train'] & ids_by_split['val']:
        raise ValueError('Train/validation image IDs overlap.')
    # Select fixed visualization examples covering all five classes, never used to tune hyperparameters.
    selected = []
    for cid in range(1,6):
        candidates = sorted({a['image_id'] for a in docs['val']['annotations'] if a['category_id']==cid})
        choice = next((i for i in candidates if i not in selected), candidates[0])
        selected.append(choice)
    save_json(out/'preview_ids.json', selected)
    save_json(out/'audit.json', reports)
    save_json(out/'resolved_paths.json', dict(dataset_root=str(root), train=str(out/'train.json'), val=str(out/'val.json')))
    frames = []
    for im in docs['val']['images']:
        if im['id'] in selected:
            with Image.open(im['file_name']) as img:
                frames.append(annotated(img,gt_boxes(docs['val'],im['id']),f'Ground truth | validation {im["id"]}'))
    montage(frames,vis/'01_ground_truth.jpg')
    fig,axes = plt.subplots(1,2,figsize=(14,5))
    for ax,split in zip(axes,('train','val')):
        ax.bar(NAMES,[reports[split]['objects'][n] for n in NAMES],color=COLORS)
        ax.set_yscale('log'); ax.tick_params(axis='x',rotation=25)
        ax.set_title(f'{split}: object counts (log scale)')
    fig.tight_layout(); fig.savefig(vis/'02_class_distribution.png'); plt.close(fig)
    print(json.dumps(reports,indent=2),flush=True)

def preprocessing_visuals(settings):
    import cv2
    root = Path(settings['root'])
    coco = read_json(root/'prepared/val.json')
    selected = read_json(root/'prepared/preview_ids.json')
    im = next(i for i in coco['images'] if i['id']==selected[0])
    image = np.asarray(Image.open(im['file_name']).convert('RGB'))
    lab = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    boxes = gt_boxes(coco,im['id'])
    montage([annotated(Image.fromarray(image),boxes,'Input: original pixels'),
             annotated(Image.fromarray(enhanced),boxes,'CLAHE preview only - disabled during training')],root/'visuals/03_enhancement_comparison.jpg')
    size = settings['train_scale']
    # MMCV keep_ratio uses long-edge and short-edge bounds, including portrait images.
    factor = min(max(size)/max(image.shape[:2]),min(size)/min(image.shape[:2]))
    resized = cv2.resize(image,(round(image.shape[1]*factor),round(image.shape[0]*factor)))
    normalized = (resized.astype(np.float32)-MEAN)/STD
    reconstructed = np.clip(normalized*STD+MEAN,0,255).astype(np.uint8)
    rb = [[x*factor,y*factor,x2*factor,y2*factor,c] for x,y,x2,y2,c in boxes]
    montage([annotated(Image.fromarray(resized),rb,f'Aspect-ratio resize: {resized.shape[1]} x {resized.shape[0]}'),
             annotated(Image.fromarray(reconstructed),rb,'Normalization reversed for display (RGB)')],root/'visuals/04_resize_normalization.jpg')
    overview = Image.fromarray(image).copy(); draw=ImageDraw.Draw(overview)
    for window in tile_windows(im['width'],im['height'],settings['tile_size']):
        draw.rectangle(window,outline='yellow',width=6)
    target = min(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
    cx,cy=(target[0]+target[2])/2,(target[1]+target[3])/2
    x=max(0,min(int(cx-384),im['width']-768)); y=max(0,min(int(cy-384),im['height']-768))
    zoom = Image.fromarray(image).crop((x,y,x+768,y+768))
    zb=[[a-x,b-y,c-x,d-y,k] for a,b,c,d,k in boxes if a>=x and b>=y and c<=x+768 and d<=y+768]
    montage([annotated(overview,boxes,'Label-free inference tile grid'),annotated(zoom,zb,'Ground-truth detail view (illustration only)')],root/'visuals/05_tiles_and_detail.jpg')
    steps=['Original image / verified COCO boxes', 'Capped rare-class image repetition (train only)',
           'Aspect-ratio resize + RGB normalization + pad /32',
           'Two Swin-S backbones with CBNetV2 composite connections',
           'Shared P2-P6 FPN (stride 4 included)', 'RPN: smaller anchors / RoIAlign',
           'Cascade boxes: IoU 0.5 -> 0.6 -> 0.7', 'Original-coordinate validation / optional TTA + tiles']
    fig,ax=plt.subplots(figsize=(12,10)); ax.axis('off')
    for i,label in enumerate(steps):
        y=0.94-i*0.12
        ax.text(.5,y,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#dceeff',ec='#126db1'),fontsize=12,transform=ax.transAxes)
        if i<len(steps)-1: ax.annotate('',xy=(.5,y-.075),xytext=(.5,y-.025),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    fig.savefig(root/'visuals/06_architecture.png',bbox_inches='tight'); plt.close(fig)
