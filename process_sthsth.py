import os
os.environ["OPENCV_LOG_LEVEL"]="SILENT"
import cv2
import json
from tqdm import tqdm

with open('/shared/data/anno/anno_downstream/ssv2_ret_label_train.json', 'r') as f:
    data = json.load(f)
with open('/shared/data/sthsth-v2/train.json','r') as f:
    train_data = json.load(f)

id2noun = {}
for elem in train_data:
    id2noun[elem['id']] = " ".join(elem['placeholders'])

# Extract one frame and make webm
DATADIR = '/shared/data/sthsth-v2/raw-videos/'

# Extract placeholder to make noun
def sample_frame(vpath):
    with open(vpath, "rb") as f:
        content = f.read()
    cap = cv2.VideoCapture(vpath)
    
    # Get the total number of frames
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_point = length//2 # Approximately half if number of frames are odd
    
    # Set the reader to the given frame number (half_point)
    cap.set(cv2.CAP_PROP_POS_FRAMES, half_point)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Release the file pointer
    cap.release()
    return frame
    
fourcc = cv2.VideoWriter_fourcc(*"VP90")
def save_webm(ipath, frame, num = 2):
    H, W, _ = frame.shape
    fps = 1
    out = cv2.VideoWriter(ipath, fourcc, fps, (W, H))
    for _ in range(num):
        out.write(frame)
    out.release()

def create_n_save(vpath, ipath):
    frame = sample_frame(vpath)
    save_webm(ipath, frame)

new_data = []
for elem in tqdm(data):
    vpath = DATADIR + elem['video']
    vkey = elem['video'].replace('.webm','')
    ipath = DATADIR + f'{vkey}_im.webm'
    # if os.path.exists(ipath):
    #     continue 
    try:
        # create_n_save(vpath, ipath)
        noun = id2noun[vkey]
        new_data.append(elem)
        new_data.append({'video': ipath,
                         'caption': noun})
    except Exception as e:
        print("** Error", ipath)

print(len(new_data))
with open('/shared/data/anno/anno_downstream/ssv2_ret_image_train.json', 'w') as f:
    json.dump(new_data, f)