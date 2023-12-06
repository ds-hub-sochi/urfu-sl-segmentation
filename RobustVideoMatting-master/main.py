
import glob

from inference import convert_video


import torch
print(torch.cuda.is_available())

model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda() # or "resnet50"
#convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

# from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor
# from inference_utils import VideoReader, VideoWriter
# import cv2, time


# def test(video):
#         vid = cv2.VideoCapture(video)
#         height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
#         reader = VideoReader(video, transform=ToTensor())
#         name = video.split("\\")[-1]
#         writer = VideoWriter(f'D:\\download\\new_test\\{name}.mp4', frame_rate=30)
#         bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
#         rec = [None] * 4                                      # Initial recurrent states.
#         downsample_ratio = min(512 / max(height, width), 1)                        # Adjust based on your video.
#
#         with torch.no_grad():
#             for src in DataLoader(reader):
#                 # RGB tensor normalized to 0 ~ 1.
#                 fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio) # Cycle the recurrent states.
#                 com = fgr * pha + bgr * (1 - pha)              # Composite to green background.
#                 writer.write(com)

# reader = ImageSequenceReader('C:\\Users\\user\\Desktop\\supervisely_person_clean_2667_img\\supervisely_person_clean_2667_img\\temp', transform=ToTensor())
# writer = ImageSequenceWriter('C:\\Users\\user\\Desktop\\supervisely_person_clean_2667_img\\supervisely_person_clean_2667_img\\temp_mask')
#
# bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
# rec = [None] * 4                                      # Initial recurrent states.
# downsample_ratio = 0.6                             # Adjust based on your video.
#

# # with torch.no_grad():
#     for src in DataLoader(reader):
#         print(src)# RGB tensor normalized to 0 ~ 1.
#         fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio)  # Cycle the recurrent states.
#         #com = fgr * pha + bgr * (1 - pha)              # Composite to green background.
#         writer.write(pha)                               # Write frame.
#
# for video in glob.glob('D:\\download\\test\\*.mp4'):
#         print(video)
#         vid = cv2.VideoCapture(video)
#         height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
#         reader = VideoReader(video, transform=ToTensor())
#         name = video.split("\\")[-1]
#         writer = VideoWriter(f'D:\\download\\new_test\\{name}.mp4', frame_rate=30)
#         bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
#         rec = [None] * 4                                      # Initial recurrent states.
#         downsample_ratio = min(512 / max(height, width), 1)                        # Adjust based on your video.
#
#         with torch.no_grad():
#             for src in DataLoader(reader):
#                 # RGB tensor normalized to 0 ~ 1.
#                 fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio) # Cycle the recurrent states.
#                 com = fgr * pha + bgr * (1 - pha)              # Composite to green background.
#                 writer.write(com)


            # del rec, fgr, pha, com, vid, bgr, writer, video, reader
            # torch.cuda.empty_cache()
        # torch.cuda.memory_allocated()
#D:\download\new_test\1ed64727-a02d-43a9-ae4c-023407d757b0.mp4
flag = 0
i = 0
videos = glob.glob('D:\\download\\train\\*.mp4')
n = len(videos)
print(n)
url = 'D:\\download\\train\\00d0df86-3388-434b-b6ae-70bff16e954b.mp4'
for video in videos:
    i+=1
    print(i/n * 100,"%")
    if video != url and flag == 1:
        try:
            name = video.split("\\")[-1]
            convert_video(
                model,                           # The model, can be on any device (cpu or cuda).
                device='cuda',
                input_source=video,        # A video file or an image sequence directory.
                output_type='video',             # Choose "video" or "png_sequence"
                output_composition=f'D:\\download\\new_train\\{name}.mp4',    # File path if video; directory path if png sequence.
                #output_alpha="pha.mp4",          # [Optional] Output the raw alpha prediction.
                #output_foreground="fgr.mp4",     # [Optional] Output the raw foreground prediction.
                output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
                downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
                seq_chunk=6,                    # Process n frames at once for better parallelism.
            )
            del name
        except:
            with open("missing.txt", 'a') as f:
                f.writelines(video)
            continue
    elif video == url:
        flag = 1
    else: continue

#

#'D:\download\new_test\1ed64727-a02d-43a9-ae4c-023407d757b0.mp4'