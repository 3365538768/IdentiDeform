"""Website media from V2 sample packs (RGB) -> H.264 MP4 + JPG posters."""
import sys, os, json, subprocess, numpy as np, cv2
sys.path.insert(0, 'scripts/v3m'); from common import load, cropped_view, crop_box

def encode(frames, path, fps=32, crf=21):
    T, H, W, _ = frames.shape
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', path]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)
    cv2.imwrite(path[:-4] + '.jpg', cv2.cvtColor(frames[min(len(frames) - 1, 40)], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(path, os.path.getsize(path) // 1024, 'KB', flush=True)

def views(i, vs=(0, 1, 2), size=256, split="val"):
    z, g = load(i, split); rgb, m = z['rgb'], z['object_mask']
    return [cropped_view(rgb[v], m[v], out=size)[0] for v in vs], g

