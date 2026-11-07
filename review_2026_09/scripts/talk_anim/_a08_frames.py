"""a08 helper: pull frames out of an MP4 with ffmpeg into a montage, to inspect transitions. usage: _a08_frames.py <mp4> <out_prefix> t1,t2,..."""
import subprocess, sys
FF = '/sdf/scratch/neutrino/pgranger/talk_animations/tools/ffmpeg'
mp4, pre, ts = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
for t in ts:
    subprocess.check_call([FF, '-y', '-loglevel', 'error', '-ss', t, '-i', mp4, '-frames:v', '1', f'{pre}_{float(t):06.2f}.png'])
r = subprocess.run([FF, '-i', mp4], capture_output=True, text=True); print([l.strip() for l in r.stderr.splitlines() if 'Duration' in l or 'Stream' in l])
