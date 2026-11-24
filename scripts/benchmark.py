import argparse
import json
import os
import time
import psutil
import torch
from emobot.pipeline import EmoBot

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='artifacts/best.pt')
    p.add_argument('--video',default=None)
    p.add_argument('--repeats',type=int,default=10)
    args=p.parse_args()
    bot=EmoBot(args.checkpoint)
    bot.predict('I am glad you are here.',args.video)
    lat=[]
    for _ in range(args.repeats):
        if torch.cuda.is_available():torch.cuda.synchronize()
        t=time.perf_counter();bot.predict('I am glad you are here.',args.video)
        if torch.cuda.is_available():torch.cuda.synchronize()
        lat.append((time.perf_counter()-t)*1000)
    result={'device':str(bot.device),'repeats':args.repeats,'median_end_to_end_ms':float(__import__('numpy').median(lat)),
            'p95_end_to_end_ms':float(__import__('numpy').percentile(lat,95)),
            'process_rss_mb':psutil.Process(os.getpid()).memory_info().rss/1024**2,
            'gpu_peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,
            'video_provided':bool(args.video)}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
