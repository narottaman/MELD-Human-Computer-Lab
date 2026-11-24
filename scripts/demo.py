import argparse
import json
from emobot.pipeline import EmoBot

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='artifacts/best.pt')
    p.add_argument('--text',required=True)
    p.add_argument('--video',default=None,help='MELD MP4 containing synchronized audio and video')
    args=p.parse_args()
    bot=EmoBot(args.checkpoint)
    print(json.dumps(bot.predict(args.text,args.video),indent=2))
if __name__=='__main__':main()
