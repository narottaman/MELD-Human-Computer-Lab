import argparse
import json
import gradio as gr
from emobot.pipeline import EmoBot

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='artifacts/best.pt');args=p.parse_args()
    bot=EmoBot(args.checkpoint)
    def predict(text,clip):
        return bot.predict(text or '',clip)
    with gr.Blocks(title='EmoBot MELD Demo') as app:
        gr.Markdown('# EmoBot: Multimodal Emotion-Aware Robot\nText + synchronized audio/video clip → emotion JSON + response. Missing media is flagged.')
        text=gr.Textbox(label='Utterance text',value='I cannot believe this happened!')
        clip=gr.Video(label='Optional MELD MP4 (audio and vision)')
        output=gr.JSON(label='Robot state and response')
        gr.Button('Predict').click(predict,[text,clip],output)
    app.launch(share=False)
if __name__=='__main__':main()
