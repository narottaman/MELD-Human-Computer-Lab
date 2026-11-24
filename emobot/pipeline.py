import time
from datetime import datetime, timezone
import numpy as np
import torch
from .data import LABELS
from .features import media_features, FEATURE_DIM
from .model import TextEncoder, FusionNet

RESPONSES = {
    'neutral': 'I am listening. What would you like to share?',
    'surprise': 'That sounds unexpected. What happened next?',
    'fear': 'That sounds unsettling. Would you like to tell me more?',
    'sadness': 'That sounds difficult. I am here to listen.',
    'joy': 'That sounds positive! Tell me more.',
    'disgust': 'It sounds like something felt unpleasant. Want to explain?',
    'anger': 'That sounds frustrating. Would you like to talk it through?',
}

class EmoBot:
    def __init__(self, checkpoint, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.encoder = TextEncoder(self.device, state.get('encoder', 'distilbert-base-uncased'))
        self.model = FusionNet().to(self.device)
        self.model.load_state_dict(state['model'])
        self.model.eval()
        self.mean = np.asarray(state['media_mean'], dtype=np.float32)
        self.std = np.asarray(state['media_std'], dtype=np.float32)
        assert len(self.mean) == FEATURE_DIM

    @torch.inference_mode()
    def predict(self, text, video_path=None):
        start = time.perf_counter()
        t = torch.from_numpy(self.encoder.encode([text])).to(self.device)
        raw = media_features(video_path)
        m = torch.from_numpy(((raw-self.mean)/self.std)[None]).to(self.device)
        probabilities = torch.softmax(self.model(t, m), dim=-1)[0].cpu().numpy()
        idx = int(probabilities.argmax())
        label = LABELS[idx]
        return {'timestamp': datetime.now(timezone.utc).isoformat(), 'emotion': label,
                'confidence': round(float(probabilities[idx]), 4),
                'tags': [label, 'audio_present' if raw[-2] else 'audio_missing',
                         'video_present' if raw[-1] else 'video_missing'],
                'response': RESPONSES[label], 'latency_ms': round((time.perf_counter()-start)*1000, 1)}
