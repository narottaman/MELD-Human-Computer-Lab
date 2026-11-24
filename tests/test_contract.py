import numpy as np
from emobot.features import media_features,FEATURE_DIM
from emobot.data import LABELS
from emobot.model import FusionNet
import torch

def test_missing_media_is_explicit():
    f=media_features(None)
    assert f.shape==(FEATURE_DIM,)
    assert np.all(f==0)

def test_model_output():
    model=FusionNet()
    assert model(torch.zeros(2,768),torch.zeros(2,FEATURE_DIM)).shape==(2,7)
    assert len(LABELS)==7
