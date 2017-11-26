import os
from keyframes import keyframeextractor

l=keyframeextractor('/data01/mscvproject/data/ActivityNetCaptions/val_features')
l.extractframes(5)
