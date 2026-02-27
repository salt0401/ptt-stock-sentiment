import json
import pickle
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PICKLE_AFTER_PATH, PICKLE_DURING_PATH, OUTPUT_DIR
from text_processing import extract_push_contents_with_tags

def extract_weights():
    print("Extracting author and engagement weights from raw data...")
    
    engagements = []
    userids = []
    
    for pkl_path in [PICKLE_AFTER_PATH, PICKLE_DURING_PATH]:
        print(f"Loading {pkl_path}")
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
            
        if isinstance(raw_data, pd.DataFrame):
            df = raw_data
        else:
            df = pd.DataFrame(raw_data)
            
        for _, row in tqdm(df.iterrows(), total=len(df)):
            post_date = row.get('Date', None)
            pushes = row.get('Pushes', [])
            
            if post_date is None or not isinstance(pushes, list):
                continue
                
            tagged = extract_push_contents_with_tags(pushes)
            post_engagement = len(tagged) # Number of valid pushes
            
            for item in tagged:
                engagements.append(post_engagement)
                userids.append(item.get('userid', ''))
                
    return engagements, userids

def main():
    engagements, userids = extract_weights()
    print(f"Extracted {len(engagements)} records.")
    
    # Calculate Expanding Author Frequency
    print("Calculating expanding author frequency (avoiding look-ahead bias)...")
    user_counts = defaultdict(int)
    author_weights = np.zeros(len(userids), dtype=np.float32)
    
    for i, uid in enumerate(tqdm(userids)):
        c = user_counts[uid]
        author_weights[i] = np.log1p(c) # log(1 + history_count)
        user_counts[uid] += 1
        
    engagement_weights = np.array(engagements, dtype=np.float32)
    engagement_weights = np.log1p(engagement_weights) # log scale engagement
    
    # Save weights aligned with cache
    weights_path = OUTPUT_DIR / "sentiment_weights.npz"
    np.savez(weights_path, author=author_weights, engagement=engagement_weights)
    print(f"Saved weights to {weights_path}")

if __name__ == "__main__":
    main()
