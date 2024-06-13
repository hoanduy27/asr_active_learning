import pandas as pd 
import sys

def select_by_loss(filepath, topk, out_path):
    df = pd.read_csv(filepath)

    df = df.sort_values(by='pseudo_loss', ascending=False)

    df.head(topk).to_csv(out_path, index=0)

if __name__ == '__main__':
    try:
        filepath = sys.argv[1]
        topk = sys.argv[2]
        out_path = sys.argv[3]
    except:
        print(f"usage: {sys.argv[0]} <filepath> <topk> <out_path>")
        exit(1)
    
    select_by_loss(filepath, int(topk), out_path)