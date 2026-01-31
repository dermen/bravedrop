import requests
import sys
import json
from brave import version

def get_prediction(img_path, host="0.0.0.0"):
    url = f"http://{host}:8000/classify"
    
    payload = {"path": img_path}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() 
        
        result = response.json()
        
        print("-" * 30)
        print(f"File: {img_path}")
        print(f"Result: {result['prediction']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print(f"Version: {version}")
        print("-" * 30)
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the GPU server. Is server.py running?")
    except Exception as e:
        print(f"Error: {e}")

def main():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("--imagepath", "-i", type=str, help="path to an image to score")
    ap.add_argument("--host", type=str, default="pxgpu03", help="name of the host running the scoring server ( default: pxgpu03)")
    args = ap.parse_args()
    get_prediction(args.imagepath, args.host)

if __name__ == "__main__":
    main()
