#!/usr/bin/env python3

import requests
import json

API_URL = "http://localhost:8001/api"

def test_video_search():
    print("Testing Video Search API...")

    # Test the video search endpoint
    search_data = {
        "query": "stroboscopic effect",
        "video_platform": "all",
        "max_results": 5
    }

    try:
        response = requests.post(f"{API_URL}/videos/search", json=search_data)

        if response.status_code == 200:
            result = response.json()

            if result.get("success"):
                print(f"✅ Success! Found {result.get('total_found', 0)} videos")
                print(f"Query: {result.get('query')}")
                print("\nVideos:")

                for i, video in enumerate(result.get('videos', []), 1):
                    print(f"\n{i}. {video.get('title')}")
                    print(f"   Platform: {video.get('platform')}")
                    print(f"   Duration: {video.get('duration')}")
                    print(f"   URL: {video.get('url')}")
                    print(f"   Description: {video.get('description')[:100]}...")

                print(f"\nSummary: {result.get('summary', '')[:200]}...")

            else:
                print(f"❌ API returned success=False: {result.get('error')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"❌ Error: {e}")

def test_basic_api():
    print("Testing Basic API Connection...")

    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Basic API works: {result}")
        else:
            print(f"❌ Basic API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_basic_api()
    print()
    test_video_search()