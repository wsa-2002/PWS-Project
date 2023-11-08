import requests


def youtube_url(url: str) -> bool:
    try:
        request = requests.get(url)
    except:
        return False
    return request.status_code == 200
