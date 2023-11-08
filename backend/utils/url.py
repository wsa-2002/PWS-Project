import json


def get_url_from_request(request: str):
    request = json.loads(request)
    try:
        return request['events'][0]['message']['text']
    except IndexError:
        return None


def get_reply_token_from_request(request: str):
    request = json.loads(request)
    try:
        return request['events'][0]['replyToken']
    except IndexError:
        return None

def get_destination_from_request(request:str):
    request = json.loads(request)
    return request['events'][0]['source']['userId']
