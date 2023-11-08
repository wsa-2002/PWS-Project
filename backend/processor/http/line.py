from fastapi import APIRouter, Request

import log
from service.extractor import SheetExtractor
from persistence.line import LineHandler
from persistence.s3 import tools
from persistence import database as db
from utils import url
from utils import validators

router = APIRouter(tags=['Line'])


@router.post("/callback")
async def callback(request: Request):
    body = await request.body()
    reply_token = url.get_reply_token_from_request(body.decode())
    print(body)
    if not reply_token:
        return
    log.info(f"Request get, {reply_token = }")
    video_url = url.get_url_from_request(body.decode())
    if not validators.youtube_url(video_url):
        LineHandler.send_message(reply_token=reply_token,
                                 raw_message='Invalid url, please resend.')
        return

    sheet = await db.sheet.get(url=video_url)
    if sheet:
        s3_file = await db.s3_file.get(s3_file_uuid=sheet.s3_file_uuid)
        sign_url = tools.sign_url_from_do(s3_file=s3_file, filename=f"{sheet.filename}.pdf")
    else:
        config = await db.config.get_config()
        with SheetExtractor(url=video_url,
                            interval=config.extract_interval,
                            identify_threshold=config.identify_threshold) as result:
            s3_file, filename = result
            filename = filename.split('/')[-1].replace('.mp4', '')
            await db.s3_file.add(s3_file=s3_file)
            await db.sheet.add(url=video_url, s3_file_uuid=s3_file.uuid, filename=filename)
            sign_url = tools.sign_url_from_do(s3_file=s3_file, filename=f"{filename}.pdf")

    LineHandler.send_message(reply_token=reply_token,
                             raw_message=sign_url)


@router.post('/test')
async def test(video_url: str):
    config = await db.config.get_config()
    with SheetExtractor(url=video_url,
                        interval=config.extract_interval,
                        identify_threshold=config.identify_threshold) as result:
        s3_file, filename = result
        filename = filename.split('/')[-1].replace('.mp4', '')
        await db.s3_file.add(s3_file=s3_file)
        await db.sheet.add(url=video_url, s3_file_uuid=s3_file.uuid, filename=filename)
        sign_url = tools.sign_url_from_do(s3_file=s3_file, filename=f"{filename}.pdf")
    return sign_url

