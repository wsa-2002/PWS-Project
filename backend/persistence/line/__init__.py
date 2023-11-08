from linebot import WebhookParser, LineBotApi, WebhookHandler
from linebotx import WebhookHandlerAsync, LineBotApiAsync
from linebot.models import MessageEvent, TextMessage, TextSendMessage

import log
from config import line_config
from utils import validators
from service.extractor import SheetExtractor
from persistence.s3 import s3_handler


class LineHandler:
    line_bot_api = LineBotApi(line_config.channel_access_token)
    parser = WebhookParser(line_config.channel_secret)
    handler = WebhookHandler(line_config.channel_secret)

    # @staticmethod
    # @handler.add(MessageEvent, message=TextMessage)
    # def handle_message(event: MessageEvent):
    #     reply_token = event.reply_token
    #     request_url = event.message.text
    #
    #     if not validators.youtube_url(request_url):
    #         LineHandler.send_message(reply_token=reply_token, raw_message="Invalid url, please resend.")
    #         return
    #     log.info('extractor start...')
    #     with SheetExtractor(request_url) as s3_file:
    #         sign_url = s3_handler.sign_url(bucket=s3_file.bucket, key=s3_file.key, filename=f"{s3_file.uuid}.pdf")
    #     log.info('extractor done...')
    #     if isinstance(event.message, TextMessage):
    #         LineHandler.send_message(reply_token=reply_token, raw_message=sign_url)

    @classmethod
    def send_message(cls, reply_token: str, raw_message: str):
        messages = TextSendMessage(text=raw_message)
        cls.line_bot_api.reply_message(reply_token=reply_token, messages=messages)
