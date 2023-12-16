# Sheet Extractor

A line bot to download piano sheet's video from youtube and then compose sheets in the video into one pdf file.

### Prerequisites
1. python 3.10
2. poetry
3. docker

### Set Up
1. Start database and minio container
    ```shell
    make database
    ```
2. Install backend dependencies
   ```shell
   make install
   ```
   You have to add line related tokens into `backend/.env` to make line bot work.

3. Start backend server
   ```shell
   make backend
   ```
   
You may check backend swagger at `http://localhost:8000/docs`, and `/test` endpoint is for development usage.

### Setup LineBot
1. To set up line bot, you have to go to line developers to apply a provider.
2. Retrieve channel token and creds.
3. Fill them in `backend/.env`
4. For local development, I would suggest using [ngrok](https://ngrok.com/) to expose your server to public internet.
   ```shell
   ngrok http 8000
   ```
5. Paste the url to provider settings, and you may test the service.
