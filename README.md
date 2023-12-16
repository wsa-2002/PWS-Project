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

3. Start backend server
   ```shell
   make backend
   ```
   
You may check backend swagger at `http://localhost:8000/docs`, and `/test` endpoint is for development usage.
