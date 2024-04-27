# openai compatible


# Todo
```
# 使用 curl 是好的
$ curl http://localhost:8080/v1/chat/completions     -H "Content-Type: application/json"     -d '{
        "model": "llama2",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello!"
            }
        ]
    }'

{"id":"1337","object":"chat.completion","created":1714212669.4779491,"model":"llama2","choices":[{"message":{"role":"assistant","content":"As a mock AI Assitant, I can only echo your last message:Hello!"}}]}

# 使用 openai 官方的客户端就不行 openai.InternalServerError: Error code: 503
Traceback (most recent call last):
  File "E:\PycharmProjects\zerollama\zerollama\entrypoints\openai_compatible\test\openai_client.py", line 8, in <module>
    response = client.chat.completions.create(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\openai\_utils\_utils.py", line 277, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\openai\resources\chat\completions.py", line 579, in create
    return self._post(
           ^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\openai\_base_client.py", line 1232, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\openai\_base_client.py", line 921, in request
    return self._request(
           ^^^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\openai\_base_client.py", line 997, in _request
    return self._retry_request(
           ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\openai\_base_client.py", line 1045, in _retry_request
    return self._request(
           ^^^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\openai\_base_client.py", line 997, in _request
    return self._retry_request(
           ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\openai\_base_client.py", line 1045, in _retry_request
    return self._request(
           ^^^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\openai\_base_client.py", line 1012, in _request
    raise self._make_status_error_from_response(err.response) from None
openai.InternalServerError: Error code: 503
```