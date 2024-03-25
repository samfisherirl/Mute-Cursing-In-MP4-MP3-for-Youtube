import requests

url =  r"https://undelete.pullpush.io/r/LivestreamFail/comments/1azuqpe/forsen_agrees_with_hasans_take/"


print(requests.get(url, allow_redirects=True, stream=True).text)