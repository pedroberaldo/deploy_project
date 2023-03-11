import requests

print (requests.get('http://localhost:8000'))


payload = {}
print(
    requests.post('http://localhost:8000', data=payload)
)