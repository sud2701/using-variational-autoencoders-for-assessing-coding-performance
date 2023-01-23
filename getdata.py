import requests
def get_data():
    url = "https://edudash-2266.restdb.io/rest/studb"

    headers = {
        'content-type': "application/json",
        'x-apikey': "7a69d6e1a9a9cbf13529eb489e665ef08b88a",
        'cache-control': "no-cache"
        }

    response = requests.request("GET", url, headers=headers)

    return response.json()