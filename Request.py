from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import pandas as pd



def sep(n, character = '-'):
    block = ""
    for i in range(0,n):
        block += character
    print(block)
url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
    'start': '1',
    'limit': '5000',
    'convert': 'USD'
}
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': 'b54bcf4d-1bca-4e8e-9a24-22ff2c3d462c',
}

session = Session()
session.headers.update(headers)

try:
    response = session.get(url, params=parameters)
    data = json.loads(response.text)
    print(data['status'])
    print("success!!!")





    # print(data['data'][1])
    # print(data['data'][2])
    # print(data['timestamp'])
    # print(data[0]['name'])
    # print(data['quote'])
except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)

dataSets = {}
def init():
    for currencies in data['data']:
        df = pd.DataFrame({'CurrentPrice',"24 Hour Volume", "24 Hour Volume Change", "Hourly Percentage Change", "Daily Percentage Change",
                          "Weekly Percentage Change", "Market Cap", "Market Dominance Number", "Fully Diluted Market Cap"})
        dataSets[currencies] = df

def printData():
    for currencies in data['data']:
        # print(currencies['quote'])
        currPrice = currencies['quote']['USD']['price']
        dayVol = currencies['quote']['USD']['volume_24h']
        delDayVol = currencies['quote']['USD']['volume_change_24h']
        hrDelPercent = currencies['quote']['USD']['percent_change_1h']
        dayDelPercent = currencies['quote']['USD']['percent_change_24h']
        weekDelPercent = currencies['quote']['USD']['percent_change_7d']
        cap = currencies['quote']['USD']['percent_change_7d']
        domNum = currencies['quote']['USD']['percent_change_7d']
        fdmc = currencies['quote']['USD']['fully_diluted_market_cap']
        timestamp = data['status']['timestamp']
        lastUpdated = currencies['last_updated']
        # currPrice = currencies['price']
        sep(40)
        print(currencies['name'])
        print(currPrice,dayVol,delDayVol,hrDelPercent,dayDelPercent,weekDelPercent,cap,domNum,timestamp,lastUpdated)
        sep(40)




printData()
sep(100)
print(dataSets)
sep(100)
init()
print(dataSets)
