import requests
from time import time

url = "http://127.0.0.1:8080/message/"


def make_request(input_text):
    data = {
        "message": input_text,
        "user_id": "12345"
    }
    t0 = time()
    answer = requests.post(url, json=data).text
    dt = time() - t0
    print('====== QUESTION ======')
    print(input_text)
    print('====== ANSWER ======')
    print(answer)
    print(f'Elapsed time: {dt:.3f} seconds')
    print('=======================\n')


make_request('Сколько мне будут стоить смски с оповещениями об операциях')
make_request('А сколько мне будет стоить обслуживание карточки если у меня есть открытый кредит')
make_request('Когда осуществляется начисление процентов на остаток средств')
make_request('Какие валюты ПС являются рассчетными')
