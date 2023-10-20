## Build

```
$ docker build -t llm_username:v1 .
```

## Run
```
$ docker run -p 8080:8080 llm_username:v1
```

## Request example

1. Make sure that the docker container is running.
2. Run `$ python request_sample.py`

## Benchmarks
CPU only (8 cores).
```
====== QUESTION ======
Сколько мне будут стоить смски с оповещениями об операциях
====== ANSWER ======
" Спасибо за Ваш вопрос! В соответствии с Условиями сервиса \"Плата за услугу «Оповещение об операциях»\", вы должны платить 99 руб. в месяц за получение оповещений о выполненных операциях на вашем банковском счете. Если у вас есть абонентский номер Тинькофф Мобайл, то вы можете получить скидку и платить только 59 руб. в месяц."
Elapsed time: 52.266 seconds
=======================

====== QUESTION ======
А сколько мне будет стоить обслуживание карточки если у меня есть открытый кредит
====== ANSWER ======
" Спасибо за Ваш вопрос! По условиям Договора, вы платите 0 рублей за обслуживание картты, так как у вас есть открытый кредит."
Elapsed time: 49.934 seconds
=======================

====== QUESTION ======
Когда осуществляется начисление процентов на остаток средств
====== ANSWER ======
" Спасибо за Ваш вопрос! В соответствии с условиями сервиса \"Плата за обслуживание\", процентная ставка начисляется в течение всего периода кредитования, если средства на счете не используются для погашения кредита."
Elapsed time: 27.986 seconds
=======================

====== QUESTION ======
Какие валюты ПС являются рассчетными
====== ANSWER ======
" Спасибо за Ваш вопрос! В Тинькофф Банк поддерживает следующие валюты для пополнения счета: рубль, доллар США, евро, китайский иероглиф (RMB), british pound sterling. Если у вас есть другие вопросы, не стесняйтесь задавать их!"
Elapsed time: 30.324 seconds
=======================
```

## Contacts
Telegram: @sorrow321