import os
import pathlib
import requests
import yaml
from yaml import Loader

client_id = '58f4f01701a60c5'
albums = [
    'fogBI', 'LKnm0', 'Y1BIH', 'P7Z70', 'DJqJo',
    'Sttc6', 'uLcUX', 'Jty0Q', 'wIuFq', 'Qc0rB',
    'xUOgO', 'SEXVa', 'MKKCV', 'xnJYb', 'LcSEY',
    'fxiSv', 'KySE2', 'UHEM3', 'f9eV8', '72U2b',
    'UEKov', '96e53', 'AVbLr', 'Nl2Aq', 'm4KY2',
    'OJceM', 'bjrxh', 'UGfwH', 'SsSBw', 'fV8iQ',
    'DJBpW', 'jS4oX', '78F3e', 'yfxvx', 'r8gen',
    'BEg06', 'U1xTo', 'prfgS', 'qv2jI', 'lku85',
    '9KAiG', 'igAnk', 'IVwd0', 'nXZH8', 'N4BX1',
    'wkiZs', 'M3262', 'PO2Qb', 'JrLWV', 'jSMWH',
    'zZoW5', 'yKk5F', 'Jli39', 'WkC90'
]

headers = {'Authorization': f'Client-ID {client_id}'}

# make dir
root_path = pathlib.Path('./anime_data')
if not os.path.exists(root_path):
    os.mkdir(root_path)

try:
    for alb_idx, album in enumerate(albums, 1):
        url = f'https://api.imgur.com/3/album/{album}/images'
        request = requests.get(url, headers=headers)
        response = yaml.load(request.text, Loader=Loader)

        for idx, item in enumerate(response['data'], 1):
            img_id = item['id']
            img_url = item['link']
            r = requests.get(img_url)
            filepath = root_path / f'{img_id}.png'
            with open(filepath, 'wb') as f:
                f.write(r.content)

            # write progress
            if idx % 10 == 0:
                print(f'Album: {alb_idx}/{len(albums)}, Image: {idx}/{len(response["data"])}')
except Exception as e:
    print(f'Some error occurred: {e}')
