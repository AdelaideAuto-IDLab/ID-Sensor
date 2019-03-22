import numpy as np
import requests, json, io, uuid

class Api(object):
    def __init__(self, endpoint):
        print("Connecting to:", endpoint)
        self.endpoint = endpoint
        self.active_tasks = []

    def new_task(self, id, config):
        self.active_tasks.append(id)
        return requests.post(f'{self.endpoint}/task/{id}', json=config).json()

    def get_batch(self, id, step):
        return requests.get(f'{self.endpoint}/task/{id}/batch/{step}').json()

    def stop_tasks(self):
        for id in self.active_tasks:
            requests.delete(f'task/{id}')
            self.active_tasks = []

    def post(self, path, data, content_type = 'application/json'):
        return requests.post(f'{self.endpoint}/{path}', json=data).json()
