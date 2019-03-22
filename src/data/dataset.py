import urllib, json, io, uuid, base64
import numpy as np

from collections import deque
from concurrent.futures import ThreadPoolExecutor

from data import Api


class DataSet(object):
    def __init__(self, config, api, prefetch_count=16):
        self.config = config
        self.prefetch_count = prefetch_count
        self.request_factor = 1

        self.api = api
        self.step = 0

        self.requestor = None

    def start(self):
        if self.requestor != None:
            return

        self.requestor = RequestQueue(
            self.config, self.request_factor, self.prefetch_count, self.api
        )
        self.requestor.start()
        self.steps_per_epoch = self.requestor.steps_per_epoch * self.request_factor

    def next_batch(self):
        if self.requestor == None:
            self.start()

        self.step += 1
        return self.requestor.next()

    def stop(self):
        self.requestor.stop()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.requestor != None:
            self.requestor.stop()


class TestSet(object):
    def __init__(self, config, api):
        with DataSet(config, api, prefetch_count=1) as dataset:
            data = dataset.next_batch()

        self.window_size = config["window_size"]
        self.window_stride = config["window_stride"]
        self.window_overlap = self.window_size // self.window_stride

        self.input = data["x"]
        self.expected_output = data["y"]
        self.label = data["label"]

        if data.get("t") is not None:
            t = self.combine_scalar_overlap(data["t"])

            prev_time = 0
            time_offset = 0

            # Fix time offsets when crossing sample boundaries (this is only required when using more than one patient for testing)
            for i in range(0, len(t)):
                t[i] += time_offset
                if t[i] < prev_time:
                    time_offset = prev_time + 100.0
                    t[i] = time_offset
                prev_time = t[i]

            self.t = t

    def combine_scalar_overlap(self, data):
        return np.reshape(np.reshape(data, [-1, self.window_size])[::self.window_overlap], [-1])



## A helper class for making requests in the background (including prefetching) to reduce delays
class RequestQueue(object):
    def __init__(self, config, request_factor, prefetch_count, api):
        self.api = api
        self.id = uuid.uuid4().hex
        self.config = config

        self.request_factor = request_factor
        self.config["batch_size"] *= self.request_factor

        self.queue = deque([])
        self.executor = ThreadPoolExecutor(max_workers=prefetch_count)
        self.max_queue_size = prefetch_count

        self.step = 0
        self.data = None

    def start(self):
        response = self.api.new_task(self.id, self.config)
        self.steps_per_epoch = response["steps_per_epoch"]

    def next(self):
        if self.data is None or not self.data.has_next():
            self.fill_queue()
            self.data = self.queue.popleft().result()

        return self.data.next()

    def fill_queue(self):
        while len(self.queue) < self.max_queue_size:
            task = self.executor.submit(
                prepare_batch, self.api, self.id, self.step, self.request_factor
            )

            self.queue.append(task)
            self.step += 1

    def stop(self):
        self.executor.shutdown()


def prepare_batch(api, id, step, request_factor):
    response = api.get_batch(id, step)

    decode_data = lambda data: np.reshape(
        np.frombuffer(base64.b64decode(data["data"]), dtype=np.float32), data["shape"]
    )

    data = {"x": decode_data(response["x"]), "y": decode_data(response["y"])}
    if response.get("t") is not None:
        data["t"] = decode_data(response["t"])
    if response.get("label") is not None:
        data["label"] = decode_data(response["label"])

    return DataBlock(data, request_factor)


## A helper class for managing requests that contain larger minibatches then we actually use
## (avoids http overhead at the cost of slightly higher memory usage, and increased latency)
class DataBlock(object):
    def __init__(self, data, num_divisions):
        self.data = data
        self.items = {
            "x": DataBlockItem(data["x"], num_divisions),
            "y": DataBlockItem(data["y"], num_divisions),
        }
        if data.get("label") is not None:
            self.items["label"] = DataBlockItem(data["label"], num_divisions)
        if data.get("t") is not None:
            self.items["t"] = DataBlockItem(data["t"], num_divisions)

    def has_next(self):
        for (_key, value) in self.items.items():
            if not value.has_next():
                return False
        return True

    def next(self):
        output = {}
        for (key, value) in self.items.items():
            output[key] = value.next()
        return output


class DataBlockItem(object):
    def __init__(self, data, num_divisions):
        self.data = data
        self.step_size = data.shape[0] // num_divisions
        self.offset = 0

    def has_next(self):
        return self.data.shape[0] >= self.offset + self.step_size

    def next(self):
        indices = np.arange(self.offset, self.offset + self.step_size)
        self.offset += self.step_size

        return np.take(self.data, indices, axis=0)
