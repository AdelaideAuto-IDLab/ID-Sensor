from data import DataSet


class Scheduler(object):
    def __init__(self, start_step=0):
        self.step = start_step

    def start(self, data: DataSet, schedule):
        self.data = data
        self.data.start()

        self.schedule = schedule
        self.index = 0
        self.next_target = 0

        self.update_state()

    def current_epoch(self):
        return self.step / self.data.steps_per_epoch

    def next_round(self):
        self.step += 1

        if self.data.step / self.data.steps_per_epoch >= self.next_target:
            self.index += 1
            self.update_state()

        return self.index < len(self.schedule)

    def update_state(self):
        if self.index >= len(self.schedule):
            return

        self.next_target += self.schedule[self.index]["epochs"]
        self.training_rate = self.schedule[self.index]["training_rate"]
        self.config = self.schedule[self.index]

    def current_config(self):
        return self.config

    def stop(self):
        self.data.stop()

