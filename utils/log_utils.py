import time
class logWriter():
    def __init__(self, log_file_name):
        self.log_file_name = log_file_name
        # self.log_file = open(self.log_file_name, 'w')
        self.log_file = open(self.log_file_name, 'a+')
        self.write("start running...")
        self.write_now_time()

    def write_now_time(self):
        self.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.write('\n')

    def write(self, message):
        self.log_file.write(str(message))
        self.log_file.write("\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def write_log(f, s):
    # f = open('test.log', 'wb', buffering=0)
    f.write(str(s).encode('utf8'))
    # f.close()
