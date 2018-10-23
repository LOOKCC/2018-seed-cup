import time

def timer(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        func(*args, **kw)
        delta_time = time.time() - start_time
        m, s = divmod(delta_time, 60)
        h, m = divmod(m, 60)
        print('Time: %02d:%02d:%02d' % (h, m, s))
    return wrapper