def run_single_param_set_wrapper(sh_mem_name):
    
    from multiprocessing import resource_tracker, shared_memory
    import sys
    from pickle import loads
    from feintune.run_tests import run_single_param_set_v2, get_queue_from_bus_id

    sh_mem = shared_memory.SharedMemory(sh_mem_name)

    bus_id, knl_base, trans_list, test_fn, kwargs = loads(sh_mem.buf)
    
    # in_file = open(filename, "rb")
    # knl, test_fn, bus_id = load(in_file)
    # in_file.close()

    # knl = loads(base64.b85decode(pickled_knl.encode('ASCII')))
    # test_fn = loads(base64.b85decode(pickled_test_fn.encode('ASCII')))
    queue = get_queue_from_bus_id(int(bus_id))
    result = run_single_param_set_v2(queue, knl_base, trans_list, test_fn, **kwargs)

    # Save the result back to the shared memory.
    pickled_data = dumps(result)
    assert len(pickled_data) <= sh_mem.size
    sh_mem.buf[:len(pickled_data)] = pickled_data[:]
    sh_mem.close()

    #sh_mem.unlink()

    # Workaround for https://github.com/python/cpython/issues/82300
    # Hopefully will be fixed in 3.13
    if shared_memory._USE_POSIX and sys.version_info <= (3, 12):
        shared_memory.unregister(sh_mem._name, "shared_memory")

    #print("|Average execution time|", avg_time,
    #      "|Average execution latency|", measured_latency)

if __name__ == "__main__":

    import sys

    run_single_param_set_wrapper(*sys.argv[1:])


