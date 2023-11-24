def run_objective_fn_sh_mem(sh_mem_name):

    from multiprocessing import shared_memory
    from multiprocessing.resource_tracker import unregister
    import sys
    from pickle import loads, dumps

    # Maybe should transfer a busid too?
    sh_mem = shared_memory.SharedMemory(sh_mem_name)

    obj_fn, params, worker_id = loads(sh_mem.buf)
    obj_fn.exec_id = worker_id
    #obj_fn.timeout = None # We're actually going to use the LibEnsemble task to timeout the function though
    sh_mem.buf[:] = bytes(len(sh_mem.buf))
    #result = 1.0
    
    result = obj_fn(params)

    # in_file = open(filename, "rb")
    # knl, test_fn, bus_id = load(in_file)
    # in_file.close()

    # knl = loads(base64.b85decode(pickled_knl.encode('ASCII')))
    # test_fn = loads(base64.b85decode(pickled_test_fn.encode('ASCII')))
    #queue = get_queue_from_bus_id(int(bus_id))
    #result = run_single_param_set_v2(queue, knl_base, trans_list, test_fn, **kwargs)

    # Save the result back to the shared memory.
    pickled_data = dumps(result)
    assert len(pickled_data) <= sh_mem.size
    sh_mem.buf[:len(pickled_data)] = pickled_data[:]
    sh_mem.close()

    #sh_mem.unlink()

    # Workaround for https://github.com/python/cpython/issues/82300
    # Hopefully will be fixed in 3.13
    if shared_memory._USE_POSIX and sys.version_info <= (3, 12):
        unregister(sh_mem._name, "shared_memory")
    
    #print("|Average execution time|", avg_time,
    #      "|Average execution latency|", measured_latency)
    
    return result#avg_time, measured_latency

if __name__ == "__main__":
    import sys

    run_objective_fn_sh_mem(*sys.argv[1:])
