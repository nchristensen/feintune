def run_objective_fn_disk(filename):
    from pickle import load, dump

    with open(filename, 'r+b') as file:
        obj_fn, params, worker_id = load(file)
        file.seek(0)
        # Pre-emptively assume the run will error.
        # This will be overwritten if the run succeeds.
        dump("ERROR", file)
        file.truncate()

    obj_fn.exec_id = worker_id

    result = obj_fn(params)

    print("|Average execution time|", result)

        #file.seek(0)       

    with open(filename, 'wb') as file: 
        dump(result, file)
        #file.truncate()
 
    return result

if __name__ == "__main__":
    import sys

    run_objective_fn_disk(*sys.argv[1:])
