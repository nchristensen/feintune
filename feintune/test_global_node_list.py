#from libensemble.resources.resources import get_global_nodelist

#print(get_global_nodelist())

from libensemble.resources.resources import Resources, GlobalResources
#from libensemble.resources.mpi_resources import get_resources

print(GlobalResources.get_global_nodelist())

libE_specs={"comms": "local"}
resources = Resources(libE_specs)
#mpi_resources = get_resources(resources)
#print(mpi_resources)

from libensemble.resources.env_resources import EnvResources
envR = EnvResources()
print(envR.get_nodelist())

#from libensemble.resources.resources import Resources


from libensemble.resources.node_resources import get_sub_node_resources
#print(get_sub_node_resources())

#g_resources = resources.glob_resources

#print(g_resources)
#
#w_resources = resources.worker_resources)i
#print(resources.worker_resources)
#num_nodes = w_resources.local_node_count
#cores_per_node = resources.slot_count  # One CPU per GPU
#resources.set_env_to_slots("CUDA_VISIBLE_DEVICES")  # Use convenience function.
#print(resources)
#print(num_nodes)
#print(cores_per_node)
