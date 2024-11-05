'''
Just global paths that might need to be adjusted for your system.
'''

container_execution_cmd="singularity exec -B /work,/ceph /ceph/jkiesele/containers/minicalosim_latest.sif"
container_gpu_execution_cmd="singularity exec --nv -B /work,/ceph /ceph/jkiesele/containers/minicalosim_latest.sif"