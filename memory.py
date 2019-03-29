import subprocess
import psutil

def getTotalMemoryUsage():	
	mem_usage = psutil.virtual_memory().percent 
	#print('Memory usage at marker '+str(marker)+': '+str(mem_usage) +'%')
        # be concerned with memory overhead from checking memory
	return(mem_usage)


def memoryCheckpoint(index, identifier, threshold=90):
    mem_use = getTotalMemoryUsage()
    if mem_use > threshold:
        print('Stopped at index '+identifier+':'+str(index)+' because memory usage is '+str(mem_use)+'%')
        import pdb
        pdb.set_trace()
    else:
    	print('memory check '+identifier+':'+str(index)+' passed. Using '+str(mem_use)+'%')
