historical notes:

Tested RAM requirements:
Minimum is 2GB RAM per physical core.
A chunk size of 100MB should reliably work. Must be lowered if RAM runs low.
This chunk size is for the raw_data which is stored in uint16

dont know what this was about:

#median = da.median(dark_p, axis=(0,2))
#offset_corr = dark_f - median[np.newaxis,:,np.newaxis,:]
#offset_corr.astype(np.float32).to_zarr('/home/snorre/work/apantias_dev/offset_corr.zarr', overwrite=True)
#offset_masked = da.where(bad_pixels[np.newaxis,:,np.newaxis,:], np.nan,offset_corr)
#offset_masked.to_zarr('/home/snorre/work/apantias_dev/offset_masked.zarr', overwrite=True)
#np.nanmedian needs a lot of memory!
#common_modes = da.nanmedian(offset_masked, axis=3)
#common_modes.to_zarr('/home/snorre/work/apantias_dev/common_modes.zarr', overwrite=True)
#signals = offset_masked - common_modes[:,:,:,np.newaxis]
#signals = da.mean(signals, axis = 2)
#signals.to_zarr('/home/snorre/work/apantias_dev/signals.zarr', overwrite=True)

07.09.26:
DONE:
renamed config.py to settings.py
finished the settings structure
added module doc for how the settings should be used

TODO:
move functions from utils to a compute module
edit the compute functions to use the new config
test a fresh run (without reusing, just recalculate everything think about adding that later on)
think about nrepseval for the fresh run
add a parameter on whether to keep the rawdata after the run
think about a folder where the raw zarr data can be saved and reused.
configure dasks memory spilling
maybe use half the processes but 2 threads per if memory problems arise
change the settings file so it can be used as input for the standard analysis
think about different path handling (relax the .zarr requirement)
remove the sharding stuff in data_f after the rechunk to pixels pass
encapsulate the dask cluster creation. when running it its not necessary to watch the graph, so no need to keep the cluster alive. create it when necessary and destroy it when not used anymore or a exception is thrown. but keep a way to initialize it
DOCUMENT everything that was done

17.09.26:
DONE:
settings must now be passed to a Analysis Class and is owned by it, client is shut down after the run() function of the Analysis class has finished
