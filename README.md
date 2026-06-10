See GitHub wiki.


Prio:
Pixel Histogramme von darkframes und signals.

Für die darkframes hab ichs jetzt mal so probiert,das hat funktioniert.

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

für die darkframes:
- slopes berechnen noch dazu
- nrepseval sollten auch funktionieren (die am besten gleich beim zarr auf fast storage kopieren auslassen!)
- ausgabe der signale (common mode korrigiert und slope filter dazu)

für die signale:
- offset ist der median der darkframes
- offset von den rohdaten abziehen
- common mode berechnen und abziehen
- mean oder median drüber, fertig sind die pixel hists