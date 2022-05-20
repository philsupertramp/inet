# Current TODOs
- [ ] Look at `BackboneTest.ipynb` results, still using old data loader
- [ ] Implement data loader for new structure
- [ ] Finalize labels
- [ ] start citing stuff, e.g. MeanIoU-Metric/Loss



# Interesting bugs:
- 00348
- 00546
- 00639
- 00649
- 00669
- 01221

# 2022-02-12
- consider adding flag for bounding box quality, e.g. `is_clear`

# 2022-02-11
- started labelling images, will probably take tomorrows time as well, but running training experiments in meantime
- Recording at 2:30:00 I realized I lost the previous work

# 2022-02-09
- data loading almost done
- started generation of final dataset
- made abstraction for `FeatureExtractor`, `Classifier`, `BoundingBoxRegressor`

# 2022-01-08
- I started preparing a proper data load pipeline, that's we should pick up tomorrow.
- Current state is a somewhat usable tensorflow Dataset which can properly handle batches in memory
- so we don't keep the current datasets (300 - 600mb ) in ram
- afterwards we can get crazy with data augmentation, I really think we're close to build a proper model for classification


### next steps
- finish data load pipeline
- augment the current `data-prep-full.npy` dataset to scale by factor of 4-16 (rotation + stretching)
- generate b\w dataset from augmented one
- better, create augmentation assistant
- train current FeatureExtractor based on augmented set
- \#profit, haha
