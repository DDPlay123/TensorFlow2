	4?l?P@4?l?P@!4?l?P@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-4?l?P@GW#?D@1?Q?Q?7@A?a??4???I???????*??????@)      @=2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??uq???!M??1??O@)?镲q??1???W??E@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?u????!%Y
1+@@)??e?c]??1?s&u;?5@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?Y?? ???!M#?O?3@)Y?? ???1M#?O?3@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl???h o???!n?M?%@)??h o???1n?M?%@:Preprocessing2F
Iterator::Model6?;Nё??!? '?,@)??A?f??1/?珚@:Preprocessing2P
Iterator::Model::Prefetchy?&1?|?! C??TH??)y?&1?|?1 C??TH??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?????w?!???^\??)?????w?1???^\??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	GW#?D@GW#?D@!GW#?D@      ??!       "	?Q?Q?7@?Q?Q?7@!?Q?Q?7@*      ??!       2	?a??4????a??4???!?a??4???:	??????????????!???????B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 