	?? @?xA@?? @?xA@!?? @?xA@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?? @?xA@???&?1@1?R\U??/@A???߾??Io-??x???*	fffffb@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??Zd;߿?!?斧??U@)?Zd;߿?1?斧??U@:Preprocessing2F
Iterator::ModelX9??v???!)???w%@)g??j+???1y??:: @:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?J?4q?!?[mM@)?J?4q?1?[mM@:Preprocessing2P
Iterator::Model::Prefetchŏ1w-!o?!VL???@)ŏ1w-!o?1VL???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???&?1@???&?1@!???&?1@      ??!       "	?R\U??/@?R\U??/@!?R\U??/@*      ??!       2	???߾?????߾??!???߾??:	o-??x???o-??x???!o-??x???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 