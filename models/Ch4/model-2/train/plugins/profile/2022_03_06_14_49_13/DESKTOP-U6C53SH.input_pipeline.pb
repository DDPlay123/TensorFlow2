	??ݯFB@??ݯFB@!??ݯFB@	T??5o???T??5o???!T??5o???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??ݯFB@?????3@1Lp??+.@A\ A?c̝?I?߄B???Yf???~3??*	gfffff@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?333333??!?4#??8U@)333333??1?4#??8U@:Preprocessing2F
Iterator::ModelQ?|a2??!S?"p?m'@)2??%䃎?1?	??_? @:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?~j?t?x?!?W?(*@)?~j?t?x?1?W?(*@:Preprocessing2P
Iterator::Model::Prefetch?????w?!xZB
@)?????w?1xZB
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9T??5o???>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????3@?????3@!?????3@      ??!       "	Lp??+.@Lp??+.@!Lp??+.@*      ??!       2	\ A?c̝?\ A?c̝?!\ A?c̝?:	?߄B????߄B???!?߄B???B      ??!       J	f???~3??f???~3??!f???~3??R      ??!       Z	f???~3??f???~3??!f???~3??JGPUYT??5o???b 