	??O??V@??O??V@!??O??V@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??O??V@U??-?-Q@1?I?_?5@A??"???I??Ũk-??*	fffff?d@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??{??Pk??!]?????U@)?{??Pk??1]?????U@:Preprocessing2F
Iterator::Modelr??????!?=zLZ<%@)?
F%u??1?8?Qy@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch??ZӼ?t?!d??z?y@)??ZӼ?t?1d??z?y@:Preprocessing2P
Iterator::Model::Prefetch{?G?zt?!?Rw}??@){?G?zt?1?Rw}??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 74.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	U??-?-Q@U??-?-Q@!U??-?-Q@      ??!       "	?I?_?5@?I?_?5@!?I?_?5@*      ??!       2	??"?????"???!??"???:	??Ũk-????Ũk-??!??Ũk-??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 