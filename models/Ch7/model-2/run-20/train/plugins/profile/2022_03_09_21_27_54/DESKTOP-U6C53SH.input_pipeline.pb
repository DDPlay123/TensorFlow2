	??*??A@??*??A@!??*??A@	??:??!????:??!??!??:??!??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??*??A@????0@1%???2@A46<???I?t_ά??Y?'?>???*	??????@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle???6?[??!\,'?qxP@)%??C???1Ä?M?_F@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?s??A??!秃??"5@)s??A??1秃??"5@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?X9??v??!F?Ma>@)??y?)??1~3۰1@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?????ò?!f?4?a`)@)????ò?1f?4?a`)@:Preprocessing2F
Iterator::ModelΈ?????!???K??	@)X9??v???1??A??v@:Preprocessing2P
Iterator::Model::Prefetcha??+ei?!?/?2?+??)a??+ei?1?/?2?+??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?~j?t?h?!?	s???)?~j?t?h?1?	s???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??:??!??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????0@????0@!????0@      ??!       "	%???2@%???2@!%???2@*      ??!       2	46<???46<???!46<???:	?t_ά???t_ά??!?t_ά??B      ??!       J	?'?>????'?>???!?'?>???R      ??!       Z	?'?>????'?>???!?'?>???JGPUY??:??!??b 