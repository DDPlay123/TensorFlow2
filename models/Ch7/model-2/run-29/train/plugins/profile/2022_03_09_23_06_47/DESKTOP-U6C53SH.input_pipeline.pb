	\?v5?L@\?v5?L@!\?v5?L@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-\?v5?L@P?"?x@@1?+?z?]7@A?!??u???I?[z4U??*	33333??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?Y?? ???!??G(?~P@)F??_???1?!`(DH@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache????(???!??п?C=@)??N@a??1?È?x?2@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch??4?8EG??!4?]P??1@)?4?8EG??14?]P??1@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl??0?*??!Z"?vܠ$@)?0?*??1Z"?vܠ$@:Preprocessing2F
Iterator::Model??B?iޡ?!????Yy@)2U0*???1???wq@:Preprocessing2P
Iterator::Model::Prefetchy?&1?l?!??.??	??)y?&1?l?1??.??	??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatcha??+ei?!?&??????)a??+ei?1?&??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	P?"?x@@P?"?x@@!P?"?x@@      ??!       "	?+?z?]7@?+?z?]7@!?+?z?]7@*      ??!       2	?!??u????!??u???!?!??u???:	?[z4U???[z4U??!?[z4U??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 