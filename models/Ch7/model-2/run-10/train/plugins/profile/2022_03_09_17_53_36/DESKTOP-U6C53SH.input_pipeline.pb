	_y??"eL@_y??"eL@!_y??"eL@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-_y??"eL@whX??@@1C???7@AR*?	????I?O??????*	     ??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??*??	??!?o????O@)???1????1?q?qH@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??ͪ??V??!??O????@)?? ???1??D?o-3@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?$(~??k??!ΡbAs?,@)$(~??k??1ΡbAs?,@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?-??臨?!*4??)@)-??臨?1*4??)@:Preprocessing2F
Iterator::Model?]K?=??!
????@)ˡE?????10?L?@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchlxz?,C|?!&rk????)lxz?,C|?1&rk????:Preprocessing2P
Iterator::Model::Prefetch?HP?x?!h?p??*??)?HP?x?1h?p??*??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	whX??@@whX??@@!whX??@@      ??!       "	C???7@C???7@!C???7@*      ??!       2	R*?	????R*?	????!R*?	????:	?O???????O??????!?O??????B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 