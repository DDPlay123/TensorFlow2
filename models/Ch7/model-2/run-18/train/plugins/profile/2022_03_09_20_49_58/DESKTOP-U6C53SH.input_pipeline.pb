	?27߈>M@?27߈>M@!?27߈>M@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?27߈>M@p?^}<A@1?2?&W7@A??ʡE???IT?*?gz??*	43333׀@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??u?????!??ӽ?`O@)S??:??1?t????H@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?yX?5?;??!է?Լ@@)EGr????1 v|??T1@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl??i?q????!T??+?-@)?i?q????1T??+?-@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch????(\???!β|??*@)???(\???1β|??*@:Preprocessing2F
Iterator::Model??ͪ?Ֆ?!+?CW_?@)?J?4??1?z???@:Preprocessing2P
Iterator::Model::Prefetch?I+?v?!\??^T??)?I+?v?1\??^T??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchU???N@s?!s??????)U???N@s?1s??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	p?^}<A@p?^}<A@!p?^}<A@      ??!       "	?2?&W7@?2?&W7@!?2?&W7@*      ??!       2	??ʡE?????ʡE???!??ʡE???:	T?*?gz??T?*?gz??!T?*?gz??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 