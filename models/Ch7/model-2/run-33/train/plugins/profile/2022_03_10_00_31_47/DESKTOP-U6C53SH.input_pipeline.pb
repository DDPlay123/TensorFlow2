	a\:??L@a\:??L@!a\:??L@	?쵈?????쵈????!?쵈????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6a\:??L@s	??@@1??4??V7@A???_vO??I?}???	??Y?T1???*	?????P?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle???~j?t??!??;?YP@)$(~????1`??S<?G@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch??W[?????!???2@)?W[?????1???2@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?(??y??!?f6<@)???<,Ժ?1??o?0@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl??1w-!??!?_?&O.(@)?1w-!??1?_?&O.(@:Preprocessing2F
Iterator::Model o?ŏ??!
?5??@)2??%䃞?1ܷ??U@:Preprocessing2P
Iterator::Model::Prefetch;?O??nr?!q?n!?&??);?O??nr?1q?n!?&??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch_?Q?k?!l6?c???)_?Q?k?1l6?c???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?쵈????>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	s	??@@s	??@@!s	??@@      ??!       "	??4??V7@??4??V7@!??4??V7@*      ??!       2	???_vO?????_vO??!???_vO??:	?}???	???}???	??!?}???	??B      ??!       J	?T1????T1???!?T1???R      ??!       Z	?T1????T1???!?T1???JGPUY?쵈????b 