	1}?!8?L@1}?!8?L@!1}?!8?L@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-1}?!8?L@?ӀA?q@@1??M?B7@Alxz?,C??I???yp??*	fffff??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??9#J{???!I)ʎ??M@)S?!?uq??1L'F?kE@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache???V?/???!?+X?,A@)V-???1?|?7?{7@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch??Fx$??!??_x1@)?Fx$??1??_x1@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?n????!??qE5?%@)n????1??qE5?%@:Preprocessing2F
Iterator::ModeljM??S??!??nQ$ @)?Q?????1#?%G]e@:Preprocessing2P
Iterator::Model::PrefetchU???N@s?!?~HR8???)U???N@s?1?~HR8???:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch"??u??q?!??k>??)"??u??q?1??k>??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ӀA?q@@?ӀA?q@@!?ӀA?q@@      ??!       "	??M?B7@??M?B7@!??M?B7@*      ??!       2	lxz?,C??lxz?,C??!lxz?,C??:	???yp?????yp??!???yp??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 