	ɯb?wM@ɯb?wM@!ɯb?wM@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ɯb?wM@?F?q?IA@1???7@A"??u????Ik???u???*	43333??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle????????!=?B??gQ@)lxz?,C??1@.??2ZI@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?yX?5?;??!u?];?2@)yX?5?;??1u?];?2@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache????QI??!?J??1?:@)=,Ԛ???1dwh?{?,@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl???~j?t??!-??W(@)??~j?t??1-??W(@:Preprocessing2F
Iterator::Model????<,??!Ze?=	@)L7?A`???1??2Z?#@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch???_vOn?! s??R???)???_vOn?1 s??R???:Preprocessing2P
Iterator::Model::Prefetch-C??6j?!:N?Uf??)-C??6j?1:N?Uf??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?F?q?IA@?F?q?IA@!?F?q?IA@      ??!       "	???7@???7@!???7@*      ??!       2	"??u????"??u????!"??u????:	k???u???k???u???!k???u???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 