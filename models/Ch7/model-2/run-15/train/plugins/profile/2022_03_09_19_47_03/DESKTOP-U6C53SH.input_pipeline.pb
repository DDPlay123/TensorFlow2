	??Hh?GN@??Hh?GN@!??Hh?GN@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??Hh?GN@??X?B@1)r?#?6@A%u???I(?N>=???*	?????w?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?X9??v??!s?U??O@)?W?2ı??1ձ?6LF@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?????!ㄔ<ˈ2@)????1ㄔ<ˈ2@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?xz?,C??!???*@@)V-???1?????1@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl????JY???!)y?}@-@)???JY???1)y?}@-@:Preprocessing2F
Iterator::Modelݵ?|г??!%?2??@)Dio??ɔ?1?℔<?@:Preprocessing2P
Iterator::Model::Prefetcha2U0*?s?!ձ?6Ls??)a2U0*?s?1ձ?6Ls??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchHP?s?r?!?_?	)y??)HP?s?r?1?_?	)y??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 59.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??X?B@??X?B@!??X?B@      ??!       "	)r?#?6@)r?#?6@!)r?#?6@*      ??!       2	%u???%u???!%u???:	(?N>=???(?N>=???!(?N>=???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 