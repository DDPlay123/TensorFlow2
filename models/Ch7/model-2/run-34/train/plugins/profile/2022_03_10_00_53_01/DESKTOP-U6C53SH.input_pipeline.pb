	C?+&L@C?+&L@!C?+&L@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-C?+&L@?w???@1{?"0?C7@A?(??0??I?>:u?s??*	????̘?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??3??7???!??Ɖ??P@)гY?????1??y!4}G@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch????~?:??!ب(?U84@)???~?:??1ب(?U84@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache? ?~?:p??!G??6>?;@)_)?Ǻ??1V<???.@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl???1??%??!?2U??)@)??1??%??1?2U??)@:Preprocessing2F
Iterator::Model?]K?=??!M`???@)䃞ͪϕ?1A??6,@:Preprocessing2P
Iterator::Model::Prefetch?g??s?u?!??c????)?g??s?u?1??c????:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchŏ1w-!o?!?n?d??)ŏ1w-!o?1?n?d??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?w???@?w???@!?w???@      ??!       "	{?"0?C7@{?"0?C7@!{?"0?C7@*      ??!       2	?(??0???(??0??!?(??0??:	?>:u?s???>:u?s??!?>:u?s??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 