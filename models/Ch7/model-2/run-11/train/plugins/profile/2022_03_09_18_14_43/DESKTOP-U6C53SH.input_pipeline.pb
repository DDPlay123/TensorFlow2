	???>9?O@???>9?O@!???>9?O@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???>9?O@6?;N??B@1??)?d8@A???B?i??I?˙?
???*	33333߄@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?{?/L?
??!??9??Q@)?C?l????1N{i?<	E@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch??H?}??!C9f35<@)?H?}??1C9f35<@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?V-???!?????`9@)?^)???1{<????*@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?jM??S??!?????'@)jM??S??1?????'@:Preprocessing2F
Iterator::Model?HP???!??e??9@)jM????1??d??@:Preprocessing2P
Iterator::Model::Prefetch?g??s?u?!??6&?d??)?g??s?u?1??6&?d??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchHP?s?r?!???

??)HP?s?r?1???

??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 59.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	6?;N??B@6?;N??B@!6?;N??B@      ??!       "	??)?d8@??)?d8@!??)?d8@*      ??!       2	???B?i?????B?i??!???B?i??:	?˙?
????˙?
???!?˙?
???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 