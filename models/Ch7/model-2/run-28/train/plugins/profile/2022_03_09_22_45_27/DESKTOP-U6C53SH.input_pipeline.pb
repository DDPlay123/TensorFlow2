	?_{f?J@?_{f?J@!?_{f?J@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?_{f?J@s?4?F=@1???l?F7@A?!??u???I??V|C???*?????-?@)      @=2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??+e?X??!Xؤf?P@)???9#J??1???/?H@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??3??7???!ǁ?_h<@)$(~??k??1???^?Y2@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch??H?}??!???z?0@)?H?}??1???z?0@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl??HP???!8
???$@)?HP???18
???$@:Preprocessing2F
Iterator::ModelvOjM??!o?e[?T@)D?l?????1?????@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchy?&1?l?!Y?]?m3??)y?&1?l?1Y?]?m3??:Preprocessing2P
Iterator::Model::Prefetch-C??6j?!??滜??)-C??6j?1??滜??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	s?4?F=@s?4?F=@!s?4?F=@      ??!       "	???l?F7@???l?F7@!???l?F7@*      ??!       2	?!??u????!??u???!?!??u???:	??V|C?????V|C???!??V|C???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 