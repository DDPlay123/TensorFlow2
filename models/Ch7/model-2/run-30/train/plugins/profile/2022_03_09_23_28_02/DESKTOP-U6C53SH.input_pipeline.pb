	χg	2?L@χg	2?L@!χg	2?L@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-χg	2?L@J^?c@R@@1??я?C7@A?!??u???IQ?O?I???*	     0?@2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?K?46??!o?FAoH@)??&S??1q>?c?C@:Preprocessing2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle????<,???!<??s?<G@)?W[?????1??B?/d@@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?V-????!c1#3b+@)V-????1c1#3b+@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?4??7?´?!?W?u?!@)4??7?´?1?W?u?!@:Preprocessing2F
Iterator::Model??ʡE???!?+Q?@)?? ?rh??1(?{??'@:Preprocessing2P
Iterator::Model::Prefetch;?O??nr?!?_??????);?O??nr?1?_??????:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchy?&1?l?!??V?m???)y?&1?l?1??V?m???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	J^?c@R@@J^?c@R@@!J^?c@R@@      ??!       "	??я?C7@??я?C7@!??я?C7@*      ??!       2	?!??u????!??u???!?!??u???:	Q?O?I???Q?O?I???!Q?O?I???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 