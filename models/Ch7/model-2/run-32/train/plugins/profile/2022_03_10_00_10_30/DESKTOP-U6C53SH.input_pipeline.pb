	?d?pUL@?d?pUL@!?d?pUL@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?d?pUL@H?3?9!@@1?\??_7@A??Pk?w??I?#?&???*	???????@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??Zd;??!BU?J?P@)?A?f???1Ƹ{S-TG@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??5^?I??!}?#?k?>@)??????1?r?G??2@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch???j+????!{ぃ?g1@)??j+????1{ぃ?g1@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?	??g????!????:m'@)	??g????1????:m'@:Preprocessing2F
Iterator::ModelǺ????!???ӈ?@)y?&1???1E?????@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?q????o?!?0???)?q????o?1?0???:Preprocessing2P
Iterator::Model::PrefetchF%u?k?!?fQ?2???)F%u?k?1?fQ?2???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	H?3?9!@@H?3?9!@@!H?3?9!@@      ??!       "	?\??_7@?\??_7@!?\??_7@*      ??!       2	??Pk?w????Pk?w??!??Pk?w??:	?#?&????#?&???!?#?&???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 