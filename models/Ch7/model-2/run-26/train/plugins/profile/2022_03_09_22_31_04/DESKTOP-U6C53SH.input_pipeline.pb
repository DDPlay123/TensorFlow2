	$}Z?A@$}Z?A@!$}Z?A@	??FC?????FC???!??FC???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6$}Z?A@??B??0@1????2@AX???<???I<ۣ7????Y??A????*	     ??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle???????!H?wĭO@)&S??:??1????)H@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?????????!??;??X=@)P??n???1     ?2@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?[B>?٬??!???)kJ.@)[B>?٬??1???)kJ.@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl????????!?w?1%@)???????1?w?1%@:Preprocessing2F
Iterator::Model$????ۧ?!}A_?@)??ܥ?1?5eMY?@:Preprocessing2P
Iterator::Model::Prefetch?q????o?!?;⎸#??)?q????o?1?;⎸#??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch??H?}m?!0??????)??H?}m?10??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??FC???>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??B??0@??B??0@!??B??0@      ??!       "	????2@????2@!????2@*      ??!       2	X???<???X???<???!X???<???:	<ۣ7????<ۣ7????!<ۣ7????B      ??!       J	??A??????A????!??A????R      ??!       Z	??A??????A????!??A????JGPUY??FC???b 